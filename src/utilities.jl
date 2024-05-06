# Ultilities function 
@inline function Base.setproperty!(c::AbstractFrankWolfeNode, s::Symbol, v)
    if s in (
        :id,
        :lb,
        :ub,
    )
        # To be bale to convert, we want the function defined in Base
        # not in Core like in Bonobo.Ultilities.jl
        Base.setproperty!(c.std, s, v) 
    else
        Core.setproperty!(c, s, v)
    end
end

"""
Compute relative gap consistently everywhere
"""
function relative_gap(primal, dual)
    gap = if signbit(primal) != signbit(dual)
        Inf
    elseif primal == dual
        0.0
    else
        (primal - dual) / min(abs(primal), abs(dual))
    end
    return gap
end

"""
Check feasibility and boundedness
"""
function check_feasibility(tlmo::TimeTrackingLMO)
    return check_feasibility(tlmo.blmo)
end


"""
Check if at a given index we have an integer constraint respectivily.
"""
function has_integer_constraint(tree::Bonobo.BnBTree, idx::Int)
    return has_integer_constraint(tree.root.problem.tlmo.blmo, idx)
end


"""
Check wether a split is valid. 
"""
function is_valid_split(tree::Bonobo.BnBTree, vidx::Int)
    return is_valid_split(tree, tree.root.problem.tlmo.blmo, vidx)
end


"""
Call this if the active set is empty after splitting.
Remark: This should not happen when using a MIP solver for the nodes!
"""
function restart_active_set(
    node::FrankWolfeNode,
    lmo::FrankWolfe.LinearMinimizationOracle,
    nvars::Int,
)
    direction = Vector{Float64}(undef, nvars)
    Random.rand!(direction)
    v = compute_extreme_point(lmo, direction)
    push!(node.active_set, (1.0, v))
    return node.active_set
end


"""
Split an active set between left and right children.
"""
function split_vertices_set!(
    active_set::FrankWolfe.ActiveSet{T,R},
    tree,
    var::Int,
    local_bounds::IntegerBounds;
    atol=1e-5,
    rtol=1e-5,
) where {T,R}
    x = FrankWolfe.get_active_set_iterate(active_set)
    right_as =
        FrankWolfe.ActiveSet{Vector{Float64},Float64,Vector{Float64}}([], [], similar(active_set.x))
    # indices to remove later from the left active set
    left_del_indices = BitSet()
    for (idx, tup) in enumerate(active_set)
        (λ, a) = tup
        if !is_bound_feasible(local_bounds, a)
            @info "removed"
            push!(left_del_indices, idx)
            continue
        end
        # if variable set to 1 in the atom,
        # place in right branch, delete from left
        if a[var] >= ceil(x[var]) || isapprox(a[var], ceil(x[var]), atol=atol, rtol=rtol)
            push!(right_as, tup)
            push!(left_del_indices, idx)
        elseif a[var] <= floor(x[var]) || isapprox(a[var], floor(x[var]), atol=atol, rtol=rtol)
            # keep in left, don't add to right
        else #floor(x[var]) < a[var] < ceil(x[var])
            # if you are in middle, delete from the left and do not add to the right!
            @warn "Attention! Vertex in the middle."
            push!(left_del_indices, idx)
        end
    end
    deleteat!(active_set, left_del_indices)
    @assert !isempty(active_set)
    @assert !isempty(right_as)
    # renormalize active set and recompute new iterates
    if !isempty(active_set)
        FrankWolfe.active_set_renormalize!(active_set)
        FrankWolfe.compute_active_set_iterate!(active_set)
    end
    if !isempty(right_as)
        FrankWolfe.active_set_renormalize!(right_as)
        FrankWolfe.compute_active_set_iterate!(right_as)
    end
    return (active_set, right_as)
end

function dicg_split_vertices_set!(x, vidx;kwargs...)
    atom_left = copy(x)
    atom_left[vidx] = floor(x[vidx])
    atom_right = zeros(length(x))
    atom_right[vidx] = ceil(x[vidx])
    active_set_left, active_set_right = 
          FrankWolfe.ActiveSet([(1.0, atom_left)]), FrankWolfe.ActiveSet([(1.0, atom_right)])
    return active_set_left, active_set_right
end

"""
Split a discarded vertices set between left and right children.
"""
function split_vertices_set!(
    discarded_set::FrankWolfe.DeletedVertexStorage{T},
    tree,
    var::Int,
    x,
    local_bounds::IntegerBounds;
    atol=1e-5,
    rtol=1e-5,
) where {T}
    right_as = FrankWolfe.DeletedVertexStorage{}(Vector{Float64}[], discarded_set.return_kth)
    # indices to remove later from the left active set
    left_del_indices = BitSet()
    for (idx, vertex) in enumerate(discarded_set.storage)
        if !is_bound_feasible(local_bounds, vertex)
            push!(left_del_indices, idx)
            continue
        end
        if vertex[var] >= ceil(x[var]) || isapprox(vertex[var], ceil(x[var]), atol=atol, rtol=rtol)
            push!(right_as.storage, vertex)
            push!(left_del_indices, idx)
        elseif vertex[var] <= floor(x[var]) ||
               isapprox(vertex[var], floor(x[var]), atol=atol, rtol=rtol)
            # keep in left, don't add to right
        else #floor(x[var]) < vertex[var] < ceil(x[var])
            # if you are in middle, delete from the left and do not add to the right!
            @warn "Attention! Vertex in the middle."
            push!(left_del_indices, idx)
        end
    end
    deleteat!(discarded_set.storage, left_del_indices)
    return (discarded_set, right_as)
end

function is_bound_feasible(bounds::IntegerBounds, v; atol=1e-5)
    for (idx, set) in bounds.lower_bounds
        if v[idx] < set - atol
            return false
        end
    end
    for (idx, set) in bounds.upper_bounds
        if v[idx] > set + atol
            return false
        end
    end
    return true
end

"""
Checks if the branch and bound can be stopped.
By default (in Bonobo) stops then the priority queue is empty. 
"""
function Bonobo.terminated(tree::Bonobo.BnBTree{<:FrankWolfeNode})
    if tree.root.problem.solving_stage == TIME_LIMIT_REACHED
        return true
    end
    absgap = tree.incumbent - tree.lb
    if absgap ≤ tree.options.abs_gap_limit
        return true
    end
    dual_gap = if signbit(tree.incumbent) != signbit(tree.lb)
        Inf
    elseif tree.incumbent == tree.lb
        0.0
    else
        absgap / min(abs(tree.incumbent), abs(tree.lb))
    end
    return isempty(tree.nodes) || dual_gap ≤ tree.options.dual_gap_limit
end


"""
Naive optimization by enumeration.
Default uses binary values.
Otherwise, third argument should be a vector of n sets of possible values for the variables.
"""
function min_via_enum(f, n, values=fill(0:1, n))
    solutions = Iterators.product(values...)
    best_val = Inf
    best_sol = nothing
    for sol in solutions
        sol_vec = collect(sol)
        val = f(sol_vec)
        if best_val > val
            best_val = val
            best_sol = sol_vec
        end
    end
    return best_val, best_sol
end

function sparse_min_via_enum(f, n, k, values=fill(0:1, n))
    solutions = Iterators.product(values...)
    best_val = Inf
    best_sol = nothing
    for sol in solutions
        sol_vec = collect(sol)
        if sum(Int.(iszero.(sol_vec))) >= (n - k)
            val = f(sol_vec)
            if best_val > val
                best_val = val
                best_sol = sol_vec
            end
        end
    end
    return best_val, best_sol
end


# utility function to print the values of the parameters
_value_to_print(::Bonobo.BestFirstSearch) = "Move best bound"
_value_to_print(::PartialStrongBranching) = "Partial strong branching"
_value_to_print(::HybridStrongBranching) = "Hybrid strong branching"
_value_to_print(::Bonobo.MOST_INFEASIBLE) = "Most infeasible"
