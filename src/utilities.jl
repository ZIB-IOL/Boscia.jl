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


function save_results(
    result::Dict{Symbol, Any},
    settings::String,
    example_name::String,
    seed::UInt64,
    file_name::String,
    over_write::Bool
    )
    
    seed = string(seed)
    l1 = []# save all keys with one entry only
    l2 = []# save all vector results of length equal to that of result[:list_ub]
    l3 = []# save all vector results of length equal to that of lmo_calls_per_layer
    for key in keys(result)
        if length(result[key]) == 1 || isa(result[key], String)
            push!(l1, key)
        elseif length(result[key]) == length(result[:list_ub])
            push!(l2, key)
        elseif length(result[key]) == length(result[:lmo_calls_per_layer])
            push!(l3, key)
        end
    end
    l11 = Dict(string(key) => result[key] for key in l1)
    l22 = Dict(string(key) => result[key] for key in l2)
    l33 = Dict(string(key) => result[key] for key in l3)
    l11 = DataFrame(l11)
    l11[:, :example_name] .= example_name
    l11[:, :seed] .= seed
    l11[:, :settings] .= settings

    l22 = DataFrame(l22)
    l22[:, :settings] .= settings
    l22[:, :example_name] .= example_name
    l22[:, :seed] .= seed

    l33 = DataFrame(l33)
    l33[:, :settings] .= settings
    l33[:, :example_name] .= example_name
    l33[:, :seed] .= seed

    file_name1 = "./results/" * file_name * "_summary.csv"

    
    if over_write# will always over write file if true
        append = false
    else
        if isfile(file_name1)# using this method the first line of the file will have column names
            append = false
        else
            append = true
        end
    end
    CSV.write(file_name1, l11, append= append)

    file_name2 = "./results/" * file_name * ".csv"

    CSV.write(file_name2, l22, append= append)
    file_name3 = "./results/" * file_name * "_layers.csv"
    CSV.write(file_name3, l33, append= append)
end

# utility function to print the values of the parameters
_value_to_print(::Bonobo.BestFirstSearch) = "Move best bound"
_value_to_print(::PartialStrongBranching) = "Partial strong branching"
_value_to_print(::HybridStrongBranching) = "Hybrid strong branching"
_value_to_print(::Bonobo.MOST_INFEASIBLE) = "Most infeasible"
_value_to_print(::PSEUDO_COST) = "Pseudo Cost"