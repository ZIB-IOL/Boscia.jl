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

function split_pre_computed_set!(
    x, 
    pre_computed_set::Vector, 
    tree, vidx::Int,
    local_bounds::IntegerBounds;
    atol = 1e-5, 
    rtol = 1e-5, 
    kwargs...
)
    pre_computed_set_left = []
    pre_computed_set_right = []
    for atom in pre_computed_set
        if !is_bound_feasible(local_bounds, atom)
            continue
        end
        if atom[vidx] >= ceil(x[vidx]) || isapprox(atom[vidx], ceil(x[vidx]), atol = atol, rtol = rtol)
            push!(pre_computed_set_right, atom)
        elseif atom[vidx] <= floor(x[vidx]) || isapprox(atom[vidx], floor(x[vidx]), atol = atol, rtol = rtol)
            push!(pre_computed_set_left, atom)
        end
    end
    return pre_computed_set_left, pre_computed_set_right
end

"""
Default starting point function which generates a random vertex
"""
function trivial_build_dicg_start_point(blmo::BoundedLinearMinimizationOracle)
    n, _ = get_list_of_variables(blmo)
    d = ones(n)
    x0 = FrankWolfe.compute_extreme_point(blmo, d)
    return x0
end

function dicg_start_point_initialize(
    lmo::TimeTrackingLMO, 
    active_set::FrankWolfe.ActiveSet{T, R},
    pre_computed_set,
    build_dicg_start_point; 
    domain_oracle = _trivial_domain
 ) where {T,R}
    if lmo.ncalls == 0
        return FrankWolfe.get_active_set_iterate(active_set)
    end
    if pre_computed_set === nothing
        x0 = build_dicg_start_point(lmo.blmo)
    else
        if !isempty(pre_computed_set)
            # We pick a point by averaging the pre_computed_atoms as warm-start.  
            num_pre_computed_set = length(pre_computed_set)
            x0 = sum(pre_computed_set) / num_pre_computed_set
            if !domain_oracle(x0)
                x0 = build_dicg_start_point(lmo.blmo)
            end
        else
            # We pick a random point.
            x0 = build_dicg_start_point(lmo.blmo)
        end
    end
    return x0
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

"""
Build a new start point and active set in case the split active set
does not lead to a domain feasible iterate.
First, try filtering the active set by the domain oracle.
If all vertices are domain infeasible, solve the projection problem
1/2 * ||x - x*||_2^2 
where x* is a domain- and bound-feasible point provided by the user.
"""
function build_active_set_by_domain_oracle(
    active_set::FrankWolfe.ActiveSet{T,R},
    tree,
    local_bounds::IntegerBounds,
    node;
    atol=1e-5,
    rtol=1e-5,
) where {T,R}
    # Check if node problem is even feasible
    build_LMO(
        tree.root.problem.tlmo,
        tree.root.problem.integer_variable_bounds,
        local_bounds,
        tree.root.problem.integer_variables,
    )
    status = check_feasibility(tree.root.problem.tlmo)
    if status == INFEASIBLE
        build_LMO(
            tree.root.problem.tlmo,
            tree.root.problem.integer_variable_bounds,
            node.local_bounds,
            tree.root.problem.integer_variables,
        )
        active_set.empty!()
        return active_set
    end
    # Filtering
    del_indices = BitSet()
    for (idx, tup) in enumerate(active_set)
        (λ, a) = tup
        if !tree.root.options[:domain_oracle](a)
            push!(del_indices, idx)
        end
    end
    # At least one vertex is domain feasible.
    if length(del_indices) < length(active_set.weights)
        deleteat!(active_set, del_indices)
        @assert !isempty(active_set)
        FrankWolfe.active_set_renormalize!(active_set)
        FrankWolfe.compute_active_set_iterate!(active_set)
    # No vertex is domain feasible
    else
        x_star = tree.root.options[:find_domain_point](local_bounds)
        # No domain feasible point can be build.
        # Node can be pruned.
        if x_star === nothing
            deleteat!(active_set, del_indices)
        else
            inner_f(x) = 1/2 * LinearAlgebra.norm(x - x_star)^2

            function inner_grad!(storage, x)
                storage .= x - x_star
                return storage
            end

            function build_inner_callback(tree)
                return function inner_callback(state, active_set, kwargs...)
                    # stop as soon as we find a domain feasible point.
                    if tree.root.options[:domain_oracle](state.x)
                        return false
                    end
                end
            end
            inner_callback = build_inner_callback(tree)

            x, _, _, _, _, active_set = FrankWolfe.blended_pairwise_conditional_gradient(
                inner_f,
                inner_grad!,
                tree.root.problem.tlmo,
                active_set,
                callback=inner_callback,
                lazy=true,
            )
            @assert tree.root.options[:domain_oracle](x)
        end
    end
    build_LMO(
        tree.root.problem.tlmo,
        tree.root.problem.integer_variable_bounds,
        node.local_bounds,
        tree.root.problem.integer_variables,
    )
    return active_set
end

"""
Check if a given point v satisfies the given bounds.
"""
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
    if tree.root.problem.solving_stage in (TIME_LIMIT_REACHED, NODE_LIMIT_REACHED)
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

function min_via_enum_simplex(f, n, N, values=fill(0:1,n))
    solutions = Iterators.product(values...)
    best_val = Inf
    best_sol = nothing
    for sol in solutions
        sol_vec = collect(sol)
        if sum(sol_vec) > N 
            continue
        end
        val = f(sol_vec)
        if best_val > val
            best_val = val
            best_sol = sol_vec
        end
    end
    return best_val, best_sol
end

function min_via_enum_prob_simplex(f, n, N, values=fill(0:1,n))
    solutions = Iterators.product(values...)
    best_val = Inf
    best_sol = nothing
    for sol in solutions
        sol_vec = collect(sol)
        if sum(sol_vec) != N 
            continue
        end
        val = f(sol_vec)
        if best_val > val
            best_val = val
            best_sol = sol_vec
        end
    end
    return best_val, best_sol
end


# utility function to print the values of the parameters
_value_to_print(::Bonobo.BestFirstSearch) = "Move best bound"
_value_to_print(::PartialStrongBranching) = "Partial strong branching"
_value_to_print(::HybridStrongBranching) = "Hybrid strong branching"
_value_to_print(::Bonobo.MOST_INFEASIBLE) = "Most infeasible"
