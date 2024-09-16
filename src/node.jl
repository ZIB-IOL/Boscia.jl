mutable struct FrankWolfeSolution{Node<:Bonobo.AbstractNode,Value,T<:Real} <:
               Bonobo.AbstractSolution{Node,Value}
    objective::T
    solution::Value
    node::Node
    source::Symbol
end

"""
    NodeInfo

Holds the necessary information of every node.
This needs to be added by every `AbstractNode` as `std::NodeInfo`

This variant is more flexibel than Bonobo.BnBNodeInfo.
"""
mutable struct NodeInfo{T<:Real} 
    id :: Int
    lb :: T
    ub :: T 
end

function Base.convert(::Type{NodeInfo{T}}, std::Bonobo.BnBNodeInfo) where T<:Real
    return NodeInfo(std.id, T(std.lb), T(std.ub)) 
end

"""
    AbtractFrankWolfeNode <: Bonobo.AbstractNode 
"""
abstract type AbstractFrankWolfeNode <: Bonobo.AbstractNode end

"""
    FrankWolfeNode <: AbstractFrankWolfeNode

A node in the branch-and-bound tree storing information for a Frank-Wolfe subproblem.

`std` stores the id, lower and upper bound of the node.
`active_set` store the active set structure.
`local_bounds` instead of storing the complete LMO, it just stores the bounds specific to THIS node.
    All other integer bounds are stored in the root.
'level' stores the level in the tree
'fw_dual_gap_limit' set the tolerance for the dual gap in the FW algorithms
"""
mutable struct FrankWolfeNode{
    AT<:FrankWolfe.ActiveSet,
    DVS<:FrankWolfe.DeletedVertexStorage,
    IB<:IntegerBounds,
    NI<:NodeInfo,
} <: AbstractFrankWolfeNode
    std::NI
    active_set::AT
    discarded_vertices::DVS
    local_bounds::IB
    level::Int
    fw_dual_gap_limit::Float64
    fw_time::Millisecond
    global_tightenings::Int
    local_tightenings::Int
    local_potential_tightenings::Int
    dual_gap::Float64
end

"""
Create the information of the new branching nodes 
based on their parent and the index of the branching variable
"""
function Bonobo.get_branching_nodes_info(tree::Bonobo.BnBTree, node::FrankWolfeNode, vidx::Int)
    if !is_valid_split(tree, vidx)
        error("Splitting on the same index as parent! Abort!")
    end

    # get iterate, primal and lower bound
    x = Bonobo.get_relaxed_values(tree, node)
    primal = tree.root.problem.f(x)
    lower_bound_base = primal - node.dual_gap
    @assert isfinite(lower_bound_base)

    # In case of strong convexity, check if a child can be pruned
    prune_left, prune_right = prune_children(tree, node, lower_bound_base, x, vidx)

    # Split active set
    active_set_left, active_set_right =
        split_vertices_set!(node.active_set, tree, vidx, node.local_bounds)
    discarded_set_left, discarded_set_right =
        split_vertices_set!(node.discarded_vertices, tree, vidx, x, node.local_bounds)

    # Sanity check
    @assert isapprox(sum(active_set_left.weights), 1.0)
    @assert sum(active_set_left.weights .< 0) == 0
    @assert isapprox(sum(active_set_right.weights), 1.0)
    @assert sum(active_set_right.weights .< 0) == 0

    # add new bounds to the feasible region left and right
    # copy bounds from parent
    varbounds_left = copy(node.local_bounds)
    varbounds_right = copy(node.local_bounds)

    if haskey(varbounds_left.upper_bounds, vidx)
        delete!(varbounds_left.upper_bounds, vidx)
    end
    if haskey(varbounds_right.lower_bounds, vidx)
        delete!(varbounds_right.lower_bounds, vidx)
    end
    push!(varbounds_left.upper_bounds, (vidx => floor(x[vidx])))
    push!(varbounds_right.lower_bounds, (vidx => ceil(x[vidx])))

    # compute new dual gap limit
    fw_dual_gap_limit = tree.root.options[:dual_gap_decay_factor] * node.fw_dual_gap_limit
    fw_dual_gap_limit = max(fw_dual_gap_limit, tree.root.options[:min_node_fw_epsilon])

    # Sanity check
    for v in active_set_left.atoms
        if !(v[vidx] <= floor(x[vidx]) + tree.options.atol)
            error("active_set_left\n$(v)\n$vidx, $(x[vidx]), $(v[vidx])")
        end
    end
    for v in discarded_set_left.storage
        if !(v[vidx] <= floor(x[vidx]) + tree.options.atol)
            error("storage left\n$(v)\n$vidx, $(x[vidx]), $(v[vidx])")
        end
    end
    for v in active_set_right.atoms
        if !(v[vidx] >= ceil(x[vidx]) - tree.options.atol)
            error("active_set_right\n$(v)\n$vidx, $(x[vidx]), $(v[vidx])")
        end
    end
    for v in discarded_set_right.storage
        if !(v[vidx] >= ceil(x[vidx]) - tree.options.atol)
            error("storage right\n$(v)\n$vidx, $(x[vidx]), $(v[vidx])")
        end
    end

    # update the LMO
    node_info_left = (
        active_set=active_set_left,
        discarded_vertices=discarded_set_left,
        local_bounds=varbounds_left,
        level=node.level + 1,
        fw_dual_gap_limit=fw_dual_gap_limit,
        fw_time=Millisecond(0),
        global_tightenings=0,
        local_tightenings=0,
        local_potential_tightenings=0,
        dual_gap=NaN,
    )
    node_info_right = (
        active_set=active_set_right,
        discarded_vertices=discarded_set_right,
        local_bounds=varbounds_right,
        level=node.level + 1,
        fw_dual_gap_limit=fw_dual_gap_limit,
        fw_time=Millisecond(0),
        global_tightenings=0,
        local_tightenings=0,
        local_potential_tightenings=0,
        dual_gap=NaN,
    )

    # in case of non trivial domain oracle: Only split if the iterate is still domain feasible
    x_left = FrankWolfe.compute_active_set_iterate!(active_set_left)
    x_right = FrankWolfe.compute_active_set_iterate!(active_set_right)
    domain_oracle = tree.root.options[:domain_oracle]

    domain_right = domain_oracle(x_right)
    domain_left = domain_oracle(x_left)

    nodes = if !prune_left && !prune_right && domain_right && domain_left
        [node_info_left, node_info_right]
    elseif prune_left
        [node_info_right]
    elseif prune_right
        [node_info_left]
    elseif domain_right # x_right in domain
        [node_info_right]
    elseif domain_left # x_left in domain
        [node_info_left]
    else
        warn("No childern nodes can be created.")
    end
    return nodes
end

"""
Use strong convexity to potentially remove one of the children nodes 
"""
function prune_children(tree, node, lower_bound_base, x, vidx)
    prune_left = false
    prune_right = false

    μ = tree.root.options[:strong_convexity]
    if μ > 0
        @debug "Using strong convexity $μ"
        for j in tree.root.problem.integer_variables
            if vidx == j
                continue
            end
            lower_bound_base += μ / 2 * min((x[j] - floor(x[j]))^2, (ceil(x[j]) - x[j])^2)
        end
        new_bound_left = lower_bound_base + μ / 2 * (x[vidx] - floor(x[vidx]))^2
        new_bound_right = lower_bound_base + μ / 2 * (ceil(x[vidx]) - x[vidx])^2
        if new_bound_left > tree.incumbent
            @debug "prune left, from $(node.lb) -> $new_bound_left, ub $(tree.incumbent), lb $(node.lb)"
            prune_left = true
        end
        if new_bound_right > tree.incumbent
            @debug "prune right, from $(node.lb) -> $new_bound_right, ub $(tree.incumbent), lb $(node.lb)"
            prune_right = true
        end
        @assert !(
            (new_bound_left > tree.incumbent + tree.root.options[:dual_gap]) &&
            (new_bound_right > tree.incumbent + tree.root.options[:dual_gap])
        ) "both sides should not be pruned"
    end

    # If both nodes are pruned, when one of them has to be equal to the incumbent.
    # Thus, we have proof of optimality by strong convexity.
    if prune_left && prune_right
        tree.lb = min(new_bound_left, new_bound_right)
    end

    return prune_left, prune_right
end

"""
Computes the relaxation at that node
"""
function Bonobo.evaluate_node!(tree::Bonobo.BnBTree, node::FrankWolfeNode)
    # check that local bounds and global tightening don't conflict
    for (j, ub) in tree.root.global_tightenings.upper_bounds
        if !haskey(node.local_bounds.lower_bounds, j)
            continue
        end
        lb = node.local_bounds.lower_bounds[j]
        if ub < lb
            @debug "local lb $(lb) conflicting with global tightening $(ub)"
            return NaN, NaN
        end
    end
    for (j, lb) in tree.root.global_tightenings.lower_bounds
        if !haskey(node.local_bounds.upper_bounds, j)
            continue
        end
        ub = node.local_bounds.upper_bounds[j]
        if ub < lb
            @debug "local ub $(ub) conflicting with global tightening $(lb)"
            return NaN, NaN
        end
    end

    # build up node LMO
    build_LMO(
        tree.root.problem.tlmo,
        tree.root.problem.integer_variable_bounds,
        node.local_bounds,
        tree.root.problem.integer_variables,
    )

    # check for feasibility and boundedness
    status = check_feasibility(tree.root.problem.tlmo)
    if status == INFEASIBLE
        @debug "Problem at node $(node.id) infeasible"
        return NaN, NaN
    end
    if status == UNBOUNDED
        error("Feasible region unbounded! Please check your constraints!")
        return NaN, NaN
    end

    # Check feasibility of the iterate
    active_set = node.active_set
    x = FrankWolfe.compute_active_set_iterate!(node.active_set)
    @assert is_linear_feasible(tree.root.problem.tlmo, x)
    for (_, v) in node.active_set
        @assert is_linear_feasible(tree.root.problem.tlmo, v)
    end

    # time tracking FW
    time_ref = Dates.now()
    domain_oracle = tree.root.options[:domain_oracle]

    x, primal, dual_gap, active_set = solve_frank_wolfe(
        tree.root.options[:variant],
        tree.root.problem.f,
        tree.root.problem.g,
        tree.root.problem.tlmo,
        node.active_set;
        epsilon=node.fw_dual_gap_limit,
        max_iteration=tree.root.options[:max_fw_iter],
        line_search=tree.root.options[:lineSearch],
        lazy=tree.root.options[:lazy],
        lazy_tolerance=tree.root.options[:lazy_tolerance],
        add_dropped_vertices=tree.root.options[:use_shadow_set],
        use_extra_vertex_storage=tree.root.options[:use_shadow_set],
        extra_vertex_storage=node.discarded_vertices,
        callback=tree.root.options[:callback],
        verbose=tree.root.options[:fwVerbose],
    )

    node.fw_time = Dates.now() - time_ref
    node.dual_gap = dual_gap

    # update active set of the node
    node.active_set = active_set

    # tightening bounds at node level
    dual_tightening(tree, node, x, dual_gap)

    # tightening the global bounds
    store_data_global_tightening(tree, node, x, dual_gap)
    global_tightening(tree, node)

    lower_bound = primal - dual_gap
    # improvement of the lower bound using strong convexity
    lower_bound = tightening_strong_convexity(tree, x, lower_bound)

    # Found an upper bound
    if is_integer_feasible(tree, x)
        node.ub = primal
        @debug "Node $(node.id) has an integer solution."
        return lower_bound, primal
        # Sanity check: If the incumbent is better than the lower bound of the root node
        # and the root node is not integer feasible, something is off!
    elseif node.id == 1
        @debug "Lower bound of root node: $(lower_bound)"
        @debug "Current incumbent: $(tree.incumbent)"
        @assert lower_bound <= tree.incumbent + node.fw_dual_gap_limit "lower_bound <= tree.incumbent + node.fw_dual_gap_limit : $(lower_bound) <= $(tree.incumbent + node.fw_dual_gap_limit)"
    end

    # Call heuristic 
    run_heuristics(tree, x, tree.root.options[:heuristics])

    return lower_bound, NaN
end

"""
Tightening of the bounds at node level. Children node inherit the updated bounds.
"""
function dual_tightening(tree, node, x, dual_gap)
    if tree.root.options[:dual_tightening] && isfinite(tree.incumbent)
        grad = similar(x)
        tree.root.problem.g(grad, x)
        num_tightenings = 0
        num_potential_tightenings = 0
        μ = tree.root.options[:strong_convexity]
        for j in tree.root.problem.integer_variables
            lb_global = get(tree.root.problem.integer_variable_bounds, (j, :greaterthan), -Inf)
            ub_global = get(tree.root.problem.integer_variable_bounds, (j, :lessthan), Inf)
            lb = get(node.local_bounds.lower_bounds, j, lb_global)
            ub = get(node.local_bounds.upper_bounds, j, ub_global)
            @assert lb >= lb_global
            @assert ub <= ub_global
            if lb ≈ ub
                # variable already fixed
                continue
            end
            gj = grad[j]
            safety_tolerance = 2.0
            rhs =
                tree.incumbent - tree.root.problem.f(x) +
                safety_tolerance * dual_gap +
                sqrt(eps(tree.incumbent))
            if ≈(x[j], lb, atol=tree.options.atol, rtol=tree.options.rtol)
                if !isapprox(gj, 0, atol=1e-5)
                    num_potential_tightenings += 1
                end
                if gj > 0
                    Mlb = 0
                    bound_tightened = true
                    @debug "starting tightening ub $(rhs)"
                    while 0.99 * (Mlb * gj + μ / 2 * Mlb^2) <= rhs
                        Mlb += 1
                        if lb + Mlb - 1 == ub
                            bound_tightened = false
                            break
                        end
                    end
                    if bound_tightened
                        new_bound = lb + Mlb - 1
                        @debug "found UB tightening $ub -> $new_bound"
                        node.local_bounds[j, :lessthan] = new_bound
                        num_tightenings += 1
                        if haskey(tree.root.problem.integer_variable_bounds, (j, :lessthan))
                            @assert node.local_bounds[j, :lessthan] <=
                                    tree.root.problem.integer_variable_bounds[j, :lessthan]
                        end
                    end
                end
            elseif ≈(x[j], ub, atol=tree.options.atol, rtol=tree.options.rtol)
                if !isapprox(gj, 0, atol=1e-5)
                    num_potential_tightenings += 1
                end
                if gj < 0
                    Mub = 0
                    bound_tightened = true
                    @debug "starting tightening lb $(rhs)"
                    while -0.99 * (Mub * gj + μ / 2 * Mub^2) <= rhs
                        Mub += 1
                        if ub - Mub + 1 == lb
                            bound_tightened = false
                            break
                        end
                    end
                    if bound_tightened
                        new_bound = ub - Mub + 1
                        @debug "found LB tightening $lb -> $new_bound"
                        node.local_bounds[j, :greaterthan] = new_bound
                        num_tightenings += 1
                        if haskey(tree.root.problem.integer_variable_bounds, (j, :greaterthan))
                            @assert node.local_bounds[j, :greaterthan] >=
                                    tree.root.problem.integer_variable_bounds[j, :greaterthan]
                        end
                    end
                end
            end
        end
        @debug "# tightenings $num_tightenings"
        node.local_tightenings = num_tightenings
        node.local_potential_tightenings = num_potential_tightenings
    end
end

"""
Save the gradient of the root solution (i.e. the relaxed solution) and the 
corresponding lower and upper bounds.
"""
function store_data_global_tightening(tree, node, x, dual_gap)
    if tree.root.options[:global_dual_tightening] && node.std.id == 1
        @debug "storing root node info for tightening"
        grad = similar(x)
        tree.root.problem.g(grad, x)
        safety_tolerance = 2.0
        tree.root.global_tightening_rhs[] = -tree.root.problem.f(x) + safety_tolerance * dual_gap
        for j in tree.root.problem.integer_variables
            if haskey(tree.root.problem.integer_variable_bounds.upper_bounds, j)
                ub = tree.root.problem.integer_variable_bounds[j, :lessthan]
                if ≈(x[j], ub, atol=tree.options.atol, rtol=tree.options.rtol) && grad[j] < 0
                    tree.root.global_tightening_root_info.upper_bounds[j] = (grad[j], ub)
                end
            end
            if haskey(tree.root.problem.integer_variable_bounds.lower_bounds, j)
                lb = tree.root.problem.integer_variable_bounds[j, :greaterthan]
                if ≈(x[j], lb, atol=tree.options.atol, rtol=tree.options.rtol) && grad[j] > 0
                    tree.root.global_tightening_root_info.lower_bounds[j] = (grad[j], lb)
                end
            end
        end
    end
end

"""
Use the gradient of the root node to tighten the global bounds.
"""
function global_tightening(tree, node)
    # new incumbent: check global fixings
    if tree.root.options[:global_dual_tightening] && tree.root.updated_incumbent[]
        num_tightenings = 0
        rhs = tree.incumbent + tree.root.global_tightening_rhs[]
        @assert isfinite(rhs)
        for (j, (gj, lb)) in tree.root.global_tightening_root_info.lower_bounds
            ub = get(tree.root.problem.integer_variable_bounds.upper_bounds, j, Inf)
            ub_new = get(tree.root.global_tightening_root_info.upper_bounds, j, Inf)
            ub = min(ub, ub_new)
            Mlb = 0
            bound_tightened = true
            lb = lb
            while Mlb * gj <= rhs
                Mlb += 1
                if lb + Mlb - 1 == ub
                    bound_tightened = false
                    break
                end
            end
            if bound_tightened
                new_bound = lb + Mlb - 1
                @debug "found global UB tightening $ub -> $new_bound"
                if haskey(tree.root.global_tightenings.upper_bounds, j)
                    if tree.root.global_tightenings.upper_bounds[j] != new_bound
                        num_tightenings += 1
                    end
                else
                    num_tightenings += 1
                end
                tree.root.global_tightenings.upper_bounds[j] = new_bound
            end
        end
        for (j, (gj, ub)) in tree.root.global_tightening_root_info.upper_bounds
            lb = get(tree.root.problem.integer_variable_bounds.lower_bounds, j, -Inf)
            lb_new = get(tree.root.global_tightening_root_info.lower_bounds, j, -Inf)
            lb = max(lb, lb_new)
            Mub = 0
            bound_tightened = true
            ub = ub
            while -Mub * gj <= rhs
                Mub += 1
                if ub - Mub + 1 == lb
                    bound_tightened = false
                    break
                end
            end
            if bound_tightened
                new_bound = ub - Mub + 1
                @debug "found global LB tightening $lb -> $new_bound"
                if haskey(tree.root.global_tightenings.lower_bounds, j)
                    if tree.root.global_tightenings.lower_bounds[j] != new_bound
                        num_tightenings += 1
                    end
                else
                    num_tightenings += 1
                end
                tree.root.global_tightenings.lower_bounds[j] = new_bound
            end
        end
        node.global_tightenings = num_tightenings
    end
end

"""
Tighten the lower bound using strong convexity of the objective.
"""
function tightening_strong_convexity(tree, x, lower_bound)
    μ = tree.root.options[:strong_convexity]
    if μ > 0
        @debug "Using strong convexity $μ"
        strong_convexity_bound = lower_bound
        num_fractional = 0
        for j in tree.root.problem.integer_variables
            if x[j] > floor(x[j]) + 1e-6 && x[j] < ceil(x[j]) - 1e-6
                num_fractional += 1
                new_left_increment = μ / 2 * (x[j] - floor(x[j]))^2
                new_right_increment = μ / 2 * (ceil(x[j]) - x[j])^2
                new_increment = min(new_left_increment, new_right_increment)
                strong_convexity_bound += new_increment
            end
        end
        @debug "Strong convexity: $lower_bound -> $strong_convexity_bound"
        @assert num_fractional == 0 || strong_convexity_bound > lower_bound
        lower_bound = strong_convexity_bound
    end
    return lower_bound
end


"""
Returns the solution vector of the relaxed problem at the node
"""
function Bonobo.get_relaxed_values(tree::Bonobo.BnBTree, node::FrankWolfeNode)
    return copy(FrankWolfe.get_active_set_iterate(node.active_set))
end

tree_lb(tree::Bonobo.BnBTree) = min(tree.lb, tree.incumbent)
