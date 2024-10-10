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
        @warn "No childern nodes can be created."
        Vector{typeof(node_info_left)}()
    end
    return nodes
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
    lower_bound = tightening_lowerbound(tree, node, x, lower_bound)

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
Returns the solution vector of the relaxed problem at the node
"""
function Bonobo.get_relaxed_values(tree::Bonobo.BnBTree, node::FrankWolfeNode)
    return copy(FrankWolfe.get_active_set_iterate(node.active_set))
end

tree_lb(tree::Bonobo.BnBTree) = min(tree.lb, tree.incumbent)
