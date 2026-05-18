"""
Computes the relaxation at that node
"""
function evaluate_node!(tree::BnBTree, node::FrankWolfeNode)
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

    decomposition_invariant_starting_point = nothing
    if !(typeof(tree.root.options[:variant]) <: DecompositionInvariant)
        # Check feasibility of the iterate
        active_set = node.active_set
        x = FrankWolfe.compute_active_set_iterate!(node.active_set)
        @assert is_linear_feasible(tree.root.problem.tlmo, x)
        for (_, v) in node.active_set
            @assert is_linear_feasible(tree.root.problem.tlmo, v)
        end
    else
        if node.id == 1 && tree.root.options[:start_solution] !== nothing
            decomposition_invariant_starting_point = tree.root.options[:start_solution]
        elseif tree.root.options[:find_domain_point] !== _trivial_domain_point
            decomposition_invariant_starting_point =
                tree.root.options[:find_domain_point](node.local_bounds)
            if decomposition_invariant_starting_point === nothing
                @debug "Node $(node.id) is infeasible: no domain-feasible starting point found."
                return NaN, NaN
            end
        end
    end

    if tree.root.options[:propagate_bounds] !== nothing
        tree.root.options[:propagate_bounds](tree, node)
    end

    # time tracking FW
    time_ref = Dates.now()
    domain_oracle = tree.root.options[:domain_oracle]

    x, primal, dual_gap, fw_status, atoms_set = solve_frank_wolfe(
        tree.root.options[:variant],
        tree.root.problem.f,
        tree.root.problem.g,
        tree.root.problem.tlmo,
        node.active_set;
        epsilon=node.fw_dual_gap_limit,
        max_iteration=tree.root.options[:max_fw_iter],
        line_search=tree.root.options[:line_search],
        lazy=tree.root.options[:lazy],
        lazy_tolerance=tree.root.options[:lazy_tolerance],
        add_dropped_vertices=tree.root.options[:use_shadow_set],
        use_extra_vertex_storage=tree.root.options[:use_shadow_set],
        extra_vertex_storage=node.discarded_vertices,
        callback=tree.root.options[:callback],
        verbose=tree.root.options[:fw_verbose],
        timeout=tree.root.options[:fw_timeout],
        pre_computed_set=node.pre_computed_set,
        domain_oracle=domain_oracle,
        decomposition_invariant_starting_point=decomposition_invariant_starting_point,
    )

    if tree.root.options[:fw_verbose]
        @show fw_status
    end

    if typeof(atoms_set).name.wrapper == FrankWolfe.ActiveSet
        # update active set of the node
        node.active_set = atoms_set
    else
        node.pre_computed_set = atoms_set
        node.active_set = FrankWolfe.ActiveSet([(1.0, x)])
        # update set of computed atoms and active set
        if isa(x, AbstractVector)
            node.pre_computed_set = atoms_set
            node.active_set = FrankWolfe.ActiveSet([(1.0, x)])
        else
            @debug "x is not a vector, returning NaN, x: $x"
            return NaN, NaN
        end
    end

    node.fw_time = Dates.now() - time_ref
    node.dual_gap = dual_gap

    # tightening bounds at node level
    dual_tightening(tree, node, x, dual_gap)

    # tightening the global bounds
    store_data_global_tightening(tree, node, x, dual_gap)
    global_tightening(tree, node)

    lower_bound = primal - dual_gap
    # tighten the lower bound if the objective is always integral
    lower_bound = tree.root.options[:integral_objective] ? ceil(lower_bound) : lower_bound
    # improvement of the lower bound using strong convexity
    lower_bound = tightening_lowerbound(tree, node, x, lower_bound)

    # Call heuristic 
    run_heuristics(tree, x, tree.root.options[:heuristics])

    # Found an upper bound
    if is_integer_feasible(tree, x)
        node.ub = primal
        @debug "Node $(node.id) has an integer solution."
        return lower_bound, primal
        # Sanity check: If the incumbent is better than the lower bound of the root node
        # and the root node is not integer feasible, something is off!
    elseif node.id == 1 && !tree.root.options[:ignore_lower_bound]
        @debug "Lower bound of root node: $(lower_bound)"
        @debug "Current incumbent: $(tree.incumbent)"
        @assert lower_bound <= tree.incumbent + dual_gap "lower_bound <= tree.incumbent + dual_gap : $(lower_bound) <= $(tree.incumbent + dual_gap)"
    end


    if tree.root.options[:ignore_lower_bound]
        return -Inf, NaN
    end

    return lower_bound, NaN
end

"""
Returns the solution vector of the relaxed problem at the node
"""
function get_relaxed_values(tree::BnBTree, node::FrankWolfeNode)
    return copy(FrankWolfe.get_active_set_iterate(node.active_set))
end

tree_lb(tree::BnBTree) = min(tree.lb, tree.incumbent)

"""
    initialize(; kwargs...)

Initialize the branch and bound framework with the the following arguments.
Later it can be dispatched on `BnBTree{Node, Root, Solution}` for various methods.

# Keyword arguments
- `traverse_strategy` [`BestFirstSearch`] currently the only supported traverse strategy is [`BestFirstSearch`](@ref). Should be an [`AbstractTraverseStrategy`](@ref)
- `branch_strategy` [`FIRST`] currently the only supported branching strategies are [`FIRST`](@ref) and [`MOST_INFEASIBLE`](@ref). Should be an [`AbstractBranchStrategy`](@ref)
- `atol` [1e-6] the absolute tolerance to check whether a value is discrete
- `rtol` [1e-6] the relative tolerance to check whether a value is discrete
- `Node` [`DefaultNode`](@ref) can be special structure which is used to store all information about a node. 
    - needs to have `AbstractNode` as the super type
    - needs to have `std :: BnBNodeInfo` as a field (see [`BnBNodeInfo`](@ref))
- `Solution` [`DefaultSolution`](@ref) stores the node and several other information about a solution
- `root` [`nothing`] the information about the root problem. The type can be used for dispatching on types 
- `sense` [`:Min`] can be `:Min` or `:Max` depending on the objective sense
- `Value` [`Vector{Float64}`] the type of a solution  

Return a [`BnBTree`](@ref) object which is the input for [`optimize!`](@ref).
"""
function initialize(;
    traverse_strategy=BestFirstSearch(),
    branch_strategy=FIRST(),
    atol=1e-6,
    rtol=1e-6,
    Node=DefaultNode,
    Value=Vector{Float64},
    Solution=DefaultSolution{Node,Value},
    root=nothing,
    sense=:Min,
    dual_gap_limit=1e-5,
    abs_gap_limit=1e-5,
)
    return BnBTree{Node,typeof(root),Value,Solution}(
        Inf,
        nothing,
        -Inf,
        Vector{Solution}(),
        PriorityQueue{Int,Tuple{Float64,Int}}(),
        Dict{Int,Node}(),
        root,
        get_branching_indices(root),
        0,
        sense,
        Options(traverse_strategy, branch_strategy, atol, rtol, dual_gap_limit, abs_gap_limit),
    )
end

"""
    get_branching_indices(root)

Return a vector of variables to branch on from the current root object.
"""
function get_branching_indices end

"""
    sort_solutions!(solutions::Vector{<:AbstractSolution})

Sort the solutions vector by objective value such that the best solution is at index 1.
"""
function sort_solutions!(solutions::Vector{<:AbstractSolution})
    return sort!(solutions; by=s -> s.objective)
end

"""
    set_node_bound!(objective_sense::Symbol, node::AbstractNode, lb, ub)

Set the bounds of the `node` object to the lower and upper bound given. 
Internally everything is stored as a minimization problem. Therefore the objective_sense `:Min`/`:Max` is needed.
"""
function set_node_bound!(objective_sense::Symbol, node::AbstractNode, lb, ub)
    if isnan(ub)
        ub = Inf
    end
    if objective_sense == :Min
        node.lb = max(lb, node.lb)
        node.ub = ub
    else
        node.lb = max(-lb, node.lb)
        node.ub = -ub
    end
end

"""
    bound!(tree::BnBTree, current_node_id::Int)

Close all nodes which have a lower bound higher or equal to the incumbent
"""
function bound!(tree::BnBTree, current_node_id::Int)
    for (_, node) in tree.nodes
        if node.id != current_node_id && node.lb >= tree.incumbent
            close_node!(tree, node)
        end
    end
end

"""
    close_node!(tree::BnBTree, node::AbstractNode)

Delete the node from the nodes dictionary and the priority queue.
"""
function close_node!(tree::BnBTree, node::AbstractNode)
    delete!(tree.nodes, node.id)
    return delete!(tree.node_queue, node.id)
end

"""
    branch!(tree, node)

Get the branching variable with [`get_branching_variable`](@ref) and then calls [`get_branching_nodes_info`](@ref) and [`add_node!`](@ref).
"""
function branch!(tree, node)
    variable_idx = get_branching_variable(tree, tree.options.branch_strategy, node)
    # no branching variable selected => return
    variable_idx == -1 && return
    nodes_info = get_branching_nodes_info(tree, node, variable_idx)
    for node_info in nodes_info
        add_node!(tree, node, node_info)
    end
end

"""
    get_branching_variable(tree::BnBTree, ::MOST_INFEASIBLE, node::AbstractNode)

Return the branching variable which is furthest away from being feasible based on [`get_distance_to_feasible`](@ref)
or `-1` if all integer constraints are respected.
"""
function get_branching_variable(tree::BnBTree, ::MOST_INFEASIBLE, node::AbstractNode)
    values = get_relaxed_values(tree, node)
    best_idx = -1
    max_distance_to_feasible = 0.0
    for i in tree.branching_indices
        value = values[i]
        if !is_approx_feasible(tree, value)
            distance_to_feasible = get_distance_to_feasible(tree, value)
            if distance_to_feasible > max_distance_to_feasible
                best_idx = i
                max_distance_to_feasible = distance_to_feasible
            end
        end
    end
    return best_idx
end

"""
    get_branching_nodes_info(tree::BnBTree, node::AbstractNode, vidx::Int)

Create the information for new branching nodes based on the variable index `vidx`.
Return a list of those information as a `NamedTuple` vector.

# Example
The following would add the necessary information about a new node and return it. The necessary information are the fields required by the [`AbstractNode`](@ref).
For this examle the required fields are the lower and upper bounds of the variables as well as the status of the node.
```julia
nodes_info = NamedTuple[]
push!(nodes_info, (
    lbs = lbs,
    ubs = ubs,
    status = MOI.OPTIMIZE_NOT_CALLED,
))
return nodes_info
```
"""
function get_branching_nodes_info end

"""
    set_root!(tree::BnBTree, node_info::NamedTuple)

Set the root node information based on the `node_info` which needs to include the same fields as the `Node` struct given 
to the [`initialize`](@ref) method. (Besides the `std` field which is set by Bonobo automatically)

# Example
If your node structure is the following:
```julia
mutable struct MIPNode <: AbstractNode
    std :: BnBNodeInfo
    lbs :: Vector{Float64}
    ubs :: Vector{Float64}
    status :: MOI.TerminationStatusCode
end
```

then you can call the function with this syntax:

```julia
set_root!(tree, (
    lbs = fill(-Inf, length(x)),
    ubs = fill(Inf, length(x)),
    status = MOI.OPTIMIZE_NOT_CALLED
))
```
"""
function set_root!(tree::BnBTree, node_info::NamedTuple)
    return add_node!(tree, nothing, node_info)
end

"""
    add_node!(tree::BnBTree{Node}, parent::Union{AbstractNode, Nothing}, node_info::NamedTuple)

Add a new node to the tree using the `node_info`. For information on that see [`set_root!`](@ref).
"""
function add_node!(
    tree::BnBTree{Node},
    parent::Union{AbstractNode,Nothing},
    node_info::NamedTuple,
) where {Node<:AbstractNode}
    node_id = tree.num_nodes + 1
    node = create_node(Node, node_id, parent, node_info)
    # only add the node if it's better than the current best solution
    if node.lb < tree.incumbent
        tree.nodes[node_id] = node
        tree.node_queue[node_id] = (node.lb, node_id)
        tree.num_nodes += 1
    end
end

"""
    create_node(Node, node_id::Int, parent::Union{AbstractNode, Nothing}, node_info::NamedTuple)

Creates a node of type `Node` with id `node_id` and the named tuple `node_info`. 
For information on that see [`set_root!`](@ref).
"""
function create_node(Node, node_id::Int, parent::Union{AbstractNode,Nothing}, node_info::NamedTuple)
    lb = -Inf
    depth = 1
    if !isnothing(parent)
        lb = parent.lb
        depth = parent.depth + 1
    end
    bnb_node = structfromnt(BnBNodeInfo, (id=node_id, lb=lb, ub=Inf, depth=depth))
    bnb_nt = (std=bnb_node,)
    node_nt = merge(bnb_nt, node_info)
    return structfromnt(Node, node_nt)
end

"""
    get_next_node(tree::BnBTree, ::BestFirstSearch)

Get the next node of the tree which shall be evaluted next by [`evaluate_node!`](@ref).
If you want to implement your own traversing strategy check out [`AbstractTraverseStrategy`](@ref).
"""
function get_next_node(tree::BnBTree, ::BestFirstSearch)
    node_id, _ = first(tree.node_queue)
    return tree.nodes[node_id]
end

function get_next_node(tree::BnBTree, ::DepthFirstSearch)
    node_id = argmax(k -> tree.nodes[k].depth, keys(tree.nodes))
    return tree.nodes[node_id]
end

"""
    evaluate_node!(tree, node)

Evaluate the current node and return the lower and upper bound of that node.
"""
function evaluate_node! end

#=
    Access standard AbstractNode internals without using .std syntax
=#
@inline function Base.getproperty(c::AbstractNode, s::Symbol)
    if s in (:id, :lb, :ub, :depth)
        Core.getproperty(Core.getproperty(c, :std), s)
    else
        getfield(c, s)
    end
end

@inline function Base.setproperty!(c::AbstractNode, s::Symbol, v)
    if s in (:id, :lb, :ub, :depth)
        Core.setproperty!(c.std, s, v)
    else
        Core.setproperty!(c, s, v)
    end
end

"""
    is_approx_feasible(tree::BnBTree, value)

Return whether a given `value` is approximately feasible based on the tolerances defined in the tree options. 
"""
function is_approx_feasible(tree::BnBTree, value::Number)
    return is_approx_feasible(value; atol=tree.options.atol, rtol=tree.options.rtol)
end

function is_approx_feasible(value::Number; atol=1e-6, rtol=1e-6)
    return isapprox(value, round(value); atol, rtol)
end

"""
    get_distance_to_feasible(tree::BnBTree, value)

Return the distance of feasibility for the given value.

- if `value::Number` this returns the distance to the nearest discrete value
"""
function get_distance_to_feasible(tree::BnBTree, value::Number)
    return abs(round(value) - value)
end

"""
    optimize!(tree::BnBTree; callback=(args...; kwargs...)->())
Optimize the problem using a branch and bound approach. 
The steps, repeated until terminated is true, are the following:
```julia
# 1. get the next open node depending on the traverse strategy
node = get_next_node(tree, tree.options.traverse_strategy)
# 2. evaluate the current node and return the lower and upper bound
# if the problem is infeasible both values should be set to NaN
lb, ub = evaluate_node!(tree, node)
# 3. update the upper and lower bound of the node struct
set_node_bound!(tree.sense, node, lb, ub)
# 4. update the best solution
updated = update_best_solution!(tree, node)
updated && bound!(tree, node.id)
# 5. remove the current node
close_node!(tree, node)
# 6. compute the node children and adds them to the tree
# internally calls get_branching_variable and branch_on_variable!
branch!(tree, node)
```
A `callback` function can be provided which will be called whenever a node is closed.
It always has the arguments `tree` and `node` and is called after the `node` is closed. 
Additionally the callback function **must** accept additional keyword arguments (`kwargs`) 
which are set in the following ways:
1. If the node is infeasible the kwarg `node_infeasible` is set to `true`.
2. If the node has a higher lower bound than the incumbent the kwarg `worse_than_incumbent` is set to `true`.
"""
function optimize!(tree::BnBTree{<:FrankWolfeNode}; callback=(args...; kwargs...) -> ())

    while !terminated(tree)
        node = get_next_node(tree, tree.options.traverse_strategy)
        lb, ub = evaluate_node!(tree, node)
        # if the problem was infeasible we simply close the node and continue
        if isnan(lb) && isnan(ub)
            close_node!(tree, node)
            callback(tree, node; node_infeasible=true)
            continue
        end

        set_node_bound!(tree.sense, node, lb, ub)

        # if the evaluated lower bound is worse than the best incumbent -> close and continue
        if !tree.root.options[:no_pruning] && node.lb >= tree.incumbent
            close_node!(tree, node)
            callback(
                tree,
                node;
                worse_than_incumbent=true,
                lb_update=isapprox(node.lb, tree.incumbent),
            )
            continue
        end

        if node.lb >= tree.incumbent
            # In pseudocost branching we need to perform the update now for nodes which will never be seen by get_branching_variable
            if isa(tree.options.branch_strategy, Boscia.Hierarchy) ||
               isa(tree.options.branch_strategy, Boscia.PseudocostBranching)
                if !isinf(node.parent_lower_bound_base)
                    idx = node.branched_on
                    update = lb - node.parent_lower_bound_base
                    update = update / node.distance_to_int
                    if isinf(update)
                        @debug "update is $(Inf)"
                    end
                    r_idx = node.branched_right ? 1 : 2
                    tree.options.branch_strategy.pseudos[idx, r_idx] = update_avg(
                        update,
                        tree.options.branch_strategy.pseudos[idx, r_idx],
                        tree.options.branch_strategy.branch_tracker[idx, r_idx],
                    )
                    tree.options.branch_strategy.branch_tracker[idx, r_idx] += 1
                end
            end
        end

        tree.node_queue[node.id] = (node.lb, node.id)
        #_ , prio = peek(tree.node_queue)
        #@assert tree.lb <= prio[1]
        #tree.lb = prio[1]
        p_lb = tree.lb
        tree.lb = minimum([prio[2][1] for prio in tree.node_queue])
        @assert p_lb <= tree.lb

        updated = update_best_solution!(tree, node)
        if updated
            bound!(tree, node.id)
            if isapprox(tree.incumbent, tree.lb; atol=tree.options.atol, rtol=tree.options.rtol)
                break
            end
        end

        close_node!(tree, node)
        branch!(tree, node)
        callback(tree, node)
    end
    # To make sure that we collect the statistics in case the time limit is reached.
    if !haskey(tree.root.result, :global_tightenings)
        y = get_solution(tree)
        vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(y)[], 1)
        dummy_node = FrankWolfeNode(
            NodeInfo(-1, Inf, Inf, 0),
            FrankWolfe.ActiveSet([(1.0, y)]),
            vertex_storage,
            IntegerBounds(),
            1e-3,
            Millisecond(0),
            0,
            0,
            0,
            0.0,
            [],
        )
        callback(tree, dummy_node, node_infeasible=true)
    end
    return sort_solutions!(tree.solutions)
end

function update_best_solution!(tree::BnBTree{<:FrankWolfeNode}, node::AbstractNode)
    isinf(node.ub) && return false

    if !tree.root.options[:add_all_solutions]
        node.ub >= tree.incumbent && return false
    end

    add_new_solution!(tree, node)
    return true
end

function add_new_solution!(
    tree::BnBTree{N,R,V,S},
    node::AbstractNode,
) where {N,R,V,S<:FrankWolfeSolution{N,V}}
    return add_new_solution!(tree, node, node.ub, get_relaxed_values(tree, node), :iterate)
end

function add_new_solution!(
    tree::BnBTree{N,R,V,S},
    node::AbstractNode,
    objective::T,
    solution::V,
    origin::Symbol,
) where {N,R,V,S<:FrankWolfeSolution{N,V},T<:Real}
    time = Inf
    if tree.root.options[:post_heuristics_callback] !== nothing
        add_solution, time, objective, solution =
            tree.root.options[:post_heuristics_callback](tree, node, solution)
    end

    sol = FrankWolfeSolution(objective, solution, node, origin, time)
    sol.solution = solution
    sol.objective = objective

    push!(tree.solutions, sol)
    if tree.incumbent_solution === nothing || sol.objective < tree.incumbent_solution.objective
        tree.root.updated_incumbent[] = true
        tree.incumbent_solution = sol
        tree.incumbent = sol.objective
    end
end

function get_solution(tree::BnBTree{N,R,V,S}; result=1) where {N,R,V,S<:FrankWolfeSolution{N,V}}
    if isempty(tree.solutions)
        @warn "There is no solution in the tree. This behaviour can happen if you have supplied 
        \na custom domain oracle. In that case, try to increase the time or node limit. If you have not specified a 
        \ndomain oracle, please report!"
        @assert tree.root.problem.solving_stage in (TIME_LIMIT_REACHED, NODE_LIMIT_REACHED)
        return nothing
    end
    return tree.solutions[result].solution
end

struct BiasedDepthFirstSearch <: AbstractTraverseStrategy
    favor_right::Bool
end

BiasedDepthFirstSearch() = BiasedDepthFirstSearch(true)

function get_next_node(tree::BnBTree, strategy::BiasedDepthFirstSearch)
    node_queue = tree.node_queue
    nodes = tree.nodes

    # For favored branch side (e.g. right if strategy.favor_right == true)
    favored_id = -1
    favored_depth = -1
    favored_lb = Inf  # we maximize depth, then minimize lb

    # For unfavored side
    unfavored_id = -1
    unfavored_lb = Inf          # we minimize lb

    for id in keys(node_queue)

        node = nodes[id]

        if node.branched_right == strategy.favor_right
            # Favored: maximize depth, tie-break by smaller lb
            if node.depth > favored_depth || (node.depth == favored_depth && node.lb < favored_lb)
                favored_depth = node.depth
                favored_lb = node.lb
                favored_id = id
            end
        else
            # Unfavored: choose smallest lb
            if node.lb < unfavored_lb
                unfavored_lb = node.lb
                unfavored_id = id
            end
        end
    end

    if favored_id !== -1
        return nodes[favored_id]
    end

    return nodes[unfavored_id]
end

export BnBTree, BnBNodeInfo, AbstractNode, AbstractSolution
export AbstractTraverseStrategy, AbstractBranchStrategy
export BestFirstSearch, DepthFirstSearch
export FIRST, MOST_INFEASIBLE
