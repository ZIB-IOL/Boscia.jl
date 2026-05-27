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
    sort_solutions!(solutions::Vector{<:AbstractSolution})

Sort the solutions vector by objective value such that the best solution is at index 1.
"""
function sort_solutions!(solutions::Vector{<:AbstractSolution})
    return sort!(solutions; by=s -> s.objective)
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