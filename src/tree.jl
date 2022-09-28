
"""
Branch-and-Bound data structure.
"""
mutable struct BranchingTree{T<:Real,BT<:Real}
    node_counter::Int
    nodes::Vector{Node{T}}
    lowerbound::BT
    upperbound::BT #current best objective value
end

function run(tree::BranchingTree)
    while !termination(tree)
        node = next_node(tree)
        update_best_bound!(tree)

        bound!(node)
        heuristics!(node)
        processed!(tree, node)
        branch!(tree, node)
    end
end

function update_best_bound!(tree::BranchingTree)
    return tree.lowerbound = minimum(node.local_lowerbound for node in tree.nodes)
end

function process_node(tree::BranchingTree, node::Node, problem::SimpleOptimizationProblem)
    # TODO: call LP solver to detect infeasibility first
    # if infeasible, prune

    # TODO call warm-started FW from the node bound
    val, xsol = nothing, nothing
    node.solution_status = MOI.OPTIMAL
    node.local_lowerbound = val
    node.active_set = xsol
    is_integer = true
    for var in problem.binary_variables
        if xsol[var] ≉ 0 && xsol[var] ≉ 1
            is_integer = false
        end
    end
    if is_integer
        # update best incumbent
        if tree.upperbound > val
            tree.upperbound = val
        end
    end
end
