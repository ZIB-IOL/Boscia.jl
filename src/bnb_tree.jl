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