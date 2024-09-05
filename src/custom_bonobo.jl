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
function Bonobo.optimize!(
    tree::Bonobo.BnBTree{<:FrankWolfeNode};
    callback=(args...; kwargs...) -> (),
)
    while !Bonobo.terminated(tree)
        node = Bonobo.get_next_node(tree, tree.options.traverse_strategy)
        lb, ub = Bonobo.evaluate_node!(tree, node)
        # if the problem was infeasible we simply close the node and continue
        if isnan(lb) && isnan(ub)
            Bonobo.close_node!(tree, node)
            callback(tree, node; node_infeasible=true)
            continue
        end

        Bonobo.set_node_bound!(tree.sense, node, lb, ub)

        # if the evaluated lower bound is worse than the best incumbent -> close and continue
        if node.lb >= tree.incumbent
            Bonobo.close_node!(tree, node)
            callback(
                tree,
                node;
                worse_than_incumbent=true,
                lb_update=isapprox(node.lb, tree.incumbent),
            )
            continue
        end

        tree.node_queue[node.id] = (node.lb, node.id)
        #_ , prio = peek(tree.node_queue)
        #@assert tree.lb <= prio[1]
        #tree.lb = prio[1]
        p_lb = tree.lb
        tree.lb = minimum([prio[2][1] for prio in tree.node_queue])
        @assert p_lb <= tree.lb

        updated = Bonobo.update_best_solution!(tree, node)
        if updated
            Bonobo.bound!(tree, node.id)
            if isapprox(tree.incumbent, tree.lb; atol=tree.options.atol, rtol=tree.options.rtol)
                break
            end
        end

        Bonobo.close_node!(tree, node)
        Bonobo.branch!(tree, node)
        callback(tree, node)
    end
    return Bonobo.sort_solutions!(tree.solutions, tree.sense)
end

function Bonobo.update_best_solution!(
    tree::Bonobo.BnBTree{<:FrankWolfeNode},
    node::Bonobo.AbstractNode,
)
    isinf(node.ub) && return false
    node.ub >= tree.incumbent && return false

    Bonobo.add_new_solution!(tree, node)
    return true
end

function Bonobo.add_new_solution!(
    tree::Bonobo.BnBTree{N,R,V,S},
    node::Bonobo.AbstractNode,
) where {N,R,V,S<:FrankWolfeSolution{N,V}}
    add_new_solution!(tree, node, node.ub, Bonobo.get_relaxed_values(tree, node), :iterate)
end

function add_new_solution!(
    tree::Bonobo.BnBTree{N,R,V,S},
    node::Bonobo.AbstractNode,
    objective::T,
    solution::V,
    origin::Symbol,
) where {N,R,V,S<:FrankWolfeSolution{N,V},T<:Real}
    sol = FrankWolfeSolution(objective, solution, node, origin)
    sol.solution = solution
    sol.objective = objective

    push!(tree.solutions, sol)
    if tree.incumbent_solution === nothing || sol.objective < tree.incumbent_solution.objective
        tree.root.updated_incumbent[] = true
        tree.incumbent_solution = sol
        tree.incumbent = sol.objective
    end
end

function Bonobo.get_solution(
    tree::Bonobo.BnBTree{N,R,V,S};
    result=1,
) where {N,R,V,S<:FrankWolfeSolution{N,V}}
    if isempty(tree.solutions)
        @warn "There is no solution in the tree. This behaviour can happen if you have supplied 
        \na custom domain oracle. In that case, try to increase the time limit. If you have not specified a 
        \ndomain oracle, please report!"
        @assert tree.root.problem.solving_stage == TIME_LIMIT_REACHED
        return nothing
    end
    return tree.solutions[result].solution
end

