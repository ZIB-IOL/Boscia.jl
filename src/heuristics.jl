## TO DO 

## Have a general interface for heuristics 
## so that the user can link in custom heuristics.
"""
    Boscia Heuristic

Interface for heuristics in Boscia.    
`h` is the heuristic function receiving as input ..
`prob` is the probability with which it will be called.        
"""
# Would 'Heuristic' also suffice? Or might we run into Identifer conflicts with other packages?
struct BosciaHeuristic
    h::Function
    prob::Float64
end

BosciaHeuristic() = BosciaHeuristic(x -> nothing, 0.0)

const MAX_PROB = 0.3

"""
Chooses heuristic by rolling a dice.
"""
function pick_heuristic(heuristic_list)
    return heuristic_list[1]
end

"""
Simple rounding heuristic.
"""
function rounding_heuristics(tree::Bonobo.BnBTree, x)
    x_rounded = copy(x)
    for idx in tree.branching_indices
        x_rounded[idx] = round(x[idx])
    end

    # check linear feasibility
    if is_linear_feasible(tree.root.problem.tlmo, x_rounded) &&
        is_integer_feasible(tree, x_rounded)
        # evaluate f(rounded)
        val = tree.root.problem.f(x_rounded)
        if val < tree.incumbent
            tree.root.updated_incumbent[] = true
            node = tree.nodes[tree.root.current_node_id[]]
            sol = FrankWolfeSolution(val, x_rounded, node, :rounded)
            push!(tree.solutions, sol)
            if tree.incumbent_solution === nothing ||
                sol.objective < tree.incumbent_solution.objective
                tree.incumbent_solution = sol
            end
            tree.incumbent = val
            Bonobo.bound!(tree, node.id)
        end
    end
end

