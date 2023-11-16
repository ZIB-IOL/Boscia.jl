## TO DO 

## Have a general interface for heuristics 
## so that the user can link in custom heuristics.
"""
    Boscia Heuristic

Interface for heuristics in Boscia.    
`h` is the heuristic function receiving as input the tree, the bounded LMO and a point x (the current node solution).
It returns the heuristic solution (can be nothing, we check for that) and whether feasibility still has to be check.
`prob` is the probability with which it will be called.        
"""
# Would 'Heuristic' also suffice? Or might we run into Identifer conflicts with other packages?
struct Heuristic{F<:Function}
    run_heuristic::F
    prob::Float64
    identifer::Symbol
end

# Create default heuristic. Doesn't do anything and should be called.
Heuristic() = Heuristic(x -> nothing, -0.1, :default)

"""
Flip coin.
"""
function flip_coin(w=0.5)
    return rand() ≤ w
end

"""
Add a new solution found from the heuristic to the tree.
"""
function add_heuristic_solution(tree, x, val, heuristic_name::Symbol)
    tree.root.updated_incumbent[] = true
    node = tree.nodes[tree.root.current_node_id[]]
    sol = FrankWolfeSolution(val, x, node, heuristic_name)
    push!(tree.solutions, sol)
    if tree.incumbent_solution === nothing ||
        sol.objective < tree.incumbent_solution.objective
        tree.incumbent_solution = sol
    end
    tree.incumbent = val
    Bonobo.bound!(tree, node.id)
end

# TO DO: We might want to change the probability depending on the depth of the tree
# or have other additional criteria on whether to run a heuristic
"""
Choose which heuristics to run by rolling a dice.
"""
function run_heuristics(tree, x, heuristic_list)
    for heuristic in heuristic_list
        if flip_coin(heuristic.prob)
            list_x_heu, check_feasibility = heuristic.run_heuristic(tree, tree.root.problem.tlmo.blmo, x)

            # check feasibility
            if !isempty(list_x_heu)
                min_val = Inf
                min_idx = -1
                for (i, x_heu) in enumerate(list_x_heu)
                    feasible = check_feasibility ? is_linear_feasible(tree.root.problem.tlmo, x_heu) && is_integer_feasible(tree, x_heu) : true
                    if feasible
                        val = tree.root.problem.f(x_heu)
                        if val < min_val
                            min_val = val
                            min_idx = i 
                        end
                    end
                end

                if min_val < tree.incumbent # Inf < Inf = false
                    add_heuristic_solution(tree, list_x_heu[min_idx],min_val, heuristic.identifer)
                end
            end
        end
    end
end

"""
Simple rounding heuristic.
"""
function rounding_heuristic(tree::Bonobo.BnBTree, blmo::BoundedLinearMinimizationOracle, x)
    x_rounded = copy(x)
    for idx in tree.branching_indices
        x_rounded[idx] = round(x[idx])
    end
    return [x_rounded], true
end
