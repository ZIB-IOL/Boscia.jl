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
struct Heuristic
    h::Function
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
function add_heuristic_solution(tree, x, val, heu::Symbol)
    tree.root.updated_incumbent[] = true
    node = tree.nodes[tree.root.current_node_id[]]
    sol = FrankWolfeSolution(val, x, node, heu)
    push!(tree.solutions, sol)
    if tree.incumbent_solution === nothing ||
        sol.objective < tree.incumbent_solution.objective
        tree.incumbent_solution = sol
    end
    tree.incumbent = val
    Bonobo.bound!(tree, node.id)
end

"""
Choose which heuristics to run by rolling a dice.
"""
# TO DO: We might want to change the probability depending on the depth of the tree
# or have other additional criteria on whether to run a heuristic
function run_heuristics(tree, x, heuristic_list)
    for heuristic in heuristic_list
        if flip_coin(heuristic.prob)
            x_heu = heuristic.h(tree, x)

            # check feasibility
            if x_heu !== nothing && is_linear_feasible(tree.root.problem.tlmo, x_heu) &&
                is_integer_feasible(tree, x_heu)
                val = tree.root.problem.f(x_heu)
                if val < tree.incumbent
                    add_heuristic_solution(tree, x_heu, val, heuristic.identifer)
                end
            end
        end
    end
end


"""
Simple rounding heuristic.
"""
function rounding_heuristics(tree::Bonobo.BnBTree, x)
    x_rounded = copy(x)
    for idx in tree.branching_indices
        x_rounded[idx] = round(x[idx])
    end
    return x_rounded
end

