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
function flip_coin(w=0.5, rng=Random.GLOBAL_RNG)
    return rand(rng) ≤ w
end

"""
Add a new solution found from the heuristic to the tree.
"""
function add_heuristic_solution(tree, x, val, heuristic_name::Symbol)
    node = tree.nodes[tree.root.current_node_id[]]
    add_new_solution!(tree, node, val, x, heuristic_name)
    Bonobo.bound!(tree, node.id)
end

# TO DO: We might want to change the probability depending on the depth of the tree
# or have other additional criteria on whether to run a heuristic
"""
Choose which heuristics to run by rolling a dice.
"""
function run_heuristics(tree, x, heuristic_list; rng=Random.GLOBAL_RNG)
    inner_lmo = tree.root.problem.tlmo.blmo
    heuristic_lmo = TimeTrackingLMO(inner_lmo, tree.root.problem.integer_variables)

    for heuristic in heuristic_list
        if flip_coin(heuristic.prob, rng)
            list_x_heu, check_feasibility = heuristic.run_heuristic(tree, heuristic_lmo, x)

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

    # collect statistics from heuristic lmo
    tree.root.options[:heu_ncalls] += heuristic_lmo.ncalls
    return true
end

"""
Simple rounding heuristic.
"""
function rounding_heuristic(tree::Bonobo.BnBTree, blmo::Boscia.TimeTrackingLMO, x)
    x_rounded = copy(x)
    for idx in tree.branching_indices
        x_rounded[idx] = round(x[idx])
    end
    return [x_rounded], true
end


"""
    follow-the-gradient
Follow the gradient for a fixed number of steps and collect solutions on the way
"""
function follow_gradient_heuristic(tree::Bonobo.BnBTree, tlmo::Boscia.TimeTrackingLMO, x, k)
    nabla = similar(x)
    x_new = copy(x)
    sols = []
    for i in 1:k
        tree.root.problem.g(nabla,x_new)
        x_new = Boscia.compute_extreme_point(tlmo, nabla)
        push!(sols, x_new)
    end
    return sols, false
end


"""
Advanced lmo-aware rounding for binary vars. Rounding respecting the hidden feasible region structure.
"""
function rounding_lmo_01_heuristic(tree::Bonobo.BnBTree, tlmo::Boscia.TimeTrackingLMO, x)
    nabla = zeros(length(x))
    for idx in tree.branching_indices
        nabla[idx] = 1 - 2*round(x[idx]) # (0.7, 0.3) -> (1, 0) -> (-1, 1) -> min -> (1,0)
    end
    x_rounded = Boscia.compute_extreme_point(tlmo, nabla)
    return [x_rounded], false
end

"""
Probability rounding for 0/1 problems.
It decides based on the fractional value whether to ceil or floor the variable value. 
Afterward, one call to Frank-Wolfe is performed to optimize the continuous variables.    
"""
function probability_rounding(tree::Bonobo.BnBTree, tlmo::Boscia.TimeTrackingLMO, x; rng=Random.GLOBAL_RNG)
    # save original bounds
    node = tree.nodes[tree.root.current_node_id[]]
    original_bounds = copy(node.local_bounds)

    bounds = IntegerBounds()
    for (i,x_i) in zip(tlmo.blmo.int_vars, x[tlmo.blmo.int_vars])
        x_rounded = flip_coin(x_i, rng) ? ceil(x_i) : floor(x_i)
        push!(bounds, (i, x_rounded), :lessthan)
        push!(bounds, (i, x_rounded), :greaterthan)
    end

    build_LMO(tlmo, tree.root.problem.integer_variable_bounds, bounds, tlmo.blmo.int_vars)

    # check for feasibility and boundedness
    status = check_feasibility(tlmo)
    if status == INFEASIBLE || status == UNBOUNDED
        @debug "LMO state in the probability rounding heuristic: $(status)"
        # reset LMO to node state
        build_LMO(tlmo, tree.root.problem.integer_variable_bounds, original_bounds, tlmo.blmo.int_vars)
        # just return the point
        return [x], false
    end

    v = compute_extreme_point(tlmo, rand(length(x)))
    active_set = FrankWolfe.ActiveSet([(1.0, v)])

    x_rounded, _, _, _ = solve_frank_wolfe(
        tree.root.options[:variant],
        tree.root.problem.f,
        tree.root.problem.g,
        tree.root.problem.tlmo,
        active_set;
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

    # reset LMO to node state
    build_LMO(tlmo, tree.root.problem.integer_variable_bounds, original_bounds, tlmo.blmo.int_vars)
    
    return [x_rounded], true
end
