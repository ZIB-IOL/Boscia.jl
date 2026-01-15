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
    if !tree.root.options[:no_pruning]
        Bonobo.bound!(tree, node.id)
    end
end

# TO DO: We might want to change the probability depending on the depth of the tree
# or have other additional criteria on whether to run a heuristic
"""
Choose which heuristics to run by rolling a dice.
"""
function run_heuristics(tree, x, heuristic_list; rng=Random.GLOBAL_RNG)
    inner_lmo = tree.root.problem.tlmo.lmo
    heuristic_lmo = TimeTrackingLMO(
        inner_lmo,
        tree.root.problem.integer_variables,
        tree.root.problem.tlmo.time_ref,
        tree.root.problem.tlmo.time_limit,
    )

    for heuristic in heuristic_list
        if flip_coin(heuristic.prob, rng)
            list_x_heu, check_feasibility = heuristic.run_heuristic(tree, heuristic_lmo, x)

            # check feasibility
            if !isempty(list_x_heu)
                min_val = Inf
                min_idx = -1
                for (i, x_heu) in enumerate(list_x_heu)
                    feasible =
                        check_feasibility ?
                        is_linear_feasible(tree.root.problem.tlmo, x_heu) &&
                        is_integer_feasible(tree, x_heu) &&
                        tree.root.options[:domain_oracle](x_heu) :
                        tree.root.options[:domain_oracle](x_heu)
                    if feasible
                        val = tree.root.problem.f(x_heu)
                        if tree.root.options[:add_all_solutions]
                            add_heuristic_solution(tree, list_x_heu[i], val, heuristic.identifer)
                            continue
                        end
                        if val < min_val
                            min_val = val
                            min_idx = i
                        end
                    end
                end

                if min_val < tree.incumbent && !tree.root.options[:add_all_solutions] # Inf < Inf = false
                    add_heuristic_solution(tree, list_x_heu[min_idx], min_val, heuristic.identifer)
                end
            end
        end
        time = float(Dates.value(Dates.now() - tree.root.problem.tlmo.time_ref))

        if tree.root.options[:time_limit] < Inf &&
           time / 1000.0 ≥ tree.root.options[:time_limit] - 10
            break
        end
    end

    # collect statistics from heuristic lmo
    tree.root.options[:heu_ncalls] += heuristic_lmo.ncalls
    return true
end

"""
Simple rounding heuristic.
"""
function rounding_heuristic(tree::Bonobo.BnBTree, lmo::Boscia.TimeTrackingLMO, x)
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
    sol_hashes = Set{UInt}()
    for i in 1:k
        time = float(Dates.value(Dates.now() - tree.root.problem.tlmo.time_ref))
        if tree.root.options[:time_limit] < Inf &&
           time / 1000.0 ≥ tree.root.options[:time_limit] - 10
            break
        end

        tree.root.problem.g(nabla, x_new)
        x_new = Boscia.compute_extreme_point(tlmo, nabla)
        sol_hash = hash(x_new)
        if in(sol_hash, sol_hashes)
            break
        end
        push!(sols, x_new)
        push!(sol_hashes, sol_hash)
    end
    return sols, false
end


"""
Advanced lmo-aware rounding for binary vars. Rounding respecting the hidden feasible region structure.
"""
function rounding_lmo_01_heuristic(tree::Bonobo.BnBTree, tlmo::Boscia.TimeTrackingLMO, x)
    nabla = zeros(length(x))
    for idx in tree.branching_indices
        nabla[idx] = 1 - 2 * round(x[idx]) # (0.7, 0.3) -> (1, 0) -> (-1, 1) -> min -> (1,0)
    end
    x_rounded = Boscia.compute_extreme_point(tlmo, nabla)
    return [x_rounded], false
end

"""
Probability rounding for 0/1 problems.
It decides based on the fractional value whether to ceil or floor the variable value. 
Afterward, one call to Frank-Wolfe is performed to optimize the continuous variables.    
"""
function probability_rounding(
    tree::Bonobo.BnBTree,
    tlmo::Boscia.TimeTrackingLMO,
    x;
    rng=Random.GLOBAL_RNG,
)
    # save original bounds
    node = tree.nodes[tree.root.current_node_id[]]
    original_bounds = copy(node.local_bounds)

    bounds = IntegerBounds()
    for (i, x_i) in zip(tree.root.problem.integer_variables, x[tree.root.problem.integer_variables])
        x_rounded = flip_coin(x_i, rng) ? min(1.0, ceil(x_i)) : max(0.0, floor(x_i))
        push!(bounds, (i, x_rounded), :lessthan)
        push!(bounds, (i, x_rounded), :greaterthan)
    end

    build_LMO(
        tlmo,
        tree.root.problem.integer_variable_bounds,
        bounds,
        tree.root.problem.integer_variables,
    )

    # check for feasibility and boundedness
    status = check_feasibility(tlmo)
    if status == INFEASIBLE || status == UNBOUNDED
        @debug "LMO state in the probability rounding heuristic: $(status)"
        # reset LMO to node state
        build_LMO(
            tlmo,
            tree.root.problem.integer_variable_bounds,
            original_bounds,
            tree.root.problem.integer_variables,
        )
        # just return the point
        return [x], true
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
        line_search=tree.root.options[:line_search],
        lazy=tree.root.options[:lazy],
        lazy_tolerance=tree.root.options[:lazy_tolerance],
        callback=tree.root.options[:callback],
        verbose=tree.root.options[:fw_verbose],
    )

    @assert sum(
        isapprox.(
            x_rounded[tree.root.problem.integer_variables],
            round.(x_rounded[tree.root.problem.integer_variables]),
        ),
    ) == length(tree.root.problem.integer_variables) "$(sum(isapprox.(x_rounded[tree.root.problem.integer_variables], round.(x_rounded[tree.root.problem.integer_variables])))) == $(length(tree.root.problem.integer_variables)) $(x_rounded[tree.root.problem.integer_variables])"

    # reset LMO to node state
    build_LMO(
        tlmo,
        tree.root.problem.integer_variable_bounds,
        original_bounds,
        tree.root.problem.integer_variables,
    )

    return [x_rounded], false
end
