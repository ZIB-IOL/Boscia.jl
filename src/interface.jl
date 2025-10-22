# Interface.jl

"""
    solve(f, g, lmo::LinearMinimizationOracle; ...)

Requires

- `f` oracle of the objective function.
- `g` oracle of the gradient of the objective
- `lmo` encodes the feasible region and can handle additional bound constraints. This can either be a MIP solver instance (e.g., SCIP) or be a custom type (see `polytope_blmos.jl`). Has to be of type `FrankWolfe.LinearMinimizationOracle` (see `blmo_interface.jl`).

Returns

- `x` the best solution found.
- `tlmo` the LMO wrapped in a TimeTrackingLMO instance.
- `result` a dictionary containg the statistics like number of nodes, total solving etc. It also contains information for plotting progress plots like the lower and upper bound progress.

Optional settings

- `settings_bnb` dictionary of settings for the branch-and-bound algorithm. Created via `settings_bnb()`.
- `settings_frank_wolfe` dictionary of settings for the Frank-Wolfe algorithm. Created via `settings_frank_wolfe()`.
- `settings_tolerances` dictionary of settings for the tolerances. Created via `settings_tolerances()`.
- `settings_postprocessing` dictionary of settings for the postprocessing. Created via `settings_postprocessing()`.
- `settings_heuristic` dictionary of settings for the heuristics. Created via `settings_heuristic()`.
- `settings_tightening` dictionary of settings for the tightening. Created via `settings_tightening()`.
- `settings_domain` dictionary of settings for the domain. Created via `settings_domain()`.
"""
function solve(
    f,
    grad!,
    lmo::FrankWolfe.LinearMinimizationOracle;
    settings=create_default_settings(),
    kwargs...,
)
    # For as long as MathOptBLMO is not yet deleted.
    if lmo isa MathOptBLMO
        println("Convert MathOptBLMO to MathOptLMO")
        lmo = convert(MathOptLMO, lmo)
    end

    build_heuristics(settings.heuristic)
    options = merge(
        settings.mode,
        settings.branch_and_bound,
        settings.frank_wolfe,
        settings.tolerances,
        settings.postprocessing,
        settings.heuristic,
        settings.tightening,
        settings.domain,
    )
    merge!(options, Dict(:heu_ncalls => 0))
    if typeof(options[:variant]) == DecompositionInvariantConditionalGradient
        if !is_decomposition_invariant_oracle(lmo)
            error("DICG within Boscia is not implemented for $(typeof(lmo)).")
        end
    end
    if options[:verbose]
        println("\nBoscia Algorithm.\n")
        println("Parameter settings.")
        println("\t Tree traversal strategy: ", _value_to_print(options[:traverse_strategy]))
        println("\t Branching strategy: ", _value_to_print(options[:branching_strategy]))
        isa(options[:branching_strategy], Boscia.Hierarchy) && println(
            "\t Order of criteria in Hierarchy Branching: ",
            [stage.name for stage in options[:branching_strategy].stages],
        )
        println("\t FrankWolfe variant: $(options[:variant])")
        println("\t Line Search Method: $(options[:line_search])")
        println("\t Lazification: $(options[:lazy])")
        options[:lazy] ? println("\t Lazification Tolerance: $(options[:lazy_tolerance])") : nothing
        @printf("\t Absolute dual gap tolerance: %e\n", options[:dual_gap])
        @printf("\t Relative dual gap tolerance: %e\n", options[:rel_dual_gap])
        @printf("\t Frank-Wolfe subproblem tolerance: %e\n", options[:fw_epsilon])
        @printf("\t Frank-Wolfe dual gap decay factor: %e\n", options[:dual_gap_decay_factor])
        println("\t Additional kwargs: ", join(keys(kwargs), ","))
    end

    n, _ = get_list_of_variables(lmo)

    integer_variables = Vector{Int}()
    num_int = 0
    num_bin = 0
    for c_idx in get_integer_variables(lmo)
        push!(integer_variables, c_idx)
        num_int += 1
    end

    if num_int == 0
        @warn("No integer variables detected! Please use an MIP solver!")
    end

    if options[:verbose]
        println("\t Total number of variables: ", n)
        println("\t Number of integer variables: $(num_int)\n")
    end

    global_bounds = build_global_bounds(lmo, integer_variables)

    if typeof(options[:domain_oracle]) != typeof(_trivial_domain) &&
       typeof(options[:find_domain_point]) == typeof(_trivial_domain_point)
        @warn "For a non trivial domain oracle, please provide the DOMAIN POINT function. Otherwise, Boscia might not converge."
    end

    time_ref = Dates.now()
    time_lmo = TimeTrackingLMO(lmo, integer_variables, time_ref, Float64(options[:time_limit]))

    v = []
    if options[:active_set] === nothing
        direction = collect(1.0:n)
        v = compute_extreme_point(lmo, direction)
        v[integer_variables] = round.(v[integer_variables])
        @assert isfinite(f(v))
        options[:active_set] = FrankWolfe.ActiveSet([(1.0, v)])
        vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)
    else
        @assert FrankWolfe.active_set_validate(options[:active_set])
        for a in options[:active_set].atoms
            @assert is_linear_feasible(lmo, a)
        end
        x = FrankWolfe.compute_active_set_iterate!(options[:active_set])
        v = x
        @assert isfinite(f(x))
    end
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    pre_computed_set =
        if typeof(options[:variant]) == DecompositionInvariantConditionalGradient &&
           options[:variant].use_DICG_warm_start
            [v]
        else
            nothing
        end

    m = SimpleOptimizationProblem(f, grad!, n, integer_variables, time_lmo, global_bounds)
    nodeEx = FrankWolfeNode(
        NodeInfo(1, f(v), f(v)),
        options[:active_set],
        vertex_storage,
        IntegerBounds(),
        1,
        1e-3,
        Millisecond(0),
        0,
        0,
        0,
        0.0,
        [v],
    )

    # If we cannot trust the lower bound, we also shouldn't do any tighening.
    if options[:ignore_lower_bound]
        options[:dual_tightening] = false
        options[:global_dual_tightening] = false
    end

    Node = typeof(nodeEx)
    Value = typeof(options[:active_set].atoms[1])
    tree = Bonobo.initialize(;
        traverse_strategy=options[:traverse_strategy],
        Node=Node,
        Value=Value,
        Solution=FrankWolfeSolution{Node,Value},
        root=(
            problem=m,
            current_node_id=Ref{Int}(0),
            updated_incumbent=Ref{Bool}(false),
            global_tightening_rhs=Ref(-Inf),
            global_tightening_root_info=(
                lower_bounds=Dict{Int,Tuple{Float64,Float64}}(),
                upper_bounds=Dict{Int,Tuple{Float64,Float64}}(),
            ),
            global_tightenings=IntegerBounds(),
            options=options,
            result=Dict{Symbol,Any}(),
        ),
        branch_strategy=options[:branching_strategy],
        dual_gap_limit=options[:rel_dual_gap],
        abs_gap_limit=options[:dual_gap],
    )
    Bonobo.set_root!(
        tree,
        (
            active_set=options[:active_set],
            discarded_vertices=vertex_storage,
            local_bounds=IntegerBounds(),
            level=1,
            fw_dual_gap_limit=options[:fw_epsilon],
            fw_time=Millisecond(0),
            global_tightenings=0,
            local_tightenings=0,
            local_potential_tightenings=0,
            dual_gap=(-Inf),
            pre_computed_set=pre_computed_set,
            parent_lower_bound_base=Inf,
            branched_on=-1,
            branched_right=false,
            distance_to_int=0.0,
        ),
    )

    if options[:start_solution] !== nothing
        if size(options[:start_solution]) != size(v)
            error(
                "size of starting solution differs from vertices: $(size(options[:start_solution])), $(size(v))",
            )
        end
        # Sanity check that the provided solution is in fact feasible.
        @assert is_linear_feasible(lmo, options[:start_solution]) &&
                is_integer_feasible(tree, options[:start_solution])
        node = tree.nodes[1]
        add_new_solution!(tree, node, f(options[:start_solution]), options[:start_solution], :start)
    end

    # build callbacks
    list_ub_cb = Float64[]
    list_lb_cb = Float64[]
    list_time_cb = Float64[]
    list_num_nodes_cb = Int[]
    list_lmo_calls_cb = Int[]
    list_active_set_size_cb = Int[]
    list_discarded_set_size_cb = Int[]
    fw_iterations = Int[]
    node_level = Int[]
    lmo_calls_per_layer = Vector{Vector{Int}}()
    active_set_size_per_layer = Vector{Vector{Int}}()
    discarded_set_size_per_layer = Vector{Vector{Int}}()
    global_tightenings = Int[]
    local_tightenings = Int[]
    local_potential_tightenings = Int[]

    bnb_callback = build_bnb_callback(
        tree,
        time_ref,
        list_lb_cb,
        list_ub_cb,
        list_time_cb,
        list_num_nodes_cb,
        list_lmo_calls_cb,
        options[:verbose],
        fw_iterations,
        list_active_set_size_cb,
        list_discarded_set_size_cb,
        tree.root.result,
        lmo_calls_per_layer,
        active_set_size_per_layer,
        discarded_set_size_per_layer,
        node_level,
        options[:bnb_callback],
        global_tightenings,
        local_tightenings,
        local_potential_tightenings,
        num_bin,
        num_int,
    )

    fw_callback = build_FW_callback(
        tree,
        options[:min_number_lower],
        true,
        fw_iterations,
        options[:min_fw_iterations],
        time_ref,
        options[:time_limit],
        use_DICG=typeof(options[:variant]) == DecompositionInvariantConditionalGradient,
    )

    tree.root.options[:callback] = fw_callback
    tree.root.current_node_id[] = Bonobo.get_next_node(tree, tree.options.traverse_strategy).id

    Bonobo.optimize!(tree; callback=bnb_callback)

    x = postsolve(tree, tree.root.result, time_ref, options[:verbose], options[:max_iteration_post])

    # Check solution and polish
    x_polished = x
    if x !== nothing
        if !is_linear_feasible(tree.root.problem.tlmo, x)
            error("Reported solution not linear feasbile!")
        end
        if !is_integer_feasible(tree.root.problem.integer_variables, x, atol=1e-16, rtol=1e-16) &&
           x !== nothing
            @info "Polish solution"
            for i in tree.root.problem.integer_variables
                x_polished[i] = round(x_polished[i])
            end
            if !is_linear_feasible(tree.root.problem.tlmo, x_polished)
                @warn "Polished solution not linear feasible"
            else
                x = x_polished
            end
        end
    end
    if options[:verbose]
        println() # cleaner output
    end

    return x, tree.root.problem.tlmo, tree.root.result
end

"""
    postsolve(tree, result, time_ref, verbose, max_iteration_post)

Runs the post solve to optimize for the continuous variables if present.
Is called if `use_post_solve` is enabled in the `solve` function.
Prints solution statistics if verbose is set to `true`.        
"""
function postsolve(tree, result, time_ref, verbose, max_iteration_post)
    x = Bonobo.get_solution(tree)
    primal = x !== nothing ? tree.incumbent_solution.objective : Inf

    status_string = "FIX ME" # should report "feasible", "optimal", "infeasible", "gap tolerance met"
    if isempty(tree.nodes)
        status_string = "Optimal (tree empty)"
        tree.root.problem.solving_stage = OPT_TREE_EMPTY
    elseif tree.root.problem.solving_stage == TIME_LIMIT_REACHED
        status_string = "Time limit reached"
    elseif tree.root.problem.solving_stage == NODE_LIMIT_REACHED
        status_string = "Node limit reached"
    elseif tree.root.problem.solving_stage == USER_STOP
        status_string = "User defined stop"
    else
        status_string = "Optimal (tolerance reached)"
        tree.root.problem.solving_stage = OPT_GAP_REACHED
    end

    only_integer_vars = tree.root.problem.nvars == length(tree.root.problem.integer_variables)
    if tree.root.options[:use_postsolve] && !only_integer_vars
        # Build solution lmo
        fix_bounds = IntegerBounds()
        for i in tree.root.problem.integer_variables
            push!(fix_bounds, (i => round(x[i])), :lessthan)
            push!(fix_bounds, (i => round(x[i])), :greaterthan)
        end

        free_model(tree.root.problem.tlmo.lmo)
        build_LMO(
            tree.root.problem.tlmo,
            tree.root.problem.integer_variable_bounds,
            fix_bounds,
            tree.root.problem.integer_variables,
        )
        # previous solution rounded to account for 0.99999.. or 1.00000000002 types of values
        prev_x_rounded = copy(x)
        prev_x_rounded[tree.root.problem.integer_variables] .=
            round.(prev_x_rounded[tree.root.problem.integer_variables])
        prev_x_rounded =
            is_linear_feasible(tree.root.problem.tlmo, prev_x_rounded) ? prev_x_rounded : x

        # Postprocessing
        direction = ones(length(x))
        v = compute_extreme_point(tree.root.problem.tlmo, direction)
        active_set = FrankWolfe.ActiveSet([(1.0, v)])
        verbose && println("Postprocessing")
        x, _, primal, dual_gap, _, _ = FrankWolfe.blended_pairwise_conditional_gradient(
            tree.root.problem.f,
            tree.root.problem.g,
            tree.root.problem.tlmo,
            active_set,
            line_search=tree.root.options[:line_search],
            lazy=true,
            verbose=verbose,
            max_iteration=max_iteration_post,
        )

        # update tree
        if primal < tree.incumbent
            tree.root.updated_incumbent[] = true
            tree.incumbent = primal
            tree.lb =
                tree.root.problem.solving_stage == OPT_TREE_EMPTY ? primal - dual_gap : tree.lb
            tree.incumbent_solution.objective = tree.solutions[1].objective = primal
            tree.incumbent_solution.solution = tree.solutions[1].solution = x
        else
            if primal < tree.incumbent && tree.lb > primal - dual_gap
                @info "tree.lb > primal - dual_gap"
            else
                @info "primal >= tree.incumbent"
                @assert primal <= tree.incumbent + 1e-3 ||
                        isapprox(primal, tree.incumbent, atol=1e-6, rtol=1e-2) ||
                        primal <= tree.root.problem.f(prev_x_rounded) "primal <= tree.incumbent + 1e-3 ||
                        isapprox(primal, tree.incumbent, atol=1e-6, rtol=1e-2) || primal <= tree.root.problem.f(prev_x_rounded) : primal=$(primal) and tree.incumbent=$(tree.incumbent) and previous solution rounded $(tree.root.problem.f(prev_x_rounded))"
            end
            @info "postsolve did not improve the solution"
            primal = tree.incumbent_solution.objective = tree.solutions[1].objective
            x = tree.incumbent_solution.solution = tree.solutions[1].solution
        end
    end

    result[:primal_objective] = primal
    result[:dual_bound] = tree_lb(tree)
    result[:rel_dual_gap] = relative_gap(primal, tree_lb(tree))
    result[:dual_gap] = tree.incumbent - tree_lb(tree)
    result[:raw_solution] = x
    total_time_in_sec = (Dates.value(Dates.now() - time_ref)) / 1000.0
    result[:total_time_in_sec] = total_time_in_sec
    result[:status] = tree.root.problem.solving_stage
    result[:status_string] = status_string
    result[:solving_stage] = tree.root.problem.solving_stage

    if verbose
        println()

        println("Solution Statistics.")

        println("\t Solution Status: ", status_string)
        println("\t Primal Objective: ", primal)
        println("\t Dual Bound: ", tree_lb(tree))
        println("\t Dual Gap (relative): $(relative_gap(primal,tree_lb(tree)))\n")
        println("Search Statistics.")
        println("\t Total number of nodes processed: ", tree.num_nodes)
        if tree.root.options[:heu_ncalls] != 0
            println("\t LMO calls over all nodes: ", tree.root.problem.tlmo.ncalls)
            println("\t LMO calls in the heuristics: ", tree.root.options[:heu_ncalls])
            println(
                "\t Total number of lmo calls: ",
                tree.root.problem.tlmo.ncalls + tree.root.options[:heu_ncalls],
            )
        else
            println("\t Total number of lmo calls: ", tree.root.problem.tlmo.ncalls)
        end
        println("\t Total time (s): ", total_time_in_sec)
        println("\t LMO calls / sec: ", tree.root.problem.tlmo.ncalls / total_time_in_sec)
        println("\t Nodes / sec: ", tree.num_nodes / total_time_in_sec)
        println("\t LMO calls / node: $(tree.root.problem.tlmo.ncalls / tree.num_nodes)\n")
        if tree.root.options[:global_dual_tightening]
            println("\t Total number of global tightenings: ", sum(result[:global_tightenings]))
            println(
                "\t Global tightenings / node: ",
                round(
                    sum(result[:global_tightenings]) / length(result[:global_tightenings]),
                    digits=2,
                ),
            )
        end
        if tree.root.options[:dual_tightening]
            println("\t Total number of local tightenings: ", sum(result[:local_tightenings]))
            println(
                "\t Local tightenings / node: ",
                round(
                    sum(result[:local_tightenings]) / length(result[:local_tightenings]),
                    digits=2,
                ),
            )
            println(
                "\t Total number of potential local tightenings: ",
                sum(result[:local_potential_tightenings]),
            )
        end
        if isa(tree.options.branch_strategy, Boscia.Hierarchy)
            fraction_of_decisions = [
                (stage.decision_counter, stage.min_cutoff_counter) for
                stage in tree.options.branch_strategy.stages
            ]
            println("\t Decisions made: ", fraction_of_decisions)
        end
        if isa(tree.options.branch_strategy, Boscia.PseudocostBranching)
            println(
                "\t Number of alternative decisions: ",
                tree.options.branch_strategy.alt_decision_number,
            )
            println(
                "\t Number of stable decisions: ",
                tree.options.branch_strategy.stable_decision_number,
            )
            println(
                "\t Minimum number of branchings per variable: ",
                minimum(tree.options.branch_strategy.branch_tracker) - 1,
            )
        end
    end

    # Reset LMO
    int_bounds = IntegerBounds()
    build_LMO(
        tree.root.problem.tlmo,
        tree.root.problem.integer_variable_bounds,
        int_bounds,
        tree.root.problem.integer_variables,
    )

    return x
end
