# Interface.jl

"""
    solve(f, g, blmo::BoundedLinearMinimizationOracle; ...)

Requires

- `f` oracle of the objective function.
- `g` oracle of the gradient of the objective
- `blmo` encodes the feasible region and can handle additional bound constraints. This can either be a MIP solver instance (e.g., SCIP) or be a custom type (see `polytope_blmos.jl`). Has to be of type `BoundedLinearMinimizationOracle` (see `blmo_interface.jl`).

Returns

- `x` the best solution found.
- `tlmo` the BLMO wrapped in a TimeTrackingLMO instance.
- `result` a dictionary containg the statistics like number of nodes, total solving etc. It also contains information for plotting progress plots like the lower and upper bound progress.

Optional settings

- `traverse_strategy` encodes how to choose the next node for evaluation. By default the node with the best lower bound is picked.
- `branching_strategy` fixes the branching strategy. By default, weuse `MOST_INFEASIBLE`, i.e. we branch on the entry which is the farthest away from being an integer.
- `variant` the Frank-Wolfe variant to be used to solve the node problem. Options currently available are `AwayFrankWolfe`, `Blended`, `BPCG` and `VanillaFrankWolfe`.
- `line_search` specifies the line search method used in the FrankWolfe variant. Default is the `Adaptive` line search. For other available types, check the FrankWolfe.jl package.
- `active_set` can be used to specify a starting point. By default, the direction (1,..,n) where n is the size of the problem is used to find a start vertex. This has to be of the type `FrankWolfe.ActiveSet`. Beware that the active set may only contain actual vertices of the feasible region.
- `lazy` flag specifies whether the lazification of the Frank-Wolfe variant should be used. Per default `true`. Note that it has no effect on Vanilla Frank-Wolfe.
- `lazy_tolerance` decides how much progress is deemed enough to not have to call the LMO. Only used if the `lazy` flag is activated.
- `fw_epsilon` the solving precision of Frank-Wolfe at the root node.
- `verbose` if `true`, logs and solution statistics are printed.
- `dual_gap` absolute dual gap. If  reached, the algorithm stops.
- `rel_dual_gap` relative dual gap. If reached, the algorithm stops.
- `time_limit` algorithm will stop if the time limit is reached. Depending on the problem it is possible that no feasible solution has been found yet. On default, there is no time limit.
- `print_iter` encodes after how many processed nodes the current node and solution status is printed. The logs are always printed if a new integral solution has been found.
- `dual_gap_decay_factor` the FrankWolfe tolerance at a given level `i` in the tree is given by `fw_epsilon * dual_gap_decay_factor^i` until we reach the `min_node_fw_epsilon`.
- `max_fw_iter` maximum number of iterations in a Frank-Wolfe run.
- `min_number_lower` if not `Inf`, evaluation of a node is stopped if at least `min_number_lower` open nodes have a better lower bound.
- `min_node_fw_epsilon` smallest fw epsilon tolerance, see also `dual_gap_decay_factor`.
- `use_postsolve` if `true`, runs the specified Frank-Wolfe variant on the problem with the integral variables fixed to the solution, i.e. it only optimizes over the continuous variables. This might improve the solution if one has many continuous variables.
- `min_fw_iterations` the minimum number of Frank-Wolfe iterations performed in the node evaluation.
- `max_iteration_post` maximum number of iterations in the Frank-Wolfe run during postsolve.
- `dual_tightening` flag to decide  whether to use dual tightening techniques at node level. Note that this only porvides valid tightenings if your function is convex!
- `global_dual_tightening` flag to decide whether to generate dual tightenings from new solutions that are gloablly valid.
- `bnb_callback` an optional callback called after every node evaluation.
- `strong_convexity` strong convexity parameter of the objective `f`, used for tightening the dual bound at every node.
- `sharpness_constant` - the constant `M > 0` for `(θ, M)`-sharpness. `f` is `(θ, M)`-sharpness: `f` satisfies `min_{x^* ∈ X^*} || x - x^* || ≤ M (f(x) - f^(x^*))^θ` where `X^*` is the set of minimizer of `f`. Note that tightenings using sharpness are only valid if the problem has a unique minimizer, i.e. `f` is stricly convex!
- `sharpness_exponent` - the exponent `θ ∈ [0, 1/2]` for `(θ, M)`-sharpness.
- `domain_oracle` given a point `x`: returns `true` if `x` is in the domain of `f`, else false. Per default, it always returns `true`. In case of the non-trivial domain oracle, the starting point has to be domain feasible for `f`. Additionally, the user has to provide a function `domain_point`, see below. Also, depending on the line search method, you might have to provide the domain oracle to it, too.
- `find_domain_point` given the current node bounds return a domain feasible point respecting the bounds. If no such point can be found, return `nothing`.
- `start_solution` an initial solution can be provided if known. It will be used as the initial incumbent.
- `fw_verbose` if `true`, the Frank-Wolfe logs are printed at each node. Mostly meant for debugging.
- `use_shadow_set` the shadow set is the set of discarded vertices which is inherited by the children nodes. It is used to avoid recomputing of vertices in case the BLMO is expensive. In case of a cheap BLMO, performance might improve by disabling this option.
- `custom_heuristics` list of custom heuristics from the user. 
- `prob_rounding` the probability for calling the simple rounding heuristic. Since the feasibility has to be checked, it might be expensive to do this for every node. Per default, this is activated for every node.
- `clean_solutions` flag deciding whether new solutions should be polished. They will be rounded and then a quick Frank-Wolfe run will be started.
- `max_clean_iter` maximum number of iterations in the Frank-Wolfe call for polishing new solutions.
- `use_strong_lazy` specifies strong lazification in DICG. Otherwise, weak version is used.
- `use_DICG_warm_start` if `true`, enables DICG-specific warm-start strategy.
- `use_strong_warm_start` if `true`, performs additional in-face check for vertices.
- `build_dicg_start_point` default generates a random feasible vertex.
"""
function solve(
    f,
    grad!,
    blmo::BoundedLinearMinimizationOracle;
    traverse_strategy=Bonobo.BestFirstSearch(),
    branching_strategy=Bonobo.MOST_INFEASIBLE(),
    variant::FrankWolfeVariant=BPCG(),
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Adaptive(),
    active_set::Union{Nothing,FrankWolfe.ActiveSet}=nothing,
    lazy=true,
    lazy_tolerance=2.0,
    fw_epsilon=1e-2,
    verbose=false,
    dual_gap=1e-6,
    rel_dual_gap=1.0e-2,
    time_limit=Inf,
    node_limit=Inf,
    print_iter=100,
    dual_gap_decay_factor=0.8,
    max_fw_iter=10000,
    fw_timeout=Inf,
    min_number_lower=Inf,
    min_node_fw_epsilon=1e-6,
    use_postsolve=true,
    min_fw_iterations=5,
    max_iteration_post=10000,
    dual_tightening=true,
    global_dual_tightening=true,
    bnb_callback=nothing,
    strong_convexity=0.0,
    sharpness_constant = 0.0,
    sharpness_exponent = Inf,
    domain_oracle=_trivial_domain,
    find_domain_point= _trivial_domain_point,
    start_solution=nothing,
    fw_verbose=false,
    use_shadow_set=true,
    custom_heuristics=[Heuristic()],
    post_heuristics_callback=nothing,
    rounding_prob=1.0,
    clean_solutions=false,
    max_clean_iter=10,
    no_pruning=false,
    ignore_lower_bound=false,
    add_all_solutions=false,
    propagate_bounds=nothing,
    use_strong_lazy=false,
    use_DICG_warm_start=false,
    use_strong_warm_start=false,
    build_dicg_start_point = trivial_build_dicg_start_point,
    kwargs...,
)
    time_ref = Dates.now()

    if variant == DICG()
        if !is_decomposition_invariant_oracle(blmo)
            error("DICG within Boscia is not implemented for $(typeof(blmo)).")
        end
    end
    if verbose
        println("\nBoscia Algorithm.\n")
        println("Parameter settings.")
        println("\t Tree traversal strategy: ", _value_to_print(traverse_strategy))
        println("\t Branching strategy: ", _value_to_print(branching_strategy))
        println("\t FrankWolfe variant: $(variant)")
        println("\t Line Search Method: $(line_search)")
        println("\t Lazification: $(lazy)")
        lazy ? println("\t Lazification Tolerance: $(lazy_tolerance)") : nothing
        @printf("\t Absolute dual gap tolerance: %e\n", dual_gap)
        @printf("\t Relative dual gap tolerance: %e\n", rel_dual_gap)
        @printf("\t Frank-Wolfe subproblem tolerance: %e\n", fw_epsilon)
        @printf("\t Frank-Wolfe dual gap decay factor: %e\n", dual_gap_decay_factor)
        println("\t Additional kwargs: ", join(keys(kwargs), ","))
    end

    n, _ = get_list_of_variables(blmo)

    integer_variables = Vector{Int}()
    num_int = 0
    num_bin = 0
    for c_idx in get_integer_variables(blmo)
        push!(integer_variables, c_idx)
        num_int += 1
    end
    time_lmo = TimeTrackingLMO(blmo, integer_variables, time_ref, Float64(time_limit))

    if num_int == 0
        @warn("No integer variables detected! Please use an MIP solver!")
    end

    if verbose
        println("\t Total number of variables: ", n)
        println("\t Number of integer variables: $(num_int)\n")
    end

    global_bounds = build_global_bounds(blmo, integer_variables)

    if typeof(domain_oracle) != typeof(_trivial_domain) && typeof(find_domain_point) == typeof(_trivial_domain_point)
        @warn "For a non trivial domain oracle, please provide the DOMAIN POINT function. Otherwise, Boscia might not converge."
    end

    v = []
    if active_set === nothing
        direction = collect(1.0:n)
        v = compute_extreme_point(blmo, direction)
        v[integer_variables] = round.(v[integer_variables])
        @assert isfinite(f(v))
        active_set = FrankWolfe.ActiveSet([(1.0, v)])
        vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)
    else
        @assert FrankWolfe.active_set_validate(active_set)
        for a in active_set.atoms
            @assert is_linear_feasible(blmo, a)
        end
        x = FrankWolfe.compute_active_set_iterate!(active_set)
        v = x
        @assert isfinite(f(x))
    end
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    pre_computed_set = use_DICG_warm_start ? [v] : nothing

    m = SimpleOptimizationProblem(f, grad!, n, integer_variables, time_lmo, global_bounds)
    nodeEx = FrankWolfeNode(
        NodeInfo(1, f(v), f(v)),
        active_set,
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

    # Create standard heuristics
    heuristics = vcat([Heuristic(rounding_heuristic, rounding_prob, :rounding)], custom_heuristics)

    # If we cannot trust the lower bound, we also shouldn't do any tighening.
    if ignore_lower_bound
        dual_tightening=false
        global_dual_tightening=false
    end

    Node = typeof(nodeEx)
    Value = Vector{Float64}
    tree = Bonobo.initialize(;
        traverse_strategy=traverse_strategy,
        Node=Node,
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
            options=Dict{Symbol,Any}(
                :domain_oracle => domain_oracle,
                :find_domain_point => find_domain_point,
                :dual_gap => dual_gap,
                :dual_gap_decay_factor => dual_gap_decay_factor,
                :dual_tightening => dual_tightening,
                :fwVerbose => fw_verbose,
                :global_dual_tightening => global_dual_tightening,
                :lazy => lazy,
                :lazy_tolerance => lazy_tolerance,
                :lineSearch => line_search,
                :min_node_fw_epsilon => min_node_fw_epsilon,
                :max_fw_iter => max_fw_iter,
                :fw_timeout => fw_timeout,
                :print_iter => print_iter,
                :strong_convexity => strong_convexity,
                :sharpness_constant => sharpness_constant,
                :sharpness_exponent => sharpness_exponent,
                :time_limit => time_limit,
                :node_limit => node_limit,
                :usePostsolve => use_postsolve,
                :variant => variant,
                :use_shadow_set => use_shadow_set,
                :heuristics => heuristics,
                :post_heuristics_callback => post_heuristics_callback,
                :heu_ncalls => 0,
                :max_clean_iter => max_clean_iter,
                :clean_solutions => clean_solutions,
                :no_pruning => no_pruning,
                :ignore_lower_bound => ignore_lower_bound,
                :add_all_solutions => add_all_solutions,
                :propagate_bounds => propagate_bounds,
                :use_strong_lazy => use_strong_lazy,
                :use_strong_warm_start => use_strong_warm_start,
                :use_strong_lazy => use_strong_lazy,
                :build_dicg_start_point => build_dicg_start_point,
            ),
            result=Dict{Symbol,Any}(),
        ),
        branch_strategy=branching_strategy,
        dual_gap_limit=rel_dual_gap,
        abs_gap_limit=dual_gap,
    )
    Bonobo.set_root!(
        tree,
        (
            active_set=active_set,
            discarded_vertices=vertex_storage,
            local_bounds=IntegerBounds(),
            level=1,
            fw_dual_gap_limit=fw_epsilon,
            fw_time=Millisecond(0),
            global_tightenings=0,
            local_tightenings=0,
            local_potential_tightenings=0,
            dual_gap=-Inf,
            pre_computed_set=pre_computed_set,
        ),
    )

    if start_solution !== nothing
        if size(start_solution) != size(v)
            error(
                "size of starting solution differs from vertices: $(size(start_solution)), $(size(v))",
            )
        end
        # Sanity check that the provided solution is in fact feasible.
        @assert is_linear_feasible(blmo, start_solution) &&
                is_integer_feasible(tree, start_solution)
        node = tree.nodes[1]
        add_new_solution!(tree, node, f(start_solution), start_solution, :start)
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
        verbose,
        fw_iterations,
        list_active_set_size_cb,
        list_discarded_set_size_cb,
        tree.root.result,
        lmo_calls_per_layer,
        active_set_size_per_layer,
        discarded_set_size_per_layer,
        node_level,
        bnb_callback,
        global_tightenings,
        local_tightenings,
        local_potential_tightenings,
        num_bin,
        num_int,
    )

    fw_callback = build_FW_callback(
        tree, 
        min_number_lower, 
        true, 
        fw_iterations, 
        min_fw_iterations, 
        time_ref, 
        tree.root.options[:time_limit], 
        use_DICG=tree.root.options[:variant] == DICG(),
    )

    tree.root.options[:callback] = fw_callback
    tree.root.current_node_id[] = Bonobo.get_next_node(tree, tree.options.traverse_strategy).id

    Bonobo.optimize!(tree; callback=bnb_callback)

    x = postsolve(tree, tree.root.result, time_ref, verbose, max_iteration_post)

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
    println() # cleaner output

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
    else
        status_string = "Optimal (tolerance reached)"
        tree.root.problem.solving_stage = OPT_GAP_REACHED
    end

    only_integer_vars = tree.root.problem.nvars == length(tree.root.problem.integer_variables)
    if tree.root.options[:usePostsolve] && !only_integer_vars
        # Build solution lmo
        fix_bounds = IntegerBounds()
        for i in tree.root.problem.integer_variables
            push!(fix_bounds, (i => round(x[i])), :lessthan)
            push!(fix_bounds, (i => round(x[i])), :greaterthan)
        end

        free_model(tree.root.problem.tlmo.blmo)
        build_LMO(
            tree.root.problem.tlmo,
            tree.root.problem.integer_variable_bounds,
            fix_bounds,
            tree.root.problem.integer_variables,
        )
        # previous solution rounded to account for 0.99999.. or 1.00000000002 types of values
        prev_x_rounded = copy(x)
        prev_x_rounded[tree.root.problem.integer_variables] .= round.(prev_x_rounded[tree.root.problem.integer_variables])
        prev_x_rounded = is_linear_feasible(tree.root.problem.tlmo, prev_x_rounded) ? prev_x_rounded : x

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
            line_search=FrankWolfe.Adaptive(verbose=false),
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
    result[:status] = status_string

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
            println("\t Total number of lmo calls: ", tree.root.problem.tlmo.ncalls + tree.root.options[:heu_ncalls])
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
