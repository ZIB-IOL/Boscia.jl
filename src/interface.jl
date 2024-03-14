"""
    solve
   
f                      - objective function oracle. 
g                      - oracle for the gradient of the objective. 
blmo                    - a MIP solver instance (e.g., SCIP) encoding the feasible region. Has to be of type `BoundedLinearMinimizationOracle` (see `lmo_wrapper.jl`).    
traverse_strategy      - encodes how to choose the next node for evaluation. 
                        By default the node with the best lower bound is picked.
branching_strategy     - by default we branch on the entry which is the farthest 
                        away from being an integer.
variant                - variant of FrankWolfe to be used to solve the node problem.
                         Options: FW   -- Vanilla FrankWolfe
                                 AFW  -- Away FrankWolfe
                                 BPCG -- Blended Pairwise Conditional Gradient  
line_search            - specifies the Line Search method used in the FrankWolfe variant.
                         Default is the Adaptive Line Search. For other types, check the FrankWolfe.jl package. 
active_set             - can be used to specify a starting point, e.g. if the feasible region is not completely
                         contained in the domain of the objective. By default, the direction (1,..,n) where n is 
                         the size of the problem is used to find a start vertex. Beware that the active set may 
                         only contain actual vertices of the feasible region.   
lazy                   - specifies whether the lazification shoud be used. Per default true. 
                         Beware that it has no effect with Vanilla Frank-Wolfe. 
lazy_tolerance         - decides how much progress is deemed enough to not have to call the LMO.                                                             
fw_epsilon             - the tolerance for FrankWolfe in the root node.
verbose                - if true, a log and solution statistics are printed.
dual_gap               - if this absolute dual gap is reached, the algorithm stops.
rel_dual_gap           - if this relative dual gap is reached, the algorithm stops.
time_limit             - algorithm will stop if the time limit is reached. Depending on the problem
                        it is possible that no feasible solution has been found yet.     
print_iter             - encodes after how many proccessed nodes the current node and solution status 
                        is printed. Will always print if a new integral solution has been found. 
dual_gap_decay_factor  - the FrankWolfe tolerance at a given level i in the tree is given by 
                        fw_epsilon * dual_gap_decay_factor^i until we reach the min_node_fw_epsilon.
max_fw_iter            - maximum number of iterations in a FrankWolfe run.
min_number_lower       - If not Inf, evaluation of a node is stopped if at least min_number_lower nodes have a better 
                        lower bound.
min_node_fw_epsilon    - smallest fw epsilon possible, see dual_gap_decay_factor.
use_postsolve          - Runs the specified Frank-Wolfe variant on the problem with the integral variables fixed to the solution, i.e.
                        it only optimizes over the continuous variables. This might improve the solution if one has many continuous variables. 
min_fw_iterations      - the minimum number of FrankWolfe iterations in the node evaluation. 
max_iteration_post     - maximum number of iterations in a FrankWolfe run during postsolve
dual_tightening        - whether to use dual tightening techniques (make sure your function is convex!)
global_dual_tightening - dual tightening maintained globally valid (when new solutions are found)
bnb_callback           - an optional callback called at every node of the tree, for example for heuristics
strong_convexity       - strong convexity of the function, used for tighter dual bound at every node
domain_oracle          - For a point x: returns true if x is in the domain of f, else false. Per default is true.
                         In case of the non trivial domain oracle, the starting point has to be feasible for f. Also, depending 
                         on the Line Search method, you might have to provide the domain oracle to it, too.
start_solution         - initial solution to start with an incumbent
fw_verbose             - if true, FrankWolfe logs are printed
use_shadow_set         - The shadow set is the set of discarded vertices which is inherited by the children nodes.
                        It is used to avoid recomputing of vertices in case the LMO is expensive. In case of a cheap LMO,
                        performance might improve by disabling this option. 
custom_heuristics      - List of  custom heuristic from the user.    
prob_rounding          - The probability for calling the rounding heuristics. Since the feasibility has to be checked, it might
                        expensive to do this for every node.                    
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
    print_iter=100,
    dual_gap_decay_factor=0.8,
    max_fw_iter=10000,
    min_number_lower=Inf,
    min_node_fw_epsilon=1e-6,
    use_postsolve=true,
    min_fw_iterations=5,
    max_iteration_post=10000,
    dual_tightening=true,
    global_dual_tightening=true,
    bnb_callback=nothing,
    strong_convexity=0.0,
    domain_oracle=x -> true,
    start_solution=nothing,
    fw_verbose=false,
    use_shadow_set=true,
    custom_heuristics=[Heuristic()],
    rounding_prob=1.0,
    kwargs...,
)
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
    time_lmo = TimeTrackingLMO(blmo, integer_variables)

    if num_int == 0
        error("No integer variables detected! Please use an MIP solver!")
    end

    if verbose
        println("\t Total number of variables: ", n)
        println("\t Number of integer variables: $(num_int)\n")
    end

    global_bounds = build_global_bounds(blmo, integer_variables)

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
        v = active_set.atoms[1]
        x = FrankWolfe.compute_active_set_iterate!(active_set)
        @assert isfinite(f(x))
    end
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

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
    )

    # Create standard heuristics
    heuristics = vcat([Heuristic(rounding_heuristic, rounding_prob, :rounding)], custom_heuristics)

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
                :print_iter => print_iter,
                :strong_convexity => strong_convexity,
                :time_limit => time_limit,
                :usePostsolve => use_postsolve,
                :variant => variant,
                :use_shadow_set => use_shadow_set,
                :heuristics => heuristics, 
                :heu_ncalls => 0,
            ),
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
        sol = FrankWolfeSolution(f(start_solution), start_solution, node, :start)
        push!(tree.solutions, sol)
        if tree.incumbent_solution === nothing || sol.objective < tree.incumbent_solution.objective
            tree.incumbent_solution = sol
            tree.incumbent = sol.objective
        end
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
    result = Dict{Symbol,Any}()
    lmo_calls_per_layer = Vector{Vector{Int}}()
    active_set_size_per_layer = Vector{Vector{Int}}()
    discarded_set_size_per_layer = Vector{Vector{Int}}()
    time_ref = Dates.now()
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
        result,
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

    fw_callback = build_FW_callback(tree, min_number_lower, true, fw_iterations, min_fw_iterations, time_ref, tree.root.options[:time_limit])

    tree.root.options[:callback] = fw_callback
    tree.root.current_node_id[] = Bonobo.get_next_node(tree, tree.options.traverse_strategy).id

    Bonobo.optimize!(tree; callback=bnb_callback)

    x = postsolve(tree, result, time_ref, verbose, max_iteration_post)

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

    return x, tree.root.problem.tlmo, result
end

"""
Output of Boscia

    iter :          current iteration of Boscia
    node id :       current node id
    lower bound :   tree_lb(tree)
    incumbent :     tree.incumbent
    gap :           tree.incumbent-tree_lb(tree)
    rel. gap :      dual_gap/tree.incumbent
    time :          total time of Boscia
    time/nodes :    average time per node
    FW time :       time spent in FW 
    LMO time :      time used by LMO
    LMO calls :     number of compute_extreme_point calls in FW
    FW iterations : number of iterations in FW
"""
function build_bnb_callback(
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
    result,
    lmo_calls_per_layer,
    active_set_size_per_layer,
    discarded_set_size_per_layer,
    node_level,
    baseline_callback,
    local_tightenings,
    global_tightenings,
    local_potential_tightenings,
    num_bin,
    num_int,
)
    iteration = 0

    headers = [
        " ",
        "Iter",
        "Open",
        "Bound",
        "Incumbent",
        "Gap (abs)",
        "Gap (rel)",
        "Time (s)",
        "Nodes/sec",
        "FW (ms)",
        "LMO (ms)",
        "LMO (calls c)",
        "FW (its)",
        "#activeset",
        "#shadow",
    ]
    format_string = "%1s %5i %5i %14e %14e %14e %14e %14e %14e %12i %10i %14i %10i %8i %8i\n"
    print_iter = get(tree.root.options, :print_iter, 100)

    if verbose
        FrankWolfe.print_callback(headers, format_string, print_header=true)
    end
    return function callback(
        tree,
        node;
        worse_than_incumbent=false,
        node_infeasible=false,
        lb_update=false,
    )
        if baseline_callback !== nothing
            baseline_callback(
                tree,
                node,
                worse_than_incumbent=worse_than_incumbent,
                node_infeasible=node_infeasible,
                lb_update=lb_update,
            )
        end
        if !node_infeasible
            #update lower bound
            if lb_update == true
                tree.node_queue[node.id] = (node.lb, node.id)
                _, prio = peek(tree.node_queue)
                @assert tree.lb <= prio[1]
                tree.lb = min(minimum([prio[2][1] for prio in tree.node_queue]), tree.incumbent)
            end
            push!(list_ub_cb, tree.incumbent)
            push!(list_num_nodes_cb, tree.num_nodes)
            push!(node_level, node.level)
            iteration += 1
            if tree.lb == -Inf && isempty(tree.nodes)
                tree.lb = node.lb
            end

            time = float(Dates.value(Dates.now() - time_ref))
            push!(list_time_cb, time)

            if tree.root.options[:time_limit] < Inf
                if time / 1000.0 ≥ tree.root.options[:time_limit]
                    @assert tree.root.problem.solving_stage == SOLVING
                    tree.root.problem.solving_stage = TIME_LIMIT_REACHED
                end
            end

            fw_time = Dates.value(node.fw_time)
            fw_iter = if !isempty(fw_iterations)
                fw_iterations[end]
            else
                0
            end
            if !isempty(tree.root.problem.tlmo.optimizing_times)
                LMO_time = sum(1000 * tree.root.problem.tlmo.optimizing_times)
                empty!(tree.root.problem.tlmo.optimizing_times)
            else
                LMO_time = 0
            end
            LMO_calls_c = tree.root.problem.tlmo.ncalls
            push!(list_lmo_calls_cb, copy(LMO_calls_c))

            if !isempty(tree.node_queue)
                p_lb = tree.lb
                tree.lb = min(minimum([prio[2][1] for prio in tree.node_queue]), tree.incumbent)
                @assert p_lb <= tree.lb + tree.root.options[:dual_gap] "p_lb <= tree.lb + tree.root.options[:dual_gap] $(p_lb) <= $(tree.lb + tree.root.options[:dual_gap])"
            end
            # correct lower bound if necessary
            tree.lb = tree_lb(tree)
            dual_gap = tree.incumbent - tree_lb(tree)
            push!(list_lb_cb, tree_lb(tree))
            active_set_size = length(node.active_set)
            discarded_set_size = length(node.discarded_vertices.storage)
            push!(list_active_set_size_cb, active_set_size)
            push!(list_discarded_set_size_cb, discarded_set_size)
            nodes_left = length(tree.nodes)
            if tree.root.updated_incumbent[]
                incumbent_updated = "*"
            else
                incumbent_updated = " "
            end
            if verbose && (
                mod(iteration, print_iter) == 0 ||
                iteration == 1 ||
                Bonobo.terminated(tree) ||
                tree.root.updated_incumbent[]
            )
                if (mod(iteration, print_iter * 40) == 0)
                    FrankWolfe.print_callback(headers, format_string, print_header=true)
                end
                FrankWolfe.print_callback(
                    (
                        incumbent_updated,
                        iteration,
                        nodes_left,
                        tree_lb(tree),
                        tree.incumbent,
                        dual_gap,
                        relative_gap(tree.incumbent, tree_lb(tree)),
                        time / 1000.0,
                        tree.num_nodes / time * 1000.0,
                        fw_time,
                        LMO_time,
                        tree.root.problem.tlmo.ncalls,
                        fw_iter,
                        active_set_size,
                        discarded_set_size,
                    ),
                    format_string,
                    print_header=false,
                )
                tree.root.updated_incumbent[] = false
            end
            # lmo calls per layer
            if length(list_lmo_calls_cb) > 1
                LMO_calls = list_lmo_calls_cb[end] - list_lmo_calls_cb[end-1]
            else
                LMO_calls = list_lmo_calls_cb[end]
            end
            if length(lmo_calls_per_layer) < node.level
                push!(lmo_calls_per_layer, [LMO_calls])
                push!(active_set_size_per_layer, [active_set_size])
                push!(discarded_set_size_per_layer, [discarded_set_size])
            else
                push!(lmo_calls_per_layer[node.level], LMO_calls)
                push!(active_set_size_per_layer[node.level], active_set_size)
                push!(discarded_set_size_per_layer[node.level], discarded_set_size)
            end

            # add tightenings
            push!(global_tightenings, node.global_tightenings)
            push!(local_tightenings, node.local_tightenings)
            push!(local_potential_tightenings, node.local_potential_tightenings)
            @assert node.local_potential_tightenings <= num_bin + num_int
            @assert node.local_tightenings <= num_bin + num_int
            @assert node.global_tightenings <= num_bin + num_int
        end
        # update current_node_id
        if !Bonobo.terminated(tree)
            tree.root.current_node_id[] =
                Bonobo.get_next_node(tree, tree.options.traverse_strategy).id
        end

        if Bonobo.terminated(tree)
            Bonobo.sort_solutions!(tree.solutions, tree.sense)
            x = Bonobo.get_solution(tree)
            # x can be nothing if the user supplied a custom domain oracle and the time limit is reached
            if x === nothing
                @assert tree.root.problem.solving_stage == TIME_LIMIT_REACHED
            end
            primal_value = x !== nothing ? tree.root.problem.f(x) : Inf
            # deactivate postsolve if there is no solution
            tree.root.options[:usePostsolve] =
                x === nothing ? false : tree.root.options[:usePostsolve]

            # TODO: here we need to calculate the actual state

            # If the tree is empty, incumbent and solution should be the same!
            if isempty(tree.nodes)
                @assert isapprox(tree.incumbent, primal_value)
            end

            result[:number_nodes] = tree.num_nodes
            result[:lmo_calls] = tree.root.problem.tlmo.ncalls
            result[:heu_lmo_calls] = tree.root.options[:heu_ncalls]
            result[:list_num_nodes] = list_num_nodes_cb
            result[:list_lmo_calls_acc] = list_lmo_calls_cb
            result[:list_active_set_size] = list_active_set_size_cb
            result[:list_discarded_set_size] = list_discarded_set_size_cb
            result[:list_lb] = list_lb_cb
            result[:list_ub] = list_ub_cb
            result[:list_time] = list_time_cb
            result[:lmo_calls_per_layer] = lmo_calls_per_layer
            result[:active_set_size_per_layer] = active_set_size_per_layer
            result[:discarded_set_size_per_layer] = discarded_set_size_per_layer
            result[:node_level] = node_level
            result[:global_tightenings] = global_tightenings
            result[:local_tightenings] = local_tightenings
            result[:local_potential_tightenings] = local_potential_tightenings

            if verbose
                FrankWolfe.print_callback(headers, format_string, print_footer=true)
                println()
            end
        end
    end
end

"""
    postsolve(tree, result, time_ref, verbose)

Runs the post solve both for a cleaner solutiona and to optimize 
for the continuous variables if present.
Prints solution statistics if verbose is true.        
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
                        isapprox(primal, tree.incumbent, atol=1e-6, rtol=1e-2)
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
