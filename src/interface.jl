"""
    solve
   
f                     - objective function oracle. 
g                     - oracle for the gradient of the objective. 
lmo                   - a MIP solver instance (SCIP) encoding the feasible region.    
traverse_strategy     - encodes how to choose the next node for evaluation. 
                        By default the node with the best lower bound is picked.
branching_strategy    - by default we branch on the entry which is the farthest 
                        away from being an integer.
fw_epsilon            - the tolerance for FrankWolfe in the root node.
verbose               - if true, a log and solution statistics are printed.
dual_gap              - if this absolute dual gap is reached, the algorithm stops.
rel_dual_gap          - if this relative dual gap is reached, the algorithm stops.
time_limit            - algorithm will stop if the time limit is reached. Depending on the problem
                        it is possible that no feasible solution has been found yet.     
print_iter            - encodes after how manz proccessed nodes the current node and solution status 
                        is printed. Will always print if a new integral solution has been found. 
dual_gap_decay_factor - the FrankWolfe tolerance at a given level i in the tree is given by 
                        fw_epsilon * dual_gap_decay_factor^i until we reach the min_node_fw_epsilon.
max_fw_iter           - maximum number of iterations in a FrankWolfe run.
min_number_lower      - If not Inf, evaluation of a node is stopped if at least min_number_lower nodes have a better 
                        lower bound.
min_node_fw_epsilon   - smallest fw epsilon possible, see dual_gap_decay_factor.
min_fw_iterations     - the minimum number of FrankWolfe iterations in the node evaluation. 
max_iteration_post    - maximum number of iterations in a FrankWolfe run during postsolve
"""
function solve(
    f,
    grad!,
    lmo;
    traverse_strategy=Bonobo.BFS(),
    branching_strategy=Bonobo.MOST_INFEASIBLE(),
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
    kwargs...,
)
    if verbose
        println("\nBoscia Algorithm.\n")
        println("Parameter settings.")
        println("\t Tree traversal strategy: ", _value_to_print(traverse_strategy))
        println("\t Branching strategy: ", _value_to_print(branching_strategy))
        @printf("\t Absolute dual gap tolerance: %e\n", dual_gap)
        @printf("\t Relative dual gap tolerance: %e\n", rel_dual_gap)
        @printf("\t Frank-Wolfe subproblem tolerance: %e\n", fw_epsilon)
        @printf("\t Frank-Wolfe dual gap decay factor: %e\n", dual_gap_decay_factor)
    end

    v_indices = MOI.get(lmo.o, MOI.ListOfVariableIndices())
    n = length(v_indices)
    if v_indices != MOI.VariableIndex.(1:n)
        error("Variables are expected to be contiguous and ordered from 1 to N")
    end
    time_lmo = Boscia.TimeTrackingLMO(lmo)

    integer_variables = Vector{Int}()
    num_int = 0
    num_bin = 0
    for cidx in MOI.get(lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}())
        push!(integer_variables, cidx.value)
        num_int += 1
    end
    for cidx in MOI.get(lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.ZeroOne}())
        push!(integer_variables, cidx.value)
        num_bin += 1
    end

    if num_bin == 0 && num_int == 0
        error("No integer or binary variables detected! Please use an IP solver!")
    end

    if verbose
        println("\t Total number of variables: ", n)
        println("\t Number of integer variables: ", num_int)
        println("\t Number of binary variables: $(num_bin)\n")
    end

    global_bounds = Boscia.IntegerBounds()
    for idx in integer_variables
        for ST in (MOI.LessThan{Float64}, MOI.GreaterThan{Float64})
            cidx = MOI.ConstraintIndex{MOI.VariableIndex,ST}(idx)
            # Variable constraints to not have to be explicitly given, see Buchheim example
            if MOI.is_valid(lmo.o, cidx)
                s = MOI.get(lmo.o, MOI.ConstraintSet(), cidx)
                push!(global_bounds, (idx, s))
            end
        end
        cidx = MOI.ConstraintIndex{MOI.VariableIndex,MOI.Interval{Float64}}(idx)
        if MOI.is_valid(lmo.o, cidx)
            x = MOI.VariableIndex(idx)
            s = MOI.get(lmo.o, MOI.ConstraintSet(), cidx)
            MOI.delete(lmo.o, cidx)
            MOI.add_constraint(lmo.o, x, MOI.GreaterThan(s.lower))
            MOI.add_constraint(lmo.o, x, MOI.LessThan(s.upper))
            push!(global_bounds, (idx, MOI.GreaterThan(s.lower)))
            push!(global_bounds, (idx, MOI.LessThan(s.upper)))
        end
        @assert !MOI.is_valid(lmo.o, cidx)
    end

    direction = collect(1.0:n)
    v = compute_extreme_point(lmo, direction)
    active_set = FrankWolfe.ActiveSet([(1.0, v)])
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    m = Boscia.SimpleOptimizationProblem(f, grad!, n, integer_variables, time_lmo, global_bounds)
    nodeEx = Boscia.FrankWolfeNode(
        Bonobo.BnBNodeInfo(1, 0.0, 0.0),
        active_set,
        vertex_storage,
        Boscia.IntegerBounds(),
        1,
        1e-3,
        Millisecond(0),
    )

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
            options=Dict{Symbol,Any}(
                :dual_gap_decay_factor => dual_gap_decay_factor,
                :dual_gap => dual_gap,
                :print_iter => print_iter,
                :max_fw_iter => max_fw_iter,
                :min_node_fw_epsilon => min_node_fw_epsilon,
                :time_limit => time_limit,
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
            local_bounds=Boscia.IntegerBounds(),
            level=1,
            fw_dual_gap_limit=fw_epsilon,
            fw_time=Millisecond(0),
        ),
    )

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
    )

    fw_callback = build_FW_callback(tree, min_number_lower, true, fw_iterations, min_fw_iterations)

    tree.root.options[:callback] = fw_callback
    tree.root.current_node_id[] = Bonobo.get_next_node(tree, tree.options.traverse_strategy).id

    Bonobo.optimize!(tree; callback=bnb_callback)

    x = postsolve(tree, result, time_ref, verbose, use_postsolve, max_iteration_post)

    # Check solution and polish
    x_polished = x
    if !is_linear_feasible(tree.root.problem.lmo, x)
        error("Reported solution not linear feasbile!")
    end
    if !is_integer_feasible(tree.root.problem.integer_variables, x, atol=1e-16, rtol=1e-16)
        @info "Polish solution"
        for i in tree.root.problem.integer_variables
            x_polished[i] = round(x_polished[i])
        end
        if !is_linear_feasible(tree.root.problem.lmo, x_polished)
            @warn "Polished solution not linear feasible"
        else
            x = x_polished
        end
    end
    println() # cleaner output

    return x, tree.root.problem.lmo, result
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
)
    iteration = 0

    headers = [
        " ",
        "Iteration",
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
        "FW (Its)",
        "#ActiveSet",
        "Discarded",
    ]
    format_string = "%1s %10i %10i %14e %14e %14e %14e %14e %14e %14i %14i %14i %10i %12i %10i\n"
    print_callback = FrankWolfe.print_callback
    print_iter = get(tree.root.options, :print_iter, 100)

    if verbose
        print_callback(headers, format_string, print_header=true)
    end
    return function callback(
        tree,
        node;
        worse_than_incumbent=false,
        node_infeasible=false,
        lb_update=false,
    )
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
            if !isempty(tree.root.problem.lmo.optimizing_times)
                LMO_time = sum(1000 * tree.root.problem.lmo.optimizing_times)
                empty!(tree.root.problem.lmo.optimizing_times)
            else
                LMO_time = 0
            end
            LMO_calls_c = tree.root.problem.lmo.ncalls
            push!(list_lmo_calls_cb, copy(LMO_calls_c))

            if !isempty(tree.node_queue)
                p_lb = tree.lb
                tree.lb = min(minimum([prio[2][1] for prio in tree.node_queue]), tree.incumbent)
                @assert p_lb <= tree.lb
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
                    print_callback(headers, format_string, print_header=true)
                end
                print_callback(
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
                        tree.root.problem.lmo.ncalls,
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

        end
        # update current_node_id
        if !Bonobo.terminated(tree)
            tree.root.current_node_id[] =
                Bonobo.get_next_node(tree, tree.options.traverse_strategy).id
        end

        if Bonobo.terminated(tree)
            Bonobo.sort_solutions!(tree.solutions, tree.sense)
            x = Bonobo.get_solution(tree)
            primal_value = tree.root.problem.f(x)

            # TODO: here we need to calculate the actual state

            # If the tree is empty, incumbent and solution should be the same!
            if isempty(tree.nodes)
                @assert isapprox(tree.incumbent, primal_value)
            end

            result[:number_nodes] = tree.num_nodes
            result[:lmo_calls] = tree.root.problem.lmo.ncalls
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

            if verbose
                print_callback = FrankWolfe.print_callback
                headers = [
                    " ",
                    "Iteration",
                    "Open",
                    "Bound",
                    "Incumbent",
                    "Gap (abs)",
                    "Gap (%)",
                    "Time (s)",
                    "Nodes/sec",
                    "FW (ms)",
                    "LMO (ms)",
                    "LMO (calls c)",
                    "FW (Its)",
                    "#ActiveSet",
                    "Discarded",
                ]
                format_string = "%1s %10i %10i %14e %14e %14e %14e %14e %14e %14i %14i %14i %10i %12i %10i\n"
                print_callback(headers, format_string, print_footer=true)
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
function postsolve(tree, result, time_ref, verbose, use_postsolve, max_iteration_post)
    x = Bonobo.get_solution(tree)
    primal = tree.incumbent_solution.objective

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

    if use_postsolve
        # Build solution lmo
        fix_bounds = IntegerBounds()
        for i in tree.root.problem.integer_variables
            push!(fix_bounds, (i => MOI.LessThan(round(x[i]))))
            push!(fix_bounds, (i => MOI.GreaterThan(round(x[i]))))
        end

        MOI.set(tree.root.problem.lmo.lmo.o, MOI.Silent(), true)
        free_model(tree.root.problem.lmo.lmo.o)
        build_LMO(
            tree.root.problem.lmo,
            tree.root.problem.integer_variable_bounds,
            fix_bounds,
            tree.root.problem.integer_variables,
        )
        # Postprocessing
        direction = ones(length(x))
        v = compute_extreme_point(tree.root.problem.lmo, direction)
        active_set = FrankWolfe.ActiveSet([(1.0, v)])
        verbose && println("Postprocessing")
        x, _, primal, dual_gap, _, _ = FrankWolfe.blended_pairwise_conditional_gradient(
            tree.root.problem.f,
            tree.root.problem.g,
            tree.root.problem.lmo,
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
            tree.lb = tree.root.problem.solving_stage == OPT_TREE_EMPTY ? primal - dual_gap : tree.lb
            tree.incumbent_solution.objective = tree.solutions[1].objective = primal
            tree.incumbent_solution.solution = tree.solutions[1].solution = x
        else 
            if primal < tree.incumbent && tree.lb > primal - dual_gap
                @warn "tree.lb > primal - dual_gap"
            else 
                @warn "primal >= tree.incumbent"
            end
            @warn "postsolve did not improve the solution"
            primal = tree.incumbent_solution.objective = tree.solutions[1].objective
            x = tree.incumbent_solution.solution = tree.solutions[1].solution
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
        println("\t Total number of lmo calls: ", tree.root.problem.lmo.ncalls)
        println("\t Total time (s): ", total_time_in_sec)
        println("\t LMO calls / sec: ", tree.root.problem.lmo.ncalls / total_time_in_sec)
        println("\t Nodes / sec: ", tree.num_nodes / total_time_in_sec)
        println("\t LMO calls / node: $(tree.root.problem.lmo.ncalls / tree.num_nodes)\n")
    end

    # Reset LMO 
    int_bounds = IntegerBounds()
    build_LMO(
        tree.root.problem.lmo,
        tree.root.problem.integer_variable_bounds,
        int_bounds,
        tree.root.problem.integer_variables,
    )

    return x
end

# cleanup internal SCIP model
function free_model(o::SCIP.Optimizer)
    SCIP.SCIPfreeTransform(o)
end

# no-op by default
function free_model(o::MOI.AbstractOptimizer)   
end
