
function branch_wolfe(f, grad!, lmo; traverse_strategy = Bonobo.BFS(), branching_strategy = Bonobo.MOST_INFEASIBLE(), fw_epsilon = 1e-5, verbose = false, dual_gap = 1e-7, print_iter = 100, dual_gap_decay_factor=0.8, kwargs...)
    if verbose
        println("\nBranchWolfe Algorithm\n")
        println("Parameter settings.")
        println("\t Tree traversal strategy: ", traverse_strategy)
        println("\t Branching strategy: ", branching_strategy)
        println("\t Absolute dual gap tolerance: ", dual_gap)
        println("\t Frank-Wolfe subproblem tolerance: $(fw_epsilon)\n")
    end

    v_indices = MOI.get(lmo.o, MOI.ListOfVariableIndices())
    n = length(v_indices)
    if v_indices != MOI.VariableIndex.(1:n)
        error("Variables are expected to be contiguous and ordered from 1 to N")
    end
    time_lmo = BranchWolfe.TimeTrackingLMO(lmo)

    integer_variables = Vector{Int}()
    for cidx in MOI.get(lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.Integer}())
        push!(integer_variables, cidx.value)
    end
    for cidx in MOI.get(lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.ZeroOne}())
        push!(integer_variables, cidx.value)
    end

    global_bounds = BranchWolfe.IntegerBounds()
    for idx in integer_variables
        for ST in (MOI.LessThan{Float64}, MOI.GreaterThan{Float64})
            cidx = MOI.ConstraintIndex{MOI.VariableIndex, ST}(idx)
            # Variable constraints to not have to be explicitly given, see Buchheim example
            if MOI.is_valid(lmo.o, cidx)
                s = MOI.get(lmo.o, MOI.ConstraintSet(), cidx)
                push!(global_bounds, (idx, s))
            end
        end
    end

    direction = ones(n)
    v = compute_extreme_point(lmo, direction)
    active_set = FrankWolfe.ActiveSet([(1.0, v)])
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    m = BranchWolfe.SimpleOptimizationProblem(f, grad!, n, integer_variables, time_lmo, global_bounds)
    nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0, 0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, 1e-3, Millisecond(0))

    tree = Bonobo.initialize(; 
        traverse_strategy = traverse_strategy,
        Node = typeof(nodeEx),
        root = (problem=m, current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:dual_gap_decay_factor => dual_gap_decay_factor, :dual_gap => dual_gap, :print_iter => print_iter)),
        branch_strategy = branching_strategy,
    )
    Bonobo.set_root!(tree, 
    (active_set = active_set, 
    discarded_vertices = vertex_storage,
    local_bounds = BranchWolfe.IntegerBounds(),
    level = 1, 
    fw_dual_gap_limit= fw_epsilon,
    fw_time = Millisecond(0)))

    # build callbacks
    list_ub_cb = Float64[]
    list_lb_cb = Float64[]
    list_time_cb = Float64[] 
    list_num_nodes_cb = Int[] 
    list_lmo_calls_cb = Int[]
    list_active_set_size_cb = Int[]
    list_discarded_set_size_cb = Int[]
    fw_iterations = Int[]
    result = Dict{Symbol, Any}()
    lmo_calls_per_layer = Vector{Vector{Int}}()
    bnb_callback = build_bnb_callback(tree, list_lb_cb, list_ub_cb, list_time_cb, list_num_nodes_cb, list_lmo_calls_cb, verbose, fw_iterations, list_active_set_size_cb, list_discarded_set_size_cb, result, lmo_calls_per_layer)

    min_number_lower = Inf
    fw_callback = build_FW_callback(tree, min_number_lower, true, fw_iterations)

    tree.root.options[:callback] = fw_callback
    tree.root.current_node_id[] = Bonobo.get_next_node(tree, tree.options.traverse_strategy).id

    Bonobo.optimize!(tree; callback=bnb_callback)

    x = Bonobo.get_solution(tree)

    
    # Build solution lmo
    fix_bounds = IntegerBounds()
    for i in tree.root.problem.integer_variables
        push!(fix_bounds, (i => MOI.LessThan(round(x[i]))))
        push!(fix_bounds, (i => MOI.GreaterThan(round(x[i]))))
    end
    build_LMO(tree.root.problem.lmo, tree.root.problem.integer_variable_bounds, fix_bounds, tree.root.problem.integer_variables)
    
    # Final solve in case of mixed problem
    if tree.root.problem.nvars > length(tree.root.problem.integer_variables)
        v = compute_extreme_point(lmo, direction)
        active_set = FrankWolfe.ActiveSet([(1.0, v)])
        # evaluate 
        x,_,dual_gap,_,_ ,_ = FrankWolfe.blended_pairwise_conditional_gradient(
            tree.root.problem.f,
            tree.root.problem.g,
            tree.root.problem.lmo,
            active_set,
            add_dropped_vertices=true,
            use_extra_vertex_storage=true,
            extra_vertex_storage=FrankWolfe.DeletedVertexStorage(typeof(v)[], 1),
            #callback=fw_callback,
            lazy=true,
            verbose=true,
        ) 
    end

    # Check solution and polish
    x_raw = x
    x_polished = x
    if !is_linear_feasible(tree.root.problem.lmo, x)
        error("Reported solution not linear feasbile!")
    end
    if !is_integer_feasible(tree, x)
        for i in tree.root.problem.integer_variables
            x_polished[i] = round(x_polished[i])
        end
        if !is_linear_feasible(tree.root.problem.lmo, x_polished)
            @warn "Polished solution not linear feasible"
        else
            x = x_polished

        end
    end
    return x, time_lmo, result, dual_gap
end

"""
Output of BranchWolfe

    iter :          current iteration of BranchWolfe
    node id :       current node id
    lower bound :   tree_lb(tree)
    incumbent :     tree.incumbent
    gap :           tree.incumbent-tree_lb(tree)
    rel. gap :      dual_gap/tree.incumbent
    time :          total time of BranchWolfe
    time/nodes :    average time per node
    FW time :       time spent in FW 
    LMO time :      time used by LMO
    LMO calls :     number of compute_extreme_point calls in FW
    FW iterations : number of iterations in FW
"""
function build_bnb_callback(tree, list_lb_cb, list_ub_cb, list_time_cb, list_num_nodes_cb, list_lmo_calls_cb, verbose, fw_iterations, list_active_set_size_cb, list_discarded_set_size_cb, result, lmo_calls_per_layer)
    time_ref = Dates.now()
    iteration = 0

    headers = ["Iteration", "Open", "Bound", "Incumbent", "Gap (abs)", "Gap (%)", "Time (s)", "Nodes/Sec", "FW (ms)", "LMO (ms)", "LMO (calls c)", "FW (iters)", "Active Set", "Discarded"]
    format_string = "%10i %10i %14e %14e %14e %14e %14e %14e %14i %14i %14i %10i %10i %10i\n"
    print_callback = FrankWolfe.print_callback
    print_iter = get(tree.root.options, :print_iter, 100)

    if verbose
        print_callback(headers, format_string, print_header=true)
    end
    return function callback(tree, node; worse_than_incumbent=false, node_infeasible=false)
        if !node_infeasible
            # update lower bound
            push!(list_ub_cb, tree.incumbent) 
            push!(list_lb_cb, tree_lb(tree))
            push!(list_num_nodes_cb, tree.num_nodes)
            iteration += 1
            dual_gap = tree.incumbent-tree_lb(tree)
            time = float(Dates.value(Dates.now()-time_ref))
            push!(list_time_cb, time)
            fw_time = Dates.value(node.fw_time)
            fw_iter = if !isempty(fw_iterations)
                fw_iterations[end]
            else
                0
            end
            if !isempty(tree.root.problem.lmo.optimizing_times)
                LMO_time = sum(1000*tree.root.problem.lmo.optimizing_times)
                empty!(tree.root.problem.lmo.optimizing_times)
            else 
                LMO_time = 0
            end 
            LMO_calls_c = tree.root.problem.lmo.ncalls
            push!(list_lmo_calls_cb, copy(LMO_calls_c))

            if !isempty(tree.nodes)
                lower_bounds = [n[2].lb for n in tree.nodes]
                tree.lb = min(minimum(lower_bounds), tree.incumbent)
            end

            active_set_size = length(node.active_set)
            discarded_set_size = length(node.discarded_vertices.storage)
            push!(list_active_set_size_cb, active_set_size)
            push!(list_discarded_set_size_cb, discarded_set_size)
            nodes_left= length(tree.nodes)
            if verbose && (mod(iteration, print_iter) == 0 || iteration == 1 || Bonobo.terminated(tree)) # TODO: need to output the very last iteration also if we skip some inbetween
                if (mod(iteration, print_iter*40) == 0)
                    print_callback(headers, format_string, print_header=true)
                end
                print_callback((iteration, nodes_left, tree_lb(tree), tree.incumbent, dual_gap, relative_gap(tree.incumbent,tree_lb(tree)) * 100.0, time / 1000.0, tree.num_nodes/time * 1000.0, fw_time, LMO_time, tree.root.problem.lmo.ncalls, fw_iter, active_set_size, discarded_set_size), format_string, print_header=false)
            end
            # lmo calls per layer
            if length(list_lmo_calls_cb) > 1
                LMO_calls = list_lmo_calls_cb[end] - list_lmo_calls_cb[end-1]
            else 
                LMO_calls = list_lmo_calls_cb[end]
            end
            if length(lmo_calls_per_layer) < node.level
                push!(lmo_calls_per_layer, [LMO_calls])
            else 
                push!(lmo_calls_per_layer[node.level], LMO_calls)
            end
        end
        # update current_node_id
        if !Bonobo.terminated(tree)
            tree.root.current_node_id[] = Bonobo.get_next_node(tree, tree.options.traverse_strategy).id
        end
    
        if Bonobo.terminated(tree)
            x = Bonobo.get_solution(tree)
            primal_value = tree.root.problem.f(x)
    
            # TODO: here we need to calculate the actual state
    
            status_string = "FIX ME" # should report "feasible", "optimal", "infeasible", "gap tolerance met"
            if isempty(tree.nodes)
                status_string = "Optimal (tree empty)"
            else
                status_string = "Optimal (tolerance reached)"
            end
    
            result[:primal_objective] = primal_value 
            result[:dual_bound] = tree_lb(tree)
            result[:dual_gap] = relative_gap(primal_value,tree_lb(tree)) * 100.0
            result[:number_nodes] = tree.num_nodes
            result[:lmo_calls] = tree.root.problem.lmo.ncalls
            total_time_in_sec = (Dates.value(Dates.now()-time_ref))/1000.0
            result[:total_time_in_sec] = total_time_in_sec
            result[:list_num_nodes] = list_num_nodes_cb
            result[:list_lmo_calls_acc] = list_lmo_calls_cb
            result[:list_active_set_size] = list_active_set_size_cb
            result[:list_discarded_set_size] = list_discarded_set_size_cb
            result[:list_lb] = list_lb_cb
            result[:list_ub] = list_ub_cb 
            result[:list_time] = list_time_cb
            result[:lmo_calls_per_layer] = lmo_calls_per_layer

            if verbose
                print_callback = FrankWolfe.print_callback
                headers = ["Iteration", "Open", "Bound", "Incumbent", "Gap (abs)", "Gap (%)", "Time (s)", "Nodes/Sec", "FW (ms)", "LMO (ms)", "LMO (calls c)", "FW (iters)", "Active Set", "Discarded"]   
                format_string = "%10i %10i %14e %14e %14e %14e %14e %14e %14i %14i %14i %10i %10i %10i\n"
                print_callback(headers, format_string, print_footer=true)
                println()

                println("Solution Statistics.")
                
                println("\t Solution Status: ", status_string)
                println("\t Primal Objective: ", primal_value)
                println("\t Dual Bound: ", tree_lb(tree))
                println("\t Dual Gap (relative in %): $(relative_gap(primal_value,tree_lb(tree)) * 100.0)\n")
                println("Search Statistics.")
                println("\t Total number of nodes processed: ", tree.num_nodes)
                println("\t Total number of lmo calls: ", tree.root.problem.lmo.ncalls)
                println("\t Total time (s): ", total_time_in_sec)
                println("\t LMO calls / sec: ", tree.root.problem.lmo.ncalls / total_time_in_sec)        
                println("\t Nodes / sec: ", tree.num_nodes / total_time_in_sec)
                println("\t LMO calls / node: $(tree.root.problem.lmo.ncalls / tree.num_nodes)\n")
            end
        end
    end
end
