
function branch_wolfe(f, grad!, lmo; traverse_strategy = Bonobo.BFS(), branching_strategy = Bonobo.FIRST(), fw_epsilon = 1e-5, verbose = false, dual_gap = 1e-7, kwargs...)

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

    direction = Vector{Float64}(undef,n)
    Random.rand!(direction)
    v = compute_extreme_point(lmo, direction)
    active_set = FrankWolfe.ActiveSet([(1.0, v)])
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    m = BranchWolfe.SimpleOptimizationProblem(f, grad!, n, integer_variables, time_lmo, global_bounds)
    nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, 1e-3, Millisecond(0))

    # create tree
    function perform_strong_branch(tree, node)
        return node.level <= length(tree.root.problem.integer_variables)/3
    end
    branching_strategy = BranchWolfe.HybridStrongBranching(10, 1e-3, HiGHS.Optimizer(), perform_strong_branch)
    MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)
    tree = Bonobo.initialize(;
        traverse_strategy = traverse_strategy,
        Node = typeof(nodeEx),
        root = (problem=m, current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => 1e-5, :percentage_dual_gap => dual_gap)),
        branch_strategy = branching_strategy, #() ->
    )
    Bonobo.set_root!(tree, 
    (active_set = active_set, 
    discarded_vertices = vertex_storage,
    local_bounds = BranchWolfe.IntegerBounds(),
    level = 1, 
    fw_dual_gap_limit= fw_epsilon,
    FW_time = Millisecond(0)))

    function build_bnb_callback(tree, list_lb_cb, list_ub_cb, list_time_cb, list_num_nodes_cb, list_lmo_calls_cb)
        time_ref = Dates.now()
        iteration = 0
        println("Starting BranchWolfe")
        verbose = verbose
        if verbose
            println("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            @printf("| iter \t| node id | lower bound | incumbent | gap \t| rel. gap | total time   | time/nodes \t| FW time    | LMO time   | total LMO calls | FW iterations | active set size | discarded set size |\n")
            println("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        end
        return function bnb_callback(tree, node; worse_than_incumbent=false, node_infeasible=false) # FW_iterations
            # update lower bound
            iteration = iteration +1
            append!(list_ub_cb, copy(tree.incumbent)) # new cb structure
            append!(list_lb_cb, copy(tree.lb))
            append!(list_num_nodes_cb, copy(tree.num_nodes))
            dual_gap = tree.incumbent-tree.lb
            time = float(Dates.value(Dates.now()-time_ref))
            append!(list_time_cb, time)
            FW_time = Dates.value(node.FW_time)
            FW_iter = 0 #FW_iterations[end]
            if !isempty(tree.root.problem.lmo.optimizing_times)
                LMO_time = sum(1000*tree.root.problem.lmo.optimizing_times)
                empty!(tree.root.problem.lmo.optimizing_times)
            end 
            LMO_calls = tree.root.problem.lmo.ncalls
            append!(list_lmo_calls_cb, copy(LMO_calls))
    
            if !isempty(tree.nodes)
                lower_bounds = [n[2].lb for n in tree.nodes]
                if tree.lb>minimum(lower_bounds)
                end
                tree.lb = minimum(lower_bounds)
            end
    
            if !isempty(FW_iterations)
                FW_iter = FW_iterations[end]
            else 
                FW_iter = 0
            end
    
            active_set_size = length(node.active_set)
            discarded_set_size = length(node.discarded_vertices.storage)
    
            if verbose & !worse_than_incumbent & !node_infeasible
                @printf("|   %4i|     %4i| \t% 06.5f|    %.5f|    %.5f|     %.3f|     %6i ms|      %4i ms|   %6i ms|   %6i ms|            %5i|          %5i|            %5i|               %5i|\n", iteration, node.id, tree.lb, tree.incumbent, dual_gap, dual_gap/tree.incumbent, time, round(time/tree.num_nodes), FW_time, LMO_time, tree.root.problem.lmo.ncalls, FW_iter, active_set_size, discarded_set_size)
            end
            #FW_iter = []
    
            # update current_node_id
            if !Bonobo.terminated(tree)
                tree.root.current_node_id[] = Bonobo.get_next_node(tree, tree.options.traverse_strategy).id
            end
        end
    end
    
    # build callbacks
    list_ub_cb = []
    list_lb_cb = []
    list_time_cb = [] 
    list_num_nodes_cb = [] 
    list_lmo_calls_cb = []
    bnb_callback = build_bnb_callback(tree, list_lb_cb, list_ub_cb, list_time_cb, list_num_nodes_cb, list_lmo_calls_cb)

    FW_iterations = []
    min_number_lower = Inf
    fw_callback = BranchWolfe.build_FW_callback(tree, min_number_lower, true, FW_iterations)

    tree.root.options[:callback] = fw_callback
    tree.root.current_node_id[] = Bonobo.get_next_node(tree, tree.options.traverse_strategy).id

    time_ref = Dates.now()
    Bonobo.optimize!(tree; callback=bnb_callback)

    
    x = Bonobo.get_solution(tree)
    return x, time_lmo
end
