
function branch_wolfe(f, grad!, lmo; traverse_strategy = Bonobo.BFS(), branching_strategy = Bonobo.MOST_INFEASIBLE(), fw_epsilon = 1e-5, verbose = false, dual_gap = 1e-7, kwargs...)
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

    direction = Vector{Float64}(undef,n)
    Random.rand!(direction)
    v = compute_extreme_point(lmo, direction)
    active_set = FrankWolfe.ActiveSet([(1.0, v)])
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    m = BranchWolfe.SimpleOptimizationProblem(f, grad!, n, integer_variables, time_lmo, global_bounds)
    nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, 1e-3)
    
    tree = Bonobo.initialize(; 
        traverse_strategy = traverse_strategy,
        Node = typeof(nodeEx),
        root = (problem=m, current_node_id = current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => fw_epsilon, :verbose => verbose, :dual_gap => dual_gap)),
        branch_strategy = branching_strategy,
    )
    # set root
    Bonobo.set_root!(tree, 
        (active_set = active_set, 
        discarded_vertices = vertex_storage,
        local_bounds = BranchWolfe.IntegerBounds(),
        level = 1,
        fw_dual_gap_limit= 1e-3)
    )

    Bonobo.optimize!(tree)
    
    x = Bonobo.get_solution(tree)

    # TO DO: If problem is not purely binary or integer
    # fix all integral variables and compute again to find the optimal continuous values.
    if tree.root.problem.nvars > length(tree.root.problem.integer_variables)
        # read bounds from x
        fix_bounds = IntegerBounds()
        for i in tree.root.problem.integer_variables
            push!(fix_bounds, (i => MOI.LessThan(round(x[i]))))
            push!(fix_bounds, (i => MOI.GreaterThan(round(x[i]))))
        end
        # build lmo 
        build_LMO(tree.root.problem.lmo, tree.root.problem.integer_variable_bounds, fix_bounds, tree.root.problem.integer_variables)
        v = compute_extreme_point(lmo, direction)
        active_set = FrankWolfe.ActiveSet([(1.0, v)])
        # evaluate 
        x,_,_,_,_ ,_ = FrankWolfe.blended_pairwise_conditional_gradient(
            tree.root.problem.f,
            tree.root.problem.g,
            tree.root.problem.lmo,
            active_set,
            add_dropped_vertices=true,
            use_extra_vertex_storage=true,
            extra_vertex_storage=FrankWolfe.DeletedVertexStorage(typeof(v)[], 1),
            #callback=fw_callback,
            lazy=true,
            verbose=false,
        ) 
    end
    return x, time_lmo
end
