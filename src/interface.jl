
function branch_wolfe(f, grad!, lmo; traverse_strategy = Bonobo.BFS(), kwargs...)
    time_lmo = BranchWolfe.TimeTrackingLMO(lmo)
    
    # TODO extract global bounds from constraints of the base lmo.o
    global_bounds = BranchWolfe.IntegerBounds()

    # TODO extract integer constraints from the lmo.o
    integer_variables = BitSet()

    direction = Vector{Float64}(undef,a)
    Random.rand!(direction)
    v = compute_extreme_point(lmo, direction)
    active_set = FrankWolfe.ActiveSet([(1.0, v)])
    m = BranchWolfe.SimpleOptimizationProblemInfeasible(f, grad!, a, integer_variables, time_lmo, global_bounds, active_set)
    nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, 1e-3)
    tree = Bonobo.initialize(; 
        traverse_strategy = traverse_strategy,
        Node = typeof(nodeEx),
        root = (problem=m, current_node_id = current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => 1e-5, :verbose => false)),
    )

    # set root

    Bonobo.optimize!(tree; options...)
    
    x = Bonobo.get_solution(tree)
    return x, time_lmo
end
