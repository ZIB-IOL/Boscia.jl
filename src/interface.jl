
import MathOptInterface
const MOI = MathOptInterface

function branch_wolfe(f, grad!, lmo; traverse_strategy = Bonobo.BFS(), kwargs...)

    v_indices = MOI.get(lmo.o, MOI.ListOfVariableIndices())
    if v_indices != MOI.VariableIndex.(1:length(v_indices))
        error("Variables are expected to be contiguous and ordered from 1 to N")
    end
    time_lmo = BranchWolfe.TimeTrackingLMO(lmo)
    
    integer_variables = BitSet()
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
            s = MOI.get(lmo.o, MOI.ConstraintSet(), cidx)
            push!(global_bounds, (idx, s))
        end
    end

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
