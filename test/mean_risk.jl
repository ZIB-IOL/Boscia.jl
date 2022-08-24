using Statistics
using Distributions
using LinearAlgebra

# min h(sqrt(y' * M * y)) - r' * y
# s.t. a' * y <= b 
#           y >= 0
#           y_i in Z for i in I

n0 = 10
const r = 10 * rand(n0)
const a = rand(n0)
const Ω = 3 * rand(Float64)
const b = sum(a)
@show b
A1 = randn(n0,n0)
A1 = A1' * A1
const M1 =  (A1 + A1')/2
@assert isposdef(M1)


@testset "Buchheim et. al. mean risk" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o,n0)
    I = collect(1:n0) #rand(1:n0, Int64(floor(n0/2)))
    for i in 1:n0
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        if i in I
            MOI.add_constraint(o, x[i], MOI.Integer())
        end
    end 
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(a,x), 0.0), MOI.LessThan(b))
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(a,x), 0.0), MOI.GreaterThan(minimum(a)))
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n0),x), 0.0), MOI.GreaterThan(1.0))
    lmo = FrankWolfe.MathOptLMO(o)
    global_bounds = Boscia.IntegerBounds()
    for i = 1:n0
        push!(global_bounds, (i, MOI.GreaterThan(0.0)))
    end
    time_lmo= Boscia.TimeTrackingLMO(lmo)

    # Define the root of the tree
    # we fix the direction so we can actually find a veriable to split on later!
    direction = Vector{Float64}(undef,n0)
    Random.rand!(direction)
    v = compute_extreme_point(time_lmo, direction)
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    function f(x)
        return Ω * (x' * M1 * x) - r' * x
    end
    function grad!(storage, x)
        storage.= 2 * Ω * M1 * x - r
        return storage
    end
    active_set = FrankWolfe.ActiveSet([(1.0, v)]) 
    m = Boscia.SimpleOptimizationProblem(f, grad!, n0, I, time_lmo, global_bounds) 

    # TO DO: how to do this elegantly
    nodeEx = Boscia.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, Boscia.IntegerBounds(), 1, 1e-3, Millisecond(0))

    # create tree
    tree = Bonobo.initialize(;
    traverse_strategy = Bonobo.BFS(),
    Node = typeof(nodeEx),
    root = (problem=m, current_node_id = Ref{Int}(0), updated_incumbent = Ref{Bool}(false), options= Dict{Symbol, Any}(:verbose => false, :dual_gap_decay_factor => 0.7, :dual_gap => 1e-6, :max_fw_iter => 10000, :min_node_fw_epsilon => 1e-6)),
    )
    Bonobo.set_root!(tree, 
    (active_set = active_set, 
    discarded_vertices = vertex_storage,
    local_bounds = Boscia.IntegerBounds(),
    level = 1,
    fw_dual_gap_limit = 1e-3,
    fw_time = Millisecond(0))
    )

    function build_FW_callback(tree)
        return function fw_callback(state, active_set)
            # print("Primal: $(state.primal)\n")
            # print("Length of active set: $(length(active_set.weights))\n")
        end
    end

    fw_callback = build_FW_callback(tree)
    tree.root.options[:callback] = fw_callback
    # Profile.init()
    # ProfileView.@profview Bonobo.optimize!(tree)
    Bonobo.optimize!(tree)
    x = Bonobo.get_solution(tree)
    # @show x
    @test sum(a'* x) <= b + eps()
end
