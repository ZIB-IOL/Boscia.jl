using Statistics
using LinearAlgebra

# Constant parameters for the sparse regression
# min norm(y-A β)² + λ_0 ∑ z_i + λ_2 ||β||²
# s.t. -Mz_i<=β_i <= Mz_i
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p 
#Random.seed!(4)
n0=10; p = 5*n0; k = ceil(n0/5);
const lambda_0 = rand(Float64); const lambda_2 = 10.0*rand(Float64);
const A = rand(Float64, n0, p)
const y = rand(Float64, n0)
const M = 2*var(A)

@testset "Sparse Regression" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o,2p)
    for i in p+1:2p
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, x[i], MOI.ZeroOne()) # or MOI.Integer()
    end 
    for i in 1:p
        MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,M], [x[i], x[i+p]]), 0.0), MOI.GreaterThan(0.0))
        MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-M], [x[i], x[i+p]]), 0.0), MOI.LessThan(0.0))
        # Indicator: x[i+p] = 1 => -M <= x[i] <= M
        gl = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[i+p])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[i])),],
            [0.0, 0.0], )
        gg = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[i+p])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, x[i])),],
            [0.0, 0.0], )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(M)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-M)))
    end
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p),x[p+1:2p]), 0.0), MOI.LessThan(k))
    lmo = FrankWolfe.MathOptLMO(o)
    global_bounds = Boscia.IntegerBounds()
    for i = 1:p
        push!(global_bounds, (i+p, MOI.GreaterThan(0.0)))
        push!(global_bounds, (i+p, MOI.LessThan(1.0)))
        push!(global_bounds, (i, MOI.GreaterThan(-M)))
        push!(global_bounds, (i, MOI.LessThan(M)))
    end
    time_lmo= Boscia.TimeTrackingLMO(lmo)

    # Define the root of the tree
    # we fix the direction so we can actually find a veriable to split on later!
    direction = Vector{Float64}(undef,2p)
    Random.rand!(direction)
    v = compute_extreme_point(time_lmo, direction)
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    function f(x)
        return sum((y-A*x[1:p]).^2) + lambda_0*sum(x[p+1:2p]) + lambda_2*norm(x[1:p])^2
    end
    function grad!(storage, x)
        storage.=vcat(2*(transpose(A)*A*x[1:p] - transpose(A)*y + lambda_2*x[1:p]), lambda_0*ones(p))
        return storage
    end
    active_set = FrankWolfe.ActiveSet([(1.0, v)]) 
    m = Boscia.SimpleOptimizationProblem(f, grad!, 2p, collect(p+1:2p), time_lmo, global_bounds) 

    # TO DO: how to do this elegantly
    nodeEx = Boscia.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, Boscia.IntegerBounds(), 1, 1e-3, Millisecond(0))

    # create tree
    tree = Bonobo.initialize(; 
    traverse_strategy = Bonobo.BFS(),
    Node = typeof(nodeEx),
    root = (problem=m, current_node_id = Ref{Int}(0), updated_incumbent = Ref{Bool}(false), options= Dict{Symbol, Any}(:verbose => false,:dual_gap_decay_factor => 0.7, :dual_gap => 1e-6, :max_fw_iter => 10000, :min_node_fw_epsilon => 1e-6)),
    )
    Bonobo.set_root!(tree, 
    (active_set = active_set, 
    discarded_vertices= vertex_storage,
    local_bounds = Boscia.IntegerBounds(),
    level = 1, 
    fw_dual_gap_limit= 1e-3,
    fw_time = Millisecond(0))
    )

    function build_FW_callback(tree)
        return function fw_callback(state, active_set)
        end
    end

    fw_callback = build_FW_callback(tree)
    tree.root.options[:callback] = fw_callback
    # Profile.init()
    # ProfileView.@profview Bonobo.optimize!(tree)
    Bonobo.optimize!(tree)
    x = Bonobo.get_solution(tree)
    # println("Solution: $(x[1:p])")
    @test sum(x[1+p:2p]) <= k
end 


n0=10; p = 5*n0; k = ceil(n0/5); group_size = convert(Int64, floor(p/k))
const lambda_0_g = rand(Float64); const lambda_2_g = 10.0*rand(Float64);
const A_g = rand(Float64, n0, p)
const y_g = rand(Float64, n0)
const M_g = 2*var(A_g)
groups= []
k_int = convert(Int64, k)
for i in 1:(k_int-1)
    push!(groups, ((i-1)*group_size+p+1):(i*group_size+p))
end
push!(groups,((k_int-1)*group_size+p+1):2p)

@testset "Sparse Regression Group" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o,2p)
    for i in p+1:2p
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, x[i], MOI.ZeroOne()) # or MOI.Integer()
    end 
    for i in 1:p
        MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,M_g], [x[i], x[i+p]]), 0.0), MOI.GreaterThan(0.0))
        MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-M_g], [x[i], x[i+p]]), 0.0), MOI.LessThan(0.0))
        # Indicator: x[i+p] = 1 => -M_g <= x[i] <= M_g
        gl = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[i+p])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[i])),],
            [0.0, 0.0], )
        gg = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[i+p])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, x[i])),],
            [0.0, 0.0], )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(M_g)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-M_g)))
    end
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p),x[p+1:2p]), 0.0), MOI.LessThan(k))
    for i in 1:k_int
        MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(group_size),x[groups[i]]), 0.0), MOI.GreaterThan(1.0))
    end
    lmo = FrankWolfe.MathOptLMO(o)
    global_bounds = Boscia.IntegerBounds()
    for i = 1:p
        push!(global_bounds, (i+p, MOI.GreaterThan(0.0)))
        push!(global_bounds, (i+p, MOI.LessThan(1.0)))
        push!(global_bounds, (i, MOI.GreaterThan(-M_g)))
        push!(global_bounds, (i, MOI.LessThan(M_g)))
    end
    time_lmo = Boscia.TimeTrackingLMO(lmo)

    # Define the root of the tree
    direction = Vector{Float64}(undef,2p)
    Random.rand!(direction)
    v = compute_extreme_point(time_lmo, direction)
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    function f(x)
        return sum((y_g-A_g*x[1:p]).^2) + lambda_0_g*sum(x[p+1:2p]) + lambda_2_g*norm(x[1:p])^2
    end
    function grad!(storage, x)
        storage.=vcat(2*(transpose(A_g)*A_g*x[1:p] - transpose(A_g)*y_g + lambda_2_g*x[1:p]), lambda_0_g*ones(p))
        return storage
    end
    active_set = FrankWolfe.ActiveSet([(1.0, v)]) 
    m = Boscia.SimpleOptimizationProblem(f, grad!, 2p, collect(p+1:2p), time_lmo, global_bounds) 

    # TO DO: how to do this elegantly
    nodeEx = Boscia.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, Boscia.IntegerBounds() , 1, 1e-3, Millisecond(0))

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
        end
    end

    fw_callback = build_FW_callback(tree)
    tree.root.options[:callback] = fw_callback
    # Profile.init()
    # ProfileView.@profview Bonobo.optimize!(tree)
    Bonobo.optimize!(tree)
    x = Bonobo.get_solution(tree)
    # println("Solution: $(x[1:p])")
    @test sum(x[p+1:2p]) <= k
    for i in 1:k_int
        @test sum(x[groups[i]]) >= 1
    end
    println("Non zero entries:")
    for i in 1:p
        if x[i+p] == 1
            println("$(i)th entry: $(x[i])")
        end
    end
end 
