using Statistics
using Random
using Distributions
using LinearAlgebra
import HiGHS


# Sparse Poisson regression
# min_{w, b, z} ∑_i exp(w x_i + b) - y_i (w x_i + b) + α norm(w)^2
# s.t. -N z_i <= w_i <= N z_i
# b ∈ [-N, N]
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p

#Random.seed!(4)

n0=30
p = n0

# underlying true weights
const w0 = 2 * rand(Float64, p) .- 1 
# set 50 entries to 0
for _ in 1:20
    w0[rand(1:p)] = 0
end
const b0 = 2 * rand(Float64) - 1
const X0 = 2 * rand(Float64, n0, p) .- 1 
const y0 = map(1:n0) do idx
    a = dot(X0[idx,:], w0) + b0
    rand(Distributions.Poisson(exp(a)))
end
N = 5.0

@testset "Poisson sparse regression" begin
    k = 5
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    w = MOI.add_variables(o, p)
    z = MOI.add_variables(o, p)
    b = MOI.add_variable(o)
    for i in 1:p
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())
    end
    for i in 1:p
        MOI.add_constraint(o, -N * z[i]- w[i], MOI.LessThan(0.0))
        MOI.add_constraint(o, N * z[i]- w[i], MOI.GreaterThan(0.0))
        # Indicator: z[i] = 1 => -N <= w[i] <= N
        gl = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),],
            [0.0, 0.0], )
        gg = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),],
            [0.0, 0.0], )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(N)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-N))) 
    end
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(o, b, MOI.LessThan(N))
    MOI.add_constraint(o, b, MOI.GreaterThan(-N))
    lmo = FrankWolfe.MathOptLMO(o)
    global_bounds = Boscia.IntegerBounds()
    for i in 1:p
        push!(global_bounds, (p+i, MOI.GreaterThan(0.0)))
        push!(global_bounds, (p+i, MOI.LessThan(1.0)))
        push!(global_bounds, (i, MOI.GreaterThan(-N)))
        push!(global_bounds, (i, MOI.LessThan(N)))
    end
    push!(global_bounds, (2p+1, MOI.GreaterThan(-N)))
    push!(global_bounds, (2p+1, MOI.LessThan(N)))

    direction = Vector{Float64}(undef,2p + 1)
    Random.rand!(direction)
    v = compute_extreme_point(lmo, direction)
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)
    α = 1.3
    function f(θ)
        w = @view(θ[1:p])
        b = θ[end]
        s = sum(1:n0) do i
            a = dot(w, X0[:,i]) + b
            1/n0 * (exp(a) - y0[i] * a)
        end
        s + α * norm(w)^2
    end
    function grad!(storage, θ)
        w = @view(θ[1:p])
        b = θ[end]
        storage[1:p] .= 2α .* w
        storage[p+1:2p] .= 0
        storage[end] = 0
        for i in 1:n0
            xi = @view(X0[:,i])
            a = dot(w, xi) + b
            storage[1:p] .+= 1/n0 * xi * exp(a)
            storage[1:p] .-= 1/n0 * y0[i] * xi
            storage[end] += 1/n0 * (exp(a) - y0[i])
        end
        storage ./= norm(storage)
        return storage
    end
    time_lmo = Boscia.TimeTrackingLMO(lmo)
    active_set = FrankWolfe.ActiveSet([(1.0, v)]) 
    m = Boscia.SimpleOptimizationProblem(f, grad!, 2p+1, collect(p+1:2p), time_lmo, global_bounds) 

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
    fw_dual_gap_limit= 1e-3,
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
    # println("Solution: $(x[1:p])")
    @test sum(x[p+1:2p]) <= k
    #println("Non zero entries:")
    #for i in 1:p
    #    if isapprox(x[i+p], 1.0, atol = 1e-5, rtol= 1e-5)
    #        println("$(i)th entry: $(x[i])")
    #    end
    #end
end 

@testset "Hybrid branching poisson sparse regression" begin
    k = 5
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    w = MOI.add_variables(o, p)
    z = MOI.add_variables(o, p)
    b = MOI.add_variable(o)
    for i in 1:p
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())
    end
    for i in 1:p
        MOI.add_constraint(o, -N * z[i]- w[i], MOI.LessThan(0.0))
        MOI.add_constraint(o, N * z[i]- w[i], MOI.GreaterThan(0.0))
        # Indicator: z[i] = 1 => -N <= w[i] <= N
        gl = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),],
            [0.0, 0.0], )
        gg = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),],
            [0.0, 0.0], )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(N)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-N))) 
    end
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(o, b, MOI.LessThan(N))
    MOI.add_constraint(o, b, MOI.GreaterThan(-N))
    lmo = FrankWolfe.MathOptLMO(o)
    global_bounds = Boscia.IntegerBounds()
    for i in 1:p
        push!(global_bounds, (p+i, MOI.GreaterThan(0.0)))
        push!(global_bounds, (p+i, MOI.LessThan(1.0)))
        push!(global_bounds, (i, MOI.GreaterThan(-N)))
        push!(global_bounds, (i, MOI.LessThan(N)))
    end
    push!(global_bounds, (2p+1, MOI.GreaterThan(-N)))
    push!(global_bounds, (2p+1, MOI.LessThan(N)))

    direction = Vector{Float64}(undef,2p + 1)
    Random.rand!(direction)
    v = compute_extreme_point(lmo, direction)
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)
    α = 1.3
    function f(θ)
        w = @view(θ[1:p])
        b = θ[end]
        s = sum(1:n0) do i
            a = dot(w, X0[:,i]) + b
            1/n0 * (exp(a) - y0[i] * a)
        end
        s + α * norm(w)^2
    end
    function grad!(storage, θ)
        w = @view(θ[1:p])
        b = θ[end]
        storage[1:p] .= 2α .* w
        storage[p+1:2p] .= 0
        storage[end] = 0
        for i in 1:n0
            xi = @view(X0[:,i])
            a = dot(w, xi) + b
            storage[1:p] .+= 1/n0 * xi * exp(a)
            storage[1:p] .-= 1/n0 * y0[i] * xi
            storage[end] += 1/n0 * (exp(a) - y0[i])
        end
        storage ./= norm(storage)
        return storage
    end
    time_lmo = Boscia.TimeTrackingLMO(lmo)
    active_set = FrankWolfe.ActiveSet([(1.0, v)]) 
    m = Boscia.SimpleOptimizationProblem(f, grad!, 2p+1, collect(p+1:2p), time_lmo, global_bounds) 

    # TO DO: how to do this elegantly
    nodeEx = Boscia.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, Boscia.IntegerBounds(), 1, 1e-3, Millisecond(0))

    # create tree
    function perform_strong_branch(tree, node)
        return node.level <= length(tree.root.problem.integer_variables)
    end
    branching_strategy = Boscia.HybridStrongBranching(10, 1e-3, HiGHS.Optimizer(), perform_strong_branch)
    MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)
    tree = Bonobo.initialize(;
        traverse_strategy = Bonobo.BFS(),
        Node = typeof(nodeEx),
        root = (problem=m, current_node_id = Ref{Int}(0), updated_incumbent = Ref{Bool}(false), options= Dict{Symbol, Any}(:verbose => false, :dual_gap_decay_factor => 0.7, :dual_gap => 1e-6, :max_fw_iter => 10000, :min_node_fw_epsilon => 1e-6)),
        branch_strategy = branching_strategy, #() ->
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
    #@show x
    @test sum(x[p+1:2p]) <= k
    #println("Non zero entries:")
    #for i in 1:p
    #    if isapprox(x[i+p], 1.0, atol = 1e-5, rtol= 1e-5)
    #        println("$(i)th entry: $(x[i])")
    #    end
    #end
end 

n0g=20
pg = n0g

# underlying true weights
const w0g = 2 * rand(Float64, pg) .- 1
# set 50 entries to 0
for _ in 1:15
    w0g[rand(1:pg)] = 0
end
const b0g = 2 * rand(Float64) - 1
const X0g = 2 * rand(Float64, n0g, pg) .- 1 
const y0g = map(1:n0g) do idx
    a = dot(X0g[idx,:], w0g) + b0g
    rand(Distributions.Poisson(exp(a)))
end
Ng = 5.0

k = 10
group_size = convert(Int64, floor(pg/k))
groups= []
for i in 1:(k-1)
    push!(groups, ((i-1)*group_size+1):(i*group_size))
end
push!(groups,((k-1)*group_size+1):pg)

@testset "Sparse Group Poisson" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    w = MOI.add_variables(o, pg)
    z = MOI.add_variables(o, pg)
    b = MOI.add_variable(o)
    for i in 1:pg
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())
    end
    for i in 1:pg
        MOI.add_constraint(o, -Ng * z[i]- w[i], MOI.LessThan(0.0))
        MOI.add_constraint(o, Ng * z[i]- w[i], MOI.GreaterThan(0.0))
        # Indicator: z[i] = 1 => -Ng <= w[i] <= Ng
        gl = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),],
            [0.0, 0.0], )
        gg = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),],
            [0.0, 0.0], )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(Ng)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-Ng)))
    end
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(o, b, MOI.LessThan(Ng))
    MOI.add_constraint(o, b, MOI.GreaterThan(-Ng))
    for i in 1:k
        #MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(group_size),x[groups[i]]), 0.0), MOI.GreaterThan(1.0))
        MOI.add_constraint(o, sum(z[groups[i]], init=0.0), MOI.GreaterThan(1.0))
    end
    lmo = FrankWolfe.MathOptLMO(o)
    global_bounds = Boscia.IntegerBounds()
    for i in 1:pg
        push!(global_bounds, (pg+i, MOI.GreaterThan(0.0)))
        push!(global_bounds, (pg+i, MOI.LessThan(1.0)))
        push!(global_bounds, (i, MOI.GreaterThan(-Ng)))
        push!(global_bounds, (i, MOI.LessThan(Ng)))
    end
    push!(global_bounds, (2pg+1, MOI.GreaterThan(-Ng)))
    push!(global_bounds, (2pg+1, MOI.LessThan(Ng)))

    direction = Vector{Float64}(undef,2pg + 1)
    Random.rand!(direction)
    v = compute_extreme_point(lmo, direction)
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)
    α = 1.3
    function f(θ)
        w = @view(θ[1:pg])
        b = θ[end]
        s = sum(1:n0g) do i
            a = dot(w, X0g[:,i]) + b
            1/n0g * (exp(a) - y0g[i] * a)
        end
        s + α * norm(w)^2
    end
    function grad!(storage, θ)
        w = @view(θ[1:pg])
        b = θ[end]
        storage[1:pg] .= 2α .* w
        storage[pg+1:2pg] .= 0
        storage[end] = 0
        for i in 1:n0g
            xi = @view(X0g[:,i])
            a = dot(w, xi) + b
            storage[1:pg] .+= 1/n0g * xi * exp(a)
            storage[1:pg] .-= 1/n0g * y0g[i] * xi
            storage[end] += 1/n0g * (exp(a) - y0g[i])
        end
        storage ./= norm(storage)
        return storage
    end
    time_lmo = Boscia.TimeTrackingLMO(lmo)
    active_set = FrankWolfe.ActiveSet([(1.0, v)]) 
    m = Boscia.SimpleOptimizationProblem(f, grad!, 2pg+1, collect(pg+1:2pg), time_lmo, global_bounds) 

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
    # println("Solution: $(x[1:p])")
    @test sum(x[pg+1:2pg]) <= k
        #println("Non zero entries:")
    #for i in 1:pg
    #    if x[i+pg] == 1
    #        println("$(i)th entry: $(x[i])")
    #    end
    #end
end 


@testset "Strong branching sparse group poisson" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    w = MOI.add_variables(o, pg)
    z = MOI.add_variables(o, pg)
    b = MOI.add_variable(o)
    for i in 1:pg
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())
    end
    for i in 1:pg
        MOI.add_constraint(o, -Ng * z[i]- w[i], MOI.LessThan(0.0))
        MOI.add_constraint(o, Ng * z[i]- w[i], MOI.GreaterThan(0.0))
        # Indicator: z[i] = 1 => -Ng <= w[i] <= Ng
        gl = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),],
            [0.0, 0.0], )
        gg = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),],
            [0.0, 0.0], )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(Ng)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-Ng)))
    end
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(o, b, MOI.LessThan(Ng))
    MOI.add_constraint(o, b, MOI.GreaterThan(-Ng))
    for i in 1:k
        #MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(group_size),x[groups[i]]), 0.0), MOI.GreaterThan(1.0))
        MOI.add_constraint(o, sum(z[groups[i]], init=0.0), MOI.GreaterThan(1.0))
    end
    lmo = FrankWolfe.MathOptLMO(o)
    global_bounds = Boscia.IntegerBounds()
    for i in 1:pg
        push!(global_bounds, (pg+i, MOI.GreaterThan(0.0)))
        push!(global_bounds, (pg+i, MOI.LessThan(1.0)))
        push!(global_bounds, (i, MOI.GreaterThan(-Ng)))
        push!(global_bounds, (i, MOI.LessThan(Ng)))
    end
    push!(global_bounds, (2pg+1, MOI.GreaterThan(-Ng)))
    push!(global_bounds, (2pg+1, MOI.LessThan(Ng)))

    direction = Vector{Float64}(undef,2pg + 1)
    Random.rand!(direction)
    v = compute_extreme_point(lmo, direction)
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)
    α = 1.3
    function f(θ)
        w = @view(θ[1:pg])
        b = θ[end]
        s = sum(1:n0g) do i
            a = dot(w, X0g[:,i]) + b
            1/n0g * (exp(a) - y0g[i] * a)
        end
        s + α * norm(w)^2
    end
    function grad!(storage, θ)
        w = @view(θ[1:pg])
        b = θ[end]
        storage[1:pg] .= 2α .* w
        storage[pg+1:2pg] .= 0
        storage[end] = 0
        for i in 1:n0g
            xi = @view(X0g[:,i])
            a = dot(w, xi) + b
            storage[1:pg] .+= 1/n0g * xi * exp(a)
            storage[1:pg] .-= 1/n0g * y0g[i] * xi
            storage[end] += 1/n0g * (exp(a) - y0g[i])
        end
        storage ./= norm(storage)
        return storage
    end
    time_lmo = Boscia.TimeTrackingLMO(lmo)
    active_set = FrankWolfe.ActiveSet([(1.0, v)]) 
    m = Boscia.SimpleOptimizationProblem(f, grad!, 2pg+1, collect(pg+1:2pg), time_lmo, global_bounds) 

    # TO DO: how to do this elegantly
    nodeEx = Boscia.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, Boscia.IntegerBounds(), 1, 1e-3, Millisecond(0))

    # create tree
    function perform_strong_branch(tree, node)
        return node.level <= length(tree.root.problem.integer_variables)
    end
    branching_strategy = Boscia.HybridStrongBranching(10, 1e-3, HiGHS.Optimizer(), perform_strong_branch)
    MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)
    tree = Bonobo.initialize(;
        traverse_strategy = Bonobo.BFS(),
        Node = typeof(nodeEx),
        root = (problem=m, current_node_id = Ref{Int}(0), updated_incumbent = Ref{Bool}(false), options= Dict{Symbol, Any}(:verbose => false, :dual_gap_decay_factor => 0.7, :dual_gap => 1e-6, :max_fw_iter => 10000, :min_node_fw_epsilon => 1e-6)),
        branch_strategy = branching_strategy, #() ->
    )
    Bonobo.set_root!(tree, 
    (active_set = active_set, 
    discarded_vertices = vertex_storage,
    local_bounds = Boscia.IntegerBounds(),
    level = 1, 
    fw_dual_gap_limit= 1e-3,
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
    #@show x
    @test sum(x[pg+1:2pg]) <= k
    #println("Non zero entries:")
    #for i in 1:pg
    #    if x[i+pg] == 1
    #        println("$(i)th entry: $(x[i])")
    #    end
    #end
end 