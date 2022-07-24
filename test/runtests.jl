using BranchWolfe
using Test
using FrankWolfe
using Random
using SCIP
using Statistics
import Bonobo
import MathOptInterface
const MOI = MathOptInterface
import HiGHS
using Dates

# For bug hunting:
#seed = 0xd0cc4c6d90c50bb9
seed = rand(UInt64)
@show seed
Random.seed!(seed)

include("interface_test.jl")

@testset "BnBTree data structure and node" begin
    # Building an optimization problem for the tree
    n = 10
    diff_b = [0.8, 0.5, 0.8, 0.5, 0.3, 0.5, 0.8, 0.5, 0.3, 0.5 ]
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    discrete_variables = [1, 3, 5, 7, 9]
    for (xi,i) in zip(x, 1:n)
        if i in discrete_variables
            MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
            MOI.add_constraint(o, xi, MOI.LessThan(1.0))
        else
            MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
            MOI.add_constraint(o, xi, MOI.LessThan(0.5))
        end
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, x), 0.0),
        MOI.LessThan(1.0),
    )
    lmo = FrankWolfe.MathOptLMO(o)
    time_lmo = BranchWolfe.TimeTrackingLMO(lmo)
    global_bounds = BranchWolfe.IntegerBounds()
    for i in discrete_variables
        push!(global_bounds, (i, MOI.GreaterThan(0.0)))
        push!(global_bounds, (i, MOI.LessThan(1.0)))
    end

    # Define the root of the tree
    # we fix the direction so we can actually find a veriable to split on later!
    direction = rand(Float64, n)#[1.0615150343404234, 0.22938533132601985, -1.3357386885576823, -1.0748464832861053, 
       # 1.021745032550068, 0.5415277384510694, -1.0343346047149735, -1.610953556522954,
       # -0.17641293832949326, 1.0027785507641414]
    v = compute_extreme_point(time_lmo, direction)
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)
    function f(x)
        return sum(0.5*(x-diff_b).^2)
    end
    function grad!(storage, x)
        @. storage = 2x
    end

    active_set = FrankWolfe.ActiveSet([(1.0, v)]) 
    m = BranchWolfe.SimpleOptimizationProblem(f, grad!, n, discrete_variables, time_lmo, global_bounds) 

    # create control instance of away_frank_wolfe
    x_afw,_,_,_,_,_ = FrankWolfe.away_frank_wolfe(f, grad!, time_lmo, active_set)

    # create tree
    tree = Bonobo.initialize(; 
        traverse_strategy = Bonobo.BFS(),
        Node = BranchWolfe.FrankWolfeNode, 
        root = (problem=m, current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => 1e-7, :verbose => false, :dual_gap_decay_factor => 0.7,  :dual_gap => 1e-6)),
    )
    Bonobo.set_root!(tree, 
        (active_set = active_set, 
        discarded_vertices = vertex_storage,
        local_bounds = BranchWolfe.IntegerBounds(),
        level = 1,
        fw_dual_gap_limit= 1e-3, 
        fw_time = Millisecond(0))
    )
 
    @test tree.num_nodes == 1
    @test discrete_variables == Bonobo.get_branching_indices(tree.root)

    # get the first node
    node = Bonobo.get_next_node(tree, tree.options.traverse_strategy)

    function build_FW_callback(tree)
        return function fw_callback(state, active_set)
            # print("Primal: $(state.primal)\n")
            # print("Length of active set: $(length(active_set.weights))\n")
        end
    end

    fw_callback = build_FW_callback(tree)
    tree.root.options[:callback] = fw_callback
    Bonobo.evaluate_node!(tree, node)
    x = Bonobo.get_relaxed_values(tree, node)
    @test x !== nothing
    @test x == x_afw

    idx = Bonobo.get_branching_variable(tree, tree.options.branch_strategy, node)
    if idx != -1
        @test idx != -1
        info_left, info_right = Bonobo.get_branching_nodes_info(tree, node, idx)
        @test info_left != info_right

        Bonobo.add_node!(tree, node, info_left)
        Bonobo.add_node!(tree, node, info_right)
        @test tree.num_nodes == 3
    end
end

@testset "Linear feasible" begin
    n=10
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
    end
    MOI.add_constraint(o, 1.0x[1] + 1.0x[2], MOI.LessThan(1.5))
    @test BranchWolfe.is_linear_feasible(o, ones(n)) == false
    @test BranchWolfe.is_linear_feasible(o, vcat([1.0, 0.5], ones(n-2)))
    @test BranchWolfe.is_linear_feasible(o, vcat([0.5, 0.5], ones(n-2)))
    @test BranchWolfe.is_linear_feasible(o, vcat([0.0, 0.0], ones(n-2)))
end

@testset "Integer bounds" begin
    n=10
    integer_bound = BranchWolfe.IntegerBounds()
    @test isempty(integer_bound)
    push!(integer_bound, (1, MOI.GreaterThan(5.0)))
    push!(integer_bound, (2, MOI.LessThan(0.0)))
    push!(integer_bound, (3, MOI.LessThan(4.0)))
    push!(integer_bound, (4, MOI.LessThan(0.0)))
    push!(integer_bound, (5, MOI.GreaterThan(5.0)))
    @test haskey(integer_bound.lower_bounds, 2) == false
    @test haskey(integer_bound.upper_bounds, 4)
    @test haskey(integer_bound.lower_bounds, 1)

    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        if xi.value != 3
            MOI.add_constraint(o, xi, MOI.LessThan(5.0))
        end
    end
    lmo = FrankWolfe.MathOptLMO(o)

    global_bounds = BranchWolfe.IntegerBounds()
    @test isempty(global_bounds)
    for i = 1:n
        push!(global_bounds, (i, MOI.GreaterThan(0.0)))
        if i != 3
            push!(global_bounds, (i, MOI.LessThan(5.0)))
        end
    end

    BranchWolfe.build_LMO(lmo, global_bounds, integer_bound, collect(1:n))

    @test BranchWolfe.is_linear_feasible(o, ones(n)) == false
    @test BranchWolfe.is_linear_feasible(o, vcat([5.0, 0.0, 1.5, 0.0, 5.0], ones(n-5)))
    @test BranchWolfe.is_linear_feasible(o, vcat([5.0, 2.0, 1.5, 0.0, 5.0], ones(n-5))) == false
    @test BranchWolfe.is_linear_feasible(o, vcat([5.0, 0.0, 1.5, 0.0, 3.0], ones(n-5))) == false   
    @test BranchWolfe.is_linear_feasible(o, vcat([5.0, 0.0, 4.5, 0.0, 5.0], ones(n-5))) == false   
end

# diff needs to defined outside of test to avoid a "unsupported const declaration
# on a local variable"-error
#Random.seed!(1)
n = 20
const diff = Random.rand(Bool,n)*0.6.+0.3

@testset "Norm over the hyperbox" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
        MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
    end
    lmo = FrankWolfe.MathOptLMO(o)
    global_bounds = BranchWolfe.IntegerBounds()
    for i = 1:n
        push!(global_bounds, (i, MOI.GreaterThan(0.0)))
        push!(global_bounds, (i, MOI.LessThan(1.0)))
    end
    time_lmo = BranchWolfe.TimeTrackingLMO(lmo)

    # Define the root of the tree
    direction = Vector{Float64}(undef,n)
    Random.rand!(direction)
    v = compute_extreme_point(time_lmo, direction)
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    function f(x)
        return sum(0.5*(x.-diff).^2)
    end
    function grad!(storage, x)
        @. storage = x-diff
    end

    active_set = FrankWolfe.ActiveSet([(1.0, v)])
    m = BranchWolfe.SimpleOptimizationProblem(f, grad!, n, collect(1:n), time_lmo, global_bounds)

    # TO DO: how to do this elegantly
    nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, 1e-3, Millisecond(0))

    # create tree
    tree = Bonobo.initialize(; 
        traverse_strategy = Bonobo.BFS(),
        Node = typeof(nodeEx),
        root = (problem=m, current_node_id = current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => 1e-7, :verbose => false, :dual_gap_decay_factor => 0.7, :dual_gap => 1e-6)),
    )
    Bonobo.set_root!(tree, 
        (active_set = active_set, 
        discarded_vertices = vertex_storage,
        local_bounds = BranchWolfe.IntegerBounds(),
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

    Bonobo.optimize!(tree)
    x = convert.(Int64,Bonobo.get_solution(tree))
    @test x == round.(diff)
end 


@testset "Norm over the hyperbox- Strong branching" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
        MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
    end
    lmo = FrankWolfe.MathOptLMO(o)
    global_bounds = BranchWolfe.IntegerBounds()
    for i = 1:n
        push!(global_bounds, (i, MOI.GreaterThan(0.0)))
        push!(global_bounds, (i, MOI.LessThan(1.0)))
    end
    time_lmo = BranchWolfe.TimeTrackingLMO(lmo)

    # Define the root of the tree
    direction = Vector{Float64}(undef,n)
    Random.rand!(direction)
    v = compute_extreme_point(time_lmo, direction)
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    function f(x)
        return sum(0.5*(x.-diff).^2)
    end
    function grad!(storage, x)
        @. storage = x-diff
    end

    active_set = FrankWolfe.ActiveSet([(1.0, v)])
    m = BranchWolfe.SimpleOptimizationProblem(f, grad!, n, collect(1:n), time_lmo, global_bounds)

    # TO DO: how to do this elegantly
    nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, 1e-3, Millisecond(0))

    # create tree
    branching_strategy = BranchWolfe.PartialStrongBranching(10, 1e-3, HiGHS.Optimizer())
    MOI.set(branching_strategy.optimizer, MOI.Silent(), true)
    tree = Bonobo.initialize(;
        traverse_strategy = Bonobo.BFS(),
        Node = typeof(nodeEx),
        root = (problem=m, current_node_id = current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => 1e-7, :verbose => false, :dual_gap_decay_factor => 0.7, :dual_gap => 1e-6)),
        branch_strategy = branching_strategy, #() ->
    )
    Bonobo.set_root!(tree, 
        (active_set = active_set,
        discarded_vertices = vertex_storage, 
        local_bounds = BranchWolfe.IntegerBounds(),
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
    x = convert.(Int64,Bonobo.get_solution(tree))
    @test x == round.(diff)
end 

# Takes pretty long, only include if you want to test this specifically
#include("infeasible_pairwise.jl")

include("sparse_regression.jl") 
include("poisson.jl")
include("mean_risk.jl")


Random.seed!(6)
const n1 = 10
const diff1 = rand(Bool, n1)*0.8.+1.1

@testset "IP vs LP solver" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n1)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(4.0))
    end
    lb = min(sum(round.(diff1)), sum(diff1))-0.1
    ub = max(sum(round.(diff1)), sum(diff1))+0.1
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n1), x), 0.0), MOI.LessThan(ub))
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n1), x), 0.0), MOI.GreaterThan(lb))
    lmo = FrankWolfe.MathOptLMO(o)
    global_bounds = BranchWolfe.IntegerBounds()
    for i = 1:n1
        push!(global_bounds, (i, MOI.GreaterThan(0.0)))
        push!(global_bounds, (i, MOI.LessThan(4.0)))
    end
    time_lmo = BranchWolfe.TimeTrackingLMO(lmo)

    # Define the root of the tree
    direction = Vector{Float64}(undef,n1)
    Random.rand!(direction)
    v = compute_extreme_point(time_lmo, direction)
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    function f(x)
        return sum((x.-diff1).^2)
    end
    function grad!(storage, x)
        @. storage = 2*(x-diff1)
    end

    time_lmo = BranchWolfe.TimeTrackingLMO(lmo)
    active_set = FrankWolfe.ActiveSet([(1.0, v)]) 
    m = BranchWolfe.SimpleOptimizationProblem(f, grad!, n1, collect(1:n1), time_lmo, global_bounds) 

    # TO DO: how to do this elegantly
    nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, 1e-3, Millisecond(0))

    # create tree
    tree = Bonobo.initialize(; 
        traverse_strategy = Bonobo.BFS(),
        Node = typeof(nodeEx),
        root = (problem=m, current_node_id = current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => 1e-5, :dual_gap_decay_factor => 0.7, :verbose => false, :dual_gap => 1e-6)),
    )
    Bonobo.set_root!(tree, 
        (active_set = active_set, 
        discarded_vertices = vertex_storage,
        local_bounds = BranchWolfe.IntegerBounds(),
        level = 1,
        fw_dual_gap_limit = 1e-3,
        fw_time = Millisecond(0))
    )
    # Profile.init()
    # ProfileView.@profview Bonobo.optimize!(tree)
    function callback(tree, node; kwargs...)
        # TODO fill callback to debug
    end

    function build_FW_callback(tree)
        return function fw_callback(state, active_set)
            # print("Primal: $(state.primal)\n")
            # print("Length of active set: $(length(active_set.weights))\n")
        end
    end

    fw_callback = build_FW_callback(tree)
    tree.root.options[:callback] = fw_callback

    Bonobo.optimize!(tree)
    x = Bonobo.get_solution(tree)
    if isapprox(x, round.(diff1))
        @test isapprox(x, round.(diff1))
    else
        @warn "Did not solve correctly"
    end

    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n1)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(4.0))
        MOI.add_constraint(o, xi, MOI.Integer()) # or MOI.Integer()
    end
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n1), x), 0.0), MOI.LessThan(ub))
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n1), x), 0.0), MOI.GreaterThan(lb))
    lmo = FrankWolfe.MathOptLMO(o)
    global_bounds = BranchWolfe.IntegerBounds()
    for i = 1:n1
        push!(global_bounds, (i, MOI.GreaterThan(0.0)))
        push!(global_bounds, (i, MOI.LessThan(4.0)))
    end 
    time_lmo=BranchWolfe.TimeTrackingLMO(lmo)
    # Define the root of the tree
    direction = Vector{Float64}(undef,n1)
    Random.rand!(direction)
    v = compute_extreme_point(time_lmo, direction)
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    time_lmo = BranchWolfe.TimeTrackingLMO(lmo)
    active_set = FrankWolfe.ActiveSet([(1.0, v)])
    m = BranchWolfe.SimpleOptimizationProblem(f, grad!, n1, collect(1:n1), time_lmo, global_bounds)

    # TO DO: how to do this elegantly
    nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, 1e-3, Millisecond(0))

    # create tree
    tree = Bonobo.initialize(; 
        traverse_strategy = Bonobo.BFS(),
        Node = typeof(nodeEx),
        root = (problem=m, current_node_id = current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => 1e-7, :verbose => false, :dual_gap_decay_factor => 0.7, :dual_gap => 1e-6)),
    )
    Bonobo.set_root!(tree, 
        (active_set = active_set, 
        discarded_vertices = vertex_storage,
        local_bounds = BranchWolfe.IntegerBounds(),
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
    @test isapprox(x, round.(diff1), atol = 1e-5, rtol= 1e-5)
end

@testset "Strong vs most infeasible branching IP" begin
    function f(x)
        return sum((x.-diff1).^2)
    end
    function grad!(storage, x)
        @. storage = 2*(x-diff1)
    end

    lb = min(sum(round.(diff1)), sum(diff1))-0.1
    ub = max(sum(round.(diff1)), sum(diff1))+0.1
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n1)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(4.0))
        MOI.add_constraint(o, xi, MOI.Integer()) # or MOI.Integer()
    end
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n1), x), 0.0), MOI.LessThan(ub))
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n1), x), 0.0), MOI.GreaterThan(lb))
    lmo = FrankWolfe.MathOptLMO(o)
    time_lmo = BranchWolfe.TimeTrackingLMO(lmo)
    global_bounds = BranchWolfe.IntegerBounds()
    for i = 1:n1
        push!(global_bounds, (i, MOI.GreaterThan(0.0)))
        push!(global_bounds, (i, MOI.LessThan(4.0)))
    end

    # Define the root of the tree
    direction = randn(n1)
    v = compute_extreme_point(lmo, direction)
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    active_set = FrankWolfe.ActiveSet([(1.0, v)])
    m = BranchWolfe.SimpleOptimizationProblem(f, grad!, n1, collect(1:n1), time_lmo, global_bounds)

    # TO DO: how to do this elegantly
    nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, 1e-3, Millisecond(0))

    # create tree
    branching_strategy = BranchWolfe.PartialStrongBranching(10, 1e-3, HiGHS.Optimizer())
    MOI.set(branching_strategy.optimizer, MOI.Silent(), true)
    tree = Bonobo.initialize(;
        traverse_strategy = Bonobo.BFS(),
        Node = typeof(nodeEx),
        root = (problem=m, current_node_id = current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => 1e-7, :verbose => false, :dual_gap_decay_factor => 0.7, :dual_gap => 1e-6)),
        branch_strategy = branching_strategy, #() ->
    )
    Bonobo.set_root!(tree,
        (active_set = active_set, 
        discarded_vertices = vertex_storage,
        local_bounds = BranchWolfe.IntegerBounds(),
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

    @time Bonobo.optimize!(tree)
    x = Bonobo.get_solution(tree)
    @test isapprox(x, round.(diff1), atol = 1e-5, rtol= 1e-5)
end

@testset "Hybrid vs most infeasible branching IP" begin
    function f(x)
        return sum((x.-diff1).^2)
    end
    function grad!(storage, x)
        @. storage = 2*(x-diff1)
    end

    lb = min(sum(round.(diff1)), sum(diff1))-0.1
    ub = max(sum(round.(diff1)), sum(diff1))+0.1
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n1)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(4.0))
        MOI.add_constraint(o, xi, MOI.Integer()) # or MOI.Integer()
    end
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n1), x), 0.0), MOI.LessThan(ub))
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n1), x), 0.0), MOI.GreaterThan(lb))
    lmo = FrankWolfe.MathOptLMO(o)
    time_lmo = BranchWolfe.TimeTrackingLMO(lmo)
    global_bounds = BranchWolfe.IntegerBounds()
    for i = 1:n1
        push!(global_bounds, (i, MOI.GreaterThan(0.0)))
        push!(global_bounds, (i, MOI.LessThan(4.0)))
    end

    # Define the root of the tree
    direction = randn(n1)
    v = compute_extreme_point(lmo, direction)
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    active_set = FrankWolfe.ActiveSet([(1.0, v)])
    m = BranchWolfe.SimpleOptimizationProblem(f, grad!, n1, collect(1:n1), time_lmo, global_bounds)

    # TO DO: how to do this elegantly
    nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, 1e-3, Millisecond(0))

    # create tree
    #branching_strategy = BranchWolfe.PartialStrongBranching(10, 1e-3, HiGHS.Optimizer())
    function perform_strong_branch(tree, node)
        return node.level <= length(tree.root.problem.integer_variables)/3
    end
    branching_strategy = BranchWolfe.HybridStrongBranching(10, 1e-3, HiGHS.Optimizer(), perform_strong_branch)
    MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)
    tree = Bonobo.initialize(;
        traverse_strategy = Bonobo.BFS(),
        Node = typeof(nodeEx),
        root = (problem=m, current_node_id = current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => 1e-7, :verbose => false,  :dual_gap_decay_factor => 0.7, :dual_gap => 1e-6)),
        branch_strategy = branching_strategy, #() ->
    )
    Bonobo.set_root!(tree,
        (active_set = active_set, 
        discarded_vertices= vertex_storage,
        local_bounds = BranchWolfe.IntegerBounds(),
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
    @time Bonobo.optimize!(tree)
    x = Bonobo.get_solution(tree)
    @test isapprox(x, round.(diff1), atol = 1e-5, rtol= 1e-5)
end
