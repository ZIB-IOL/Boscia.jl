using Boscia
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

seed = rand(UInt64)
@show seed
Random.seed!(seed)

include("interface_test.jl")

@testset "Linear feasible" begin
    n = 10
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
    end
    MOI.add_constraint(o, 1.0x[1] + 1.0x[2], MOI.LessThan(1.5))
    @test Boscia.is_linear_feasible(o, 2*ones(n)) == false
    @test Boscia.is_linear_feasible(o, vcat([1.0, 0.5], ones(n - 2)))
    @test Boscia.is_linear_feasible(o, vcat([0.5, 0.5], ones(n - 2)))
    @test Boscia.is_linear_feasible(o, vcat([0.0, 0.0], ones(n - 2)))
end

include("LMO_test.jl")
include("indicator_test.jl")


# Takes pretty long, only include if you want to test this specifically
#include("infeasible_pairwise.jl")

include("sparse_regression.jl")
include("poisson.jl")
include("mean_risk.jl")
include("time_limit.jl")

n= 10
const diff1 = rand(Bool, n) * 0.8 .+ 1.1
@testset "Strong branching" begin
    function f(x)
        return sum((x .- diff1) .^ 2)
    end
    function grad!(storage, x)
        @. storage = 2 * (x - diff1)
    end

    lb = min(sum(round.(diff1)), sum(diff1)) - 0.1
    ub = max(sum(round.(diff1)), sum(diff1)) + 0.1
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(4.0))
        MOI.add_constraint(o, xi, MOI.Integer()) # or MOI.Integer()
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.LessThan(ub),
    )
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.GreaterThan(lb),
    )
    lmo = FrankWolfe.MathOptLMO(o)
   
    branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, HiGHS.Optimizer())
    MOI.set(branching_strategy.optimizer, MOI.Silent(), true)

    x, _, result_strong_branching =
        Boscia.solve(f, grad!, lmo, verbose=true, branching_strategy=branching_strategy)

    @test isapprox(x, round.(diff1), atol=1e-5, rtol=1e-5)
end

@testset "Hybrid branching" begin
    function f(x)
        return sum((x .- diff1) .^ 2)
    end
    function grad!(storage, x)
        @. storage = 2 * (x - diff1)
    end

    lb = min(sum(round.(diff1)), sum(diff1)) - 0.1
    ub = max(sum(round.(diff1)), sum(diff1)) + 0.1
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(4.0))
        MOI.add_constraint(o, xi, MOI.Integer()) # or MOI.Integer()
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.LessThan(ub),
    )
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.GreaterThan(lb),
    )
    lmo = FrankWolfe.MathOptLMO(o)
    
   
    function perform_strong_branch(tree, node)
        return node.level <= length(tree.root.problem.integer_variables) / 3
    end
    branching_strategy =
        Boscia.HybridStrongBranching(10, 1e-3, HiGHS.Optimizer(), perform_strong_branch)
    MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)

    x,_,result = Boscia.solve(f, grad!, lmo, verbose = true, branching_strategy=branching_strategy) 
    
    @test isapprox(x, round.(diff1), atol=1e-5, rtol=1e-5)
end

n = 20
diffi = Random.rand(Bool, n) * 0.6 .+ 0.3

@testset "Away FW, monotonic step size and predefined start point" begin
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

    o_f= SCIP.Optimizer()
    MOI.set(o_f, MOI.Silent(), true)
    MOI.empty!(o_f)
    x = MOI.add_variables(o_f, n)
    for xi in x
        MOI.add_constraint(o_f, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o_f, xi, MOI.LessThan(1.0))
    end
    lmo_f = FrankWolfe.MathOptLMO(o_f)

    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    direction = rand(n)
    v = compute_extreme_point(lmo, direction)
    x,_,_,_,_,active_set = FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo_f, v, max_iteration = 3)

    line_search = FrankWolfe.MonotonicStepSize()
    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, variant = Boscia.AFW, line_search=line_search, active_set=active_set, time_limit = 300)

    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
end


for file in readdir(joinpath(@__DIR__, "../examples/"), join=true)
    if endswith(file, "jl")
        include(file)
    end
end
