using Test
using Boscia
using FrankWolfe
using Random
using SCIP
import MathOptInterface
import Bonobo
using HiGHS
using Printf
using Dates
const MOI = MathOptInterface
const MOIU = MOI.Utilities
using StableRNGs

println("\nLMO Tests")

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

@testset "Integer bounds" begin
    n = 10
    integer_bound = Boscia.IntegerBounds()
    @test isempty(integer_bound)
    push!(integer_bound, (1, 5.0), :greaterthan)
    push!(integer_bound, (2, 0.0), :lessthan)
    push!(integer_bound, (3, 4.0), :lessthan)
    push!(integer_bound, (4, 0.0), :lessthan)
    push!(integer_bound, (5, 5.0), :greaterthan)
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
    lmo = Boscia.MathOptBLMO(o)

    global_bounds = Boscia.IntegerBounds()
    @test isempty(global_bounds)
    for i in 1:n
        push!(global_bounds, (i, 0.0), :greaterthan)
        if i != 3
            push!(global_bounds, (i, 5.0), :lessthan)
        end
    end

    Boscia.build_LMO(lmo, global_bounds, integer_bound, collect(1:n))

    @test Boscia.is_linear_feasible(o, ones(n)) == false
    @test Boscia.is_linear_feasible(o, vcat([5.0, 0.0, 1.5, 0.0, 5.0], ones(n - 5)))
    @test Boscia.is_linear_feasible(o, vcat([5.0, 2.0, 1.5, 0.0, 5.0], ones(n - 5))) == false
    @test Boscia.is_linear_feasible(o, vcat([5.0, 0.0, 1.5, 0.0, 3.0], ones(n - 5))) == false
    @test Boscia.is_linear_feasible(o, vcat([5.0, 0.0, 5.5, 0.0, 5.0], ones(n - 5))) == false
end

n = 20
diffi = rand(rng, Bool, n) * 0.6 .+ 0.3

@testset "Cube LMO" begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    lbs = zeros(n)
    ubs = ones(n)
    int_vars = collect(1:n)

    sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)

    x, _, result = Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n)

    # testing for cube inface oracles
    settings = Boscia.create_default_settings()
    settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()
    x_dicg, _, result_dicg =
        Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, settings=settings)

    @test sum(isapprox.(x, round.(diffi), atol=1e-6, rtol=1e-2)) == n
    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    @test sum(isapprox.(x_dicg, round.(diffi), atol=1e-6, rtol=1e-2)) == n
    @test isapprox(f(x_dicg), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
end

@testset "BLMO - Strong Branching" begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end
    @testset "Partial Strong Branching" begin
        int_vars = collect(1:n)
        lbs = zeros(n)
        ubs = ones(n)

        sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
        blmo = Boscia.ManagedBoundedLMO(sblmo, lbs[int_vars], ubs[int_vars], int_vars, n)

        branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        x, _, result = Boscia.solve(f, grad!, blmo, settings=settings)

        @test x == round.(diffi)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end
    @testset "Hybrid Strong Branching" begin
        int_vars = collect(1:n)
        lbs = zeros(n)
        ubs = ones(n)

        sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
        blmo = Boscia.ManagedBoundedLMO(sblmo, lbs[int_vars], ubs[int_vars], int_vars, n)

        function perform_strong_branch(tree, node)
            return node.level <= length(tree.root.problem.integer_variables) / 3
        end
        branching_strategy = Boscia.HybridStrongBranching(10, 1e-3, blmo, perform_strong_branch)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        x, _, result = Boscia.solve(f, grad!, blmo, settings=settings)

        @test x == round.(diffi)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end
end

n = 20
x_sol = rand(rng, 1:floor(Int, n / 4), n)
N = sum(x_sol)
dir = vcat(fill(1, floor(Int, n / 2)), fill(-1, floor(Int, n / 2)), fill(0, mod(n, 2)))
diffi = x_sol + 0.3 * dir

@testset "Probability Simplex LMO" begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    sblmo = Boscia.ProbabilitySimplexSimpleBLMO(N)

    x, _, result = Boscia.solve(f, grad!, sblmo, fill(0.0, n), fill(1.0 * N, n), collect(1:n), n)

    # testing for Probability simplex inface oracles
    settings = Boscia.create_default_settings()
    settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()
    x_dicg, _, result_dicg = Boscia.solve(
        f,
        grad!,
        sblmo,
        fill(0.0, n),
        fill(1.0 * N, n),
        collect(1:n),
        n,
        settings=settings,
    )

    @test sum(isapprox.(x, x_sol, atol=1e-6, rtol=1e-2)) == n
    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    @test sum(isapprox.(x_dicg, round.(diffi), atol=1e-6, rtol=1e-2)) == n
    @test isapprox(f(x_dicg), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
end

n = 20
x_sol = rand(rng, 1:floor(Int, n / 4), n)
diffi = x_sol + 0.3 * rand(rng, [-1, 1], n)

@testset "Unit Simplex LMO" begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    N = sum(x_sol) + floor(n / 2)
    sblmo = Boscia.UnitSimplexSimpleBLMO(N)

    x, _, result = Boscia.solve(f, grad!, sblmo, fill(0.0, n), fill(N, n), collect(1:n), n)

    @test sum(isapprox.(x, x_sol, atol=1e-6, rtol=1e-2)) == n
    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
end

n = 20
x_sol = rand(1:floor(Int, n / 4), n)
diffi = x_sol + 0.3 * rand([-1, 1], n)

@testset "Reverse Knapsack LMO" begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    N = sum(x_sol) - floor(n / 2)
    sblmo = Boscia.ReverseKnapsackBLMO(n, N=N, upper=N)

    x, _, result = Boscia.solve(f, grad!, sblmo, fill(0.0, n), fill(N, n), collect(1:n), n)

    @test sum(isapprox.(x, x_sol, atol=1e-6, rtol=1e-2)) == n
    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
end
