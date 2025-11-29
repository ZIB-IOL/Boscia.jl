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
using LinearAlgebra
const MOI = MathOptInterface
const MOIU = MOI.Utilities
using StableRNGs

println("\nHeuristics Tests")

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

n = 20
x_sol = rand(rng, 1:floor(Int, n / 4), n)
N = sum(x_sol)
dir = vcat(fill(1, floor(Int, n / 2)), fill(-1, floor(Int, n / 2)), fill(0, mod(n, 2)))
diffi = x_sol + 0.3 * dir

@testset "Hyperplane Aware Rounding - Probability Simplex" begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    sblmo = Boscia.ProbabilitySimplexSimpleBLMO(N)

    settings = Boscia.create_default_settings()
    settings.heuristic[:hyperplane_aware_rounding_prob] = 0.8
    x, _, result = Boscia.solve(
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
end

n = 20
x_sol = rand(rng, 1:floor(Int, n / 4), n)
diffi = x_sol + 0.3 * rand(rng, [-1, 1], n)

@testset "Hyperplane Aware Rounding - Unit Simplex" begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    N = sum(x_sol) + floor(n / 2)
    sblmo = Boscia.UnitSimplexSimpleBLMO(N)

    settings = Boscia.create_default_settings()
    settings.heuristic[:hyperplane_aware_rounding_prob] = 0.8
    x, _, result = Boscia.solve(
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
end

@testset "Following Gradient Heuristic - Unit Simplex" begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    N = sum(x_sol) + floor(n / 2)
    sblmo = Boscia.UnitSimplexSimpleBLMO(N)
    depth = 5

    settings = Boscia.create_default_settings()
    settings.heuristic[:follow_gradient_prob] = 1.0
    settings.heuristic[:follow_gradient_steps] = depth
    x_heu, _, result_heu = Boscia.solve(
        f,
        grad!,
        sblmo,
        fill(0.0, n),
        fill(1.0 * N, n),
        collect(1:n),
        n,
        settings=settings,
    )

    x, _, result = Boscia.solve(f, grad!, sblmo, fill(0.0, n), fill(1.0 * N, n), collect(1:n), n)

    @test sum(isapprox.(x, x_sol, atol=1e-6, rtol=1e-2)) == n
    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)

    @test sum(isapprox.(x_heu, x_sol, atol=1e-6, rtol=1e-2)) == n
    @test isapprox(f(x_heu), f(result_heu[:raw_solution]), atol=1e-6, rtol=1e-3)

    @test result[:lmo_calls] == result_heu[:lmo_calls]
    @test result_heu[:heu_lmo_calls] > 0
end

@testset "Rounding Heuristic - Unit Simplex" begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    N = sum(x_sol) + floor(n / 2)
    sblmo = Boscia.UnitSimplexSimpleBLMO(N)

    x_always, _, result_always =
        Boscia.solve(f, grad!, sblmo, fill(0.0, n), fill(1.0 * N, n), collect(1:n), n)

    sblmo = Boscia.UnitSimplexSimpleBLMO(N)
    settings = Boscia.create_default_settings()
    settings.heuristic[:rounding_prob] = 0.5
    x, _, result = Boscia.solve(
        f,
        grad!,
        sblmo,
        fill(0.0, n),
        fill(1.0 * N, n),
        collect(1:n),
        n,
        settings=settings,
    )

    @test sum(isapprox.(x_always, x_sol, atol=1e-6, rtol=1e-2)) == n
    @test sum(isapprox.(x, x_sol, atol=1e-6, rtol=1e-2)) == n
    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
end

n = 30
diffi = rand(rng, Bool, n) * 0.6 .+ 0.3

@testset "Probability Rounding - Unit Cube" begin

    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    lbs = zeros(n)
    ubs = ones(n)
    int_vars = unique!(rand(rng, 1:n, floor(Int, n / 2)))
    x_sol = copy(diffi)
    x_sol[int_vars] = round.(x_sol[int_vars])

    sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)

    settings = Boscia.create_default_settings()
    settings.heuristic[:probability_rounding_prob] = 0.6
    settings.heuristic[:rounding_prob] = 0.0
    x, _, result =
        Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, settings=settings)

    @test sum(isapprox.(x, x_sol, atol=1e-6, rtol=1e-2)) == n
    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
end

n = 20
x_sol = round.(rand(rng, n))
N = sum(x_sol)
dir = sign.(iszero.(x_sol) .- 0.5)
diffi = x_sol + 0.3 * dir

@testset "Probability rounding - Probability Simplex" begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    int_vars = unique!(rand(rng, 1:n, floor(Int, n / 2)))
    m = length(int_vars)
    cont_vars = setdiff(collect(1:n), int_vars)
    x_sol[cont_vars] = diffi[cont_vars]

    sblmo = Boscia.ProbabilitySimplexSimpleBLMO(N)

    settings = Boscia.create_default_settings()
    settings.heuristic[:probability_rounding_prob] = 0.6
    settings.heuristic[:rounding_prob] = 0.0
    x, _, result =
        Boscia.solve(f, grad!, sblmo, fill(0.0, m), fill(1.0, m), int_vars, n, settings=settings)

    @test f(x) ≥ f(x_sol)
    if isapprox(sum(x_sol), N)
        @test isapprox(f(x), f(x_sol))
    end
    @test sum(isapprox.(x[int_vars], x_sol[int_vars], atol=1e-6, rtol=1e-2)) == m
    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
end

n = 20
sparsity = 0.3
x_sol = [rand() < sparsity ? 0 : rand(1:floor(Int, n / 4)) for _ in 1:n]
diffi = x_sol + 0.3 * rand([-1, 1], n)

@testset "Rounding Heuristic - K sparse " begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    K = count(!iszero, x_sol)
    τ = 1.5 * norm(x_sol, Inf)

    sblmo = Boscia.KSparseBLMO(K, τ)

    x_always, _, result_always =
        Boscia.solve(f, grad!, sblmo, fill(0.0, n), fill(Inf64, n), collect(1:n), n)

    sblmo = Boscia.KSparseBLMO(K, τ)
    settings = Boscia.create_default_settings()
    settings.heuristic[:probability_rounding_prob] = 0.6
    settings.heuristic[:rounding_prob] = 0.0
    x, _, result = Boscia.solve(
        f,
        grad!,
        sblmo,
        fill(0.0, n),
        fill(Inf64, n),
        collect(1:n),
        n,
        settings=settings,
    )

    @test sum(isapprox.(x_always, x_sol, atol=1e-6, rtol=1e-2)) == n
    @test sum(isapprox.(x, x_sol, atol=1e-6, rtol=1e-2)) == n
    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
end
