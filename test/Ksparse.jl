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

#seed = 0x470d3b82ffb09cc7
seed = rand(UInt64)
@show seed
Random.seed!(seed)

n = 20
x_sol = rand(1:floor(Int, n / 4), n)
N = sum(x_sol)
dir = vcat(fill(1, floor(Int, n / 2)), fill(-1, floor(Int, n / 2)), fill(0, mod(n, 2)))
diffi = x_sol + 0.3 * dir

@testset "KSparse LMO" begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    K = count(!iszero, x_sol)
    τ = 1.5 * norm(x_sol, Inf) 
    
    sblmo = Boscia.KSparseBLMO(K, τ)

    x, _, result = Boscia.solve(f, grad!, sblmo, fill(0.0, n), fill(Inf64, n), collect(1:n), n)

    settings = Boscia.create_default_settings()
    settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()
    x_dicg, _, result_dicg = Boscia.solve(
        f,
        grad!,
        sblmo,
        fill(0.0, n),
        fill(Inf64, n),
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
x_sol = rand(1:floor(Int, n / 4), n)
diffi = x_sol + 0.3 * rand([-1, 1], n)

@testset "Diomond LMO" begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    lower_bounds = fill(-sum(x_sol), n)

    upper_bounds = fill(sum(x_sol), n)
    
    sblmo = Boscia.DiamondBLMO(lower_bounds, upper_bounds)

    x, _, result = Boscia.solve(f, grad!, sblmo, fill(0.0, n), fill(Inf64, n), collect(1:n), n)
    settings = Boscia.create_default_settings()
    settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()
    x_dicg, _, result_dicg = Boscia.solve(
        f,
        grad!,
        sblmo,
        fill(0.0, n),
        fill(Inf64, n),
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
sparsity = 0.3  
x_sol = [rand() < sparsity ? 0 : rand(1:floor(Int, n / 4)) for _ in 1:n]
diffi = x_sol + 0.3 * rand([-1, 1], n)

@testset "KNormBall LMO" begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    K = count(!iszero, x_sol)
    τ = 1.5 * norm(x_sol, Inf) * K
    println("$x_sol")

    sblmo = Boscia.KNormBallLMO(K, τ)

    x, _, result = Boscia.solve(f, grad!, sblmo, fill(0.0, n), fill(Inf64, n), collect(1:n), n)


    @test sum(isapprox.(x, x_sol, atol=1e-6, rtol=1e-2)) == n
    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
end