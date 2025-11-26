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

seed = 0x8df5dbb59cf49249
@show seed
rng = StableRNG(seed)

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

    sblmo = Boscia.KNormBallLMO(K, τ)

    x, _, result = Boscia.solve(f, grad!, sblmo, fill(0.0, n), fill(Inf64, n), collect(1:n), n)
    @test sum(isapprox.(x, x_sol, atol=1e-6, rtol=1e-2)) == n
    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
end