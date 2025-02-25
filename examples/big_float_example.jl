# This script performs optimization tests using the Boscia package with high precision arithmetic
# (BigFloat and DoubleFloat) on a function that has integer constraints. The goal is to verify
# the correctness and performance of the optimization process using these high-precision types.
# It includes two test sets, one using BigFloat and another using DoubleFloat, and employs 
# custom heuristics for solving the optimization problem. The results are compared against 
# expected values and checked for accuracy using `@test` assertionsusing Random

# We first import the necessary packages, which will run the required test .


using Boscia
using Test
using DoubleFloats

n = 200
diffi = Random.rand(Bool, n) * 0.6 .+ 0.3

@testset "Using BigFloat" begin
    function f(x)
        x = BigFloat.(x)
        return float(0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x)))
    end
    function grad!(storage, x)
        x = BigFloat.(x)
        @. storage = x - diffi
        return float.(storage)
    end

    int_vars = collect(1:n)
    lbs = zeros(n)
    ubs = ones(n)

    sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
    custom_heuristics= [Boscia.Heuristic(Boscia.rounding_lmo_01_heuristic, 0.7, :rounding_lmo_01_heuristic),
    Boscia.Heuristic(Boscia.probability_rounding, 0.7, :probability_rounding)]

    x, _, result =
        Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, verbose=true, time_limit=120, custom_heuristics = custom_heuristics)

    if result[:total_time_in_sec] < 125
        @test x == round.(diffi)
    end
    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
end

@testset "Using DoubleFloat" begin
    function f(x)
        x = Double64.(x)
        return float(0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x)))
    end
    function grad!(storage, x)
        x = Double64.(x)
        @. storage = x - diffi
        return float.(storage)
    end

    int_vars = collect(1:n)
    lbs = zeros(n)
    ubs = ones(n)

    sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
    custom_heuristics= [Boscia.Heuristic(Boscia.rounding_lmo_01_heuristic, 0.7, :rounding_lmo_01_heuristic),
    Boscia.Heuristic(Boscia.probability_rounding, 0.7, :probability_rounding)]

    x, _, result =
        Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, verbose=true, time_limit=125, custom_heuristics = custom_heuristics)

    if result[:total_time_in_sec] < 125
        @test x == round.(diffi)
    end
    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
end
