using Boscia
using FrankWolfe
using Test
using Random
using SCIP
# using Statistics
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface

# The example from  "Optimizing a low-dimensional convex function over a high-dimensional cube"
# by Christoph Hunkenschröder, Sebastian Pokutta, Robert Weismantel
# https://arxiv.org/abs/2204.05266. 

m = 500 # larger dimension
n = 12 # small dimension

alpha = 0.00
const refpoint = 0.5 * ones(n) + Random.rand(n) * alpha * 1 / n
W = rand(m, n)
const Ws = transpose(W) * W

function f(x)
    return 0.5 * (dot(x, Ws, x) - dot(refpoint, Ws, x) - dot(x, Ws, refpoint))
end

function grad!(storage, x)
    return mul!(storage, Ws, (x - refpoint))
end

@testset "Low-dimensional function (SCIP)" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
        MOI.add_constraint(o, xi, MOI.ZeroOne())
    end
    lmo = FrankWolfe.MathOptLMO(o)

    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true)

    if n < 15  # only do for small n 
        valopt, xopt = Boscia.min_via_enum(f, n)
        @test (f(x) - f(xopt)) / abs(f(xopt)) <= 1e-3
    end

    @test f(x) <= f(result[:raw_solution]) + 1e-6
end

@testset "Low-dimensional function (CubeSimpleBLMO)" begin

    int_vars = collect(1:n)

    lbs = zeros(n)
    ubs = ones(n)
    
    sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
    
    # modified solve call from managed_blmo.jl automatically wraps sblmo into a managed_blmo
    x, _, result = Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, verbose=true)

    if n < 15  # only do for small n 
        valopt, xopt = Boscia.min_via_enum(f, n)
        @test (f(x) - f(xopt)) / abs(f(xopt)) <= 1e-3
    end

    @test f(x) <= f(result[:raw_solution]) + 1e-6
end
