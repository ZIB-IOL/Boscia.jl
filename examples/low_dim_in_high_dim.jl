using BranchWolfe
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
const refpoint = 0.5 * ones(n) + Random.rand(n)* alpha * 1/n

@testset "Low-dimensional function" begin
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

    W = rand(m,n)
    Ws = transpose(W) * W

    function f(x)
        return 0.5 * (dot(x, Ws, x) - dot(refpoint, Ws, x) - dot(x, Ws, refpoint))
    end

    function grad!(storage, x)
        mul!(storage, Ws, (x - refpoint))
    end

    x, _, result,_ = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = true)
    
    # @show result # solution statistics also as dict for further processing // too large for printing usually

    if n < 15  # only do for small n 
        valopt, xopt = BranchWolfe.min_via_enum(f,n)
        @test f(x) ≈ f(xopt)
    end

end
