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

m = 20 # larger dimension
n = 9 # small dimension

alpha = 0.00
diffi = 0.5 * ones(n) + Random.rand(n)* alpha * 1/n

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

    function f(x)
        return sum(0.5*((W*(x.-diffi))).^2)
    end

    function grad!(storage, x)
        @. storage = (x - diffi)*W
    end

    x = rand(n)
    print("test: ", f(x))

    stoc = zeros(m)
    grad!(stoc, x)
    print("test 2: ", stoc)

    x, _ = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = true)

    # build optimal solution
    xopt = zeros(n)
    for i in 1:n
        if diffi[i] > 0.5
            xopt[i] = 1
        end
    end

    @test f(x) == f(xopt)
end
