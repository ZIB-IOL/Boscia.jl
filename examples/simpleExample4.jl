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

# This example is good example as all the gradient at each 0/1 point "look" the same and any 0/1 solution (if alpha = 0) is optimal 
# but you need to prove it
# if alpha >0 you need to find the optimal one

n = 100
alpha = 0.00

diffi = 0.5 * ones(n) + Random.rand(n)* alpha * 1/n

@testset "Interface - norm hyperbox" begin
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

    function f(x)
        return sum(0.5*(x.-diffi).^2)
    end

    function grad!(storage, x)
        @. storage = x-diffi
    end

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
