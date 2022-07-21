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


n = 20
diffi = Random.rand(Bool,n)*0.6.+0.3

@testset "Interface - norm hyperbox" begin
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

    function f(x)
        return sum(0.5*(x.-diffi).^2)
    end
    function grad!(storage, x)
        @. storage = x-diffi
    end

    x, _ = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = true)

    @test x == round.(diffi)
end
