using BranchWolfe
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface

n = 15
const ri = 10 * rand(n)
const ai = rand(n)
const Ωi = 3 * rand(Float64)
const bi = sum(ai)
Ai = randn(n,n)
Ai = Ai' * Ai
const Mi =  (Ai + Ai')/2
@assert isposdef(Mi)

@testset "Buchheim example" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o,n)
    I = collect(1:n) #rand(1:n0, Int64(floor(n0/2)))
    for i in 1:n
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        if i in I
            MOI.add_constraint(o, x[i], MOI.Integer())
        end
    end 
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai,x), 0.0), MOI.LessThan(bi))
    #MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai,x), 0.0), MOI.GreaterThan(minimum(ai)))
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n),x), 0.0), MOI.GreaterThan(1.0))
    lmo = FrankWolfe.MathOptLMO(o)

    function h(x)
        return Ωi
    end
    function f(x)
        return h(x) * (x' * Mi * x) - ri' * x
    end
    function grad!(storage, x)
        storage.= 2 * Mi * x - ri
        return storage
    end

    x, _,_,_ = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = true)
    @show x
    @test sum(ai'* x) <= bi + eps()
end
