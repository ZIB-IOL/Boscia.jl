using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface

# For bug hunting:
seed = rand(UInt64)
seed = 0x8abe67301e43f00c 
@show seed
Random.seed!(seed)

# seed = 0x8abe67301e43f00c emptz tree after 1 node??? and incument >> lower bound


n = 15
const ri = rand(n)
const ai = rand(n)
const Ωi = rand(Float64)
const bi = sum(ai)
Ai = randn(n,n)
Ai = Ai' * Ai
const Mi =  (Ai + Ai')/2
@assert isposdef(Mi)

@show bi
@show ai[end] * 93.0 <= bi +1e-6

@testset "Buchheim et. al. example" begin
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
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n),x), 0.0), MOI.GreaterThan(1.0))
    lmo = FrankWolfe.MathOptLMO(o)

    function f(x)
        return 1/2 * Ωi * dot(x, Mi, x) - dot(ri, x)
    end
    function grad!(storage, x)
        mul!(storage, Mi, x)
        storage .-= ri
        return storage
    end

    x, _,result = Boscia.solve(f, grad!, lmo, verbose = true)
    @show x
    @show result[:raw_solution]
    @test dot(ai, x) <= bi + 1e-6
    @test f(x) <= f(result[:raw_solution]) + 1e-6
end


# seed = 0x946d4b7835e92ffa takes 90 minutes to solve!
