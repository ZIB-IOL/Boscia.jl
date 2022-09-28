using Test
using Boscia
using FrankWolfe
using Random
using SCIP
import MathOptInterface
using LinearAlgebra
using Distributions
import Bonobo
using Printf
using Dates
const MOI = MathOptInterface
const MOIU = MOI.Utilities

import MathOptSetDistances
const MOD = MathOptSetDistances

n = 15
const ri = rand(n)
const ai = rand(n)
const Ωi = rand(Float64)
const bi = sum(ai)
Ai = randn(n, n)
Ai = Ai' * Ai
const Mi = (Ai + Ai') / 2
@assert isposdef(Mi)
time_limit = 30.0

@testset "Time limit test" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    I = collect(1:n) #rand(1:n0, Int64(floor(n0/2)))
    for i in 1:n
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        if i in I
            MOI.add_constraint(o, x[i], MOI.Integer())
        end
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai, x), 0.0),
        MOI.LessThan(bi),
    )
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.GreaterThan(1.0),
    )
    lmo = FrankWolfe.MathOptLMO(o)

    function f(x)
        return Ωi * (x' * Mi * x) - ri' * x
    end
    function grad!(storage, x)
        storage .= 2 * Mi * x - ri
        return storage
    end

    start_time = Dates.now()
    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=time_limit)
    time_taken = float(Dates.value(Dates.now() - start_time)) / 1000
    # @show x
    @test sum(ai' * x) <= bi + 1e-6
    @test f(x) <= f(result[:raw_solution]) + 1e-6
    @test time_taken <= time_limit + 10
end
