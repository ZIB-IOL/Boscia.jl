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
    I = collect(1:n)
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
    lmo = FrankWolfe.MathOptLMO(o)

    function f(x)
        return 1 / 2 * Ωi * dot(x, Mi, x) - dot(ri, x)
    end
    function grad!(storage, x)
        mul!(storage, Mi, x, Ωi, 0)
        storage .-= ri
        return storage
    end

    start_time = Dates.now()
    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=time_limit)
    time_taken = float(Dates.value(Dates.now() - start_time)) / 1000
    @test sum(ai' * x) <= bi + 1e-3
    @test f(x) <= f(result[:raw_solution]) + 1e-6
    @test result[:total_time_in_sec] <= time_limit + 5
end
