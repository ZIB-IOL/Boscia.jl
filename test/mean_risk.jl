using Statistics
using Random
using Distributions
using LinearAlgebra
import HiGHS
import SCIP
import MathOptInterface
const MOI = MathOptInterface
import Boscia
import Bonobo
using FrankWolfe
using Dates
using Test
using StableRNGs

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

# min h(sqrt(y' * M * y)) - r' * y
# s.t. a' * y <= b 
#           y >= 0
#           y_i in Z for i in I

n0 = 30
const r = 10 * rand(rng, n0)
const a = rand(rng, n0)
const Ω = 3 * rand(rng, Float64)
const b = sum(a)
A1 = randn(rng, n0, n0)
A1 = A1' * A1
const M1 = (A1 + A1') / 2
@assert isposdef(M1)


@testset "Buchheim et. al. mean risk" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n0)
    I = collect(1:n0) #rand(rng, 1:n0, Int64(floor(n0/2)))
    for i in 1:n0
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        if i in I
            MOI.add_constraint(o, x[i], MOI.Integer())
        end
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(a, x), 0.0),
        MOI.LessThan(b),
    )
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n0), x), 0.0),
        MOI.GreaterThan(1.0),
    )
    lmo = FrankWolfe.MathOptLMO(o)

    function f(x)
        return Ω * (x' * M1 * x) - r' * x
    end
    function grad!(storage, x)
        storage .= 2 * Ω * M1 * x - r
        return storage
    end

    x, _, result =
        Boscia.solve(f, grad!, lmo, settings_bnb=Boscia.settings_bnb(verbose=true, time_limit=300))

    @test dot(a, x) <= b + 1e-2
    @test f(x) <= f(result[:raw_solution]) + 1e-6
end
