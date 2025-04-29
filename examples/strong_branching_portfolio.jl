using Boscia
using FrankWolfe
using Test
using Random
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface
import HiGHS
using SCIP
using StableRNGs

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

# For bug hunting:
#seed = rand(UInt64)
#@show seed
#Random.seed!(seed)

# TROUBLESOME SEED seed = 0x8750860d6fd5025f -> NEEDS TO BE CHECK AGAIN!

n = 20
const ri = rand(rng, n)
const ai = rand(rng, n)
const Ωi = rand(rng, Float64)
const bi = sum(ai)
Ai = randn(rng, n, n)
Ai = Ai' * Ai
const Mi = (Ai + Ai') / 2
@assert isposdef(Mi)

function prepare_portfolio_lmo()
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
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.GreaterThan(1.0),
    )
    lmo = FrankWolfe.MathOptLMO(o)
    return lmo
end

function f(x)
    return 1 / 2 * Ωi * dot(x, Mi, x) - dot(ri, x)
end
function grad!(storage, x)
    mul!(storage, Mi, x, Ωi, 0)
    storage .-= ri
    return storage
end

@testset "Portfolio strong branching" begin
    lmo = prepare_portfolio_lmo()
    x, _, result_baseline = Boscia.solve(f, grad!, lmo, verbose=true)
    @test dot(ai, x) <= bi + 1e-6
    @test f(x) <= f(result_baseline[:raw_solution]) + 1e-6

    blmo = Boscia.MathOptBLMO(HiGHS.Optimizer())
    branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
    MOI.set(branching_strategy.bounded_lmo.o, MOI.Silent(), true)

    lmo = prepare_portfolio_lmo()
    x, _, result_strong_branching =
        Boscia.solve(f, grad!, lmo, verbose=true, branching_strategy=branching_strategy)

    @test dot(ai, x) <= bi + 1e-3
    @test f(x) <= f(result_baseline[:raw_solution]) + 1e-6
end

#plot(result_baseline[:list_time],result_baseline[:list_ub], label="BL"); plot!(result_baseline[:list_time],result_baseline[:list_lb], label="BL")
#plot!(result_strong_branching[:list_time], result_strong_branching[:list_ub], label="SB"); plot!(result_strong_branching[:list_time], result_strong_branching[:list_lb], label="SB")

#plot(result_baseline[:list_ub], label="BL")
#plot!(result_baseline[:list_lb], label="BL")
#plot!(result_strong_branching[:list_ub], label="SB")
#plot!(result_strong_branching[:list_lb], label="SB")
