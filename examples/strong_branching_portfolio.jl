using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface
import HiGHS

n = 20
const ri = 2 * rand(n)
const ai = rand(n)
const Ωi = 3 * rand(Float64)
const bi = sum(ai)
Ai = randn(n,n)
Ai = Ai' * Ai
const Mi =  (Ai + Ai')/2
@assert isposdef(Mi)

function prepare_portfolio_lmo()
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o,n)
    I = 1:n
    for i in 1:n
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        if i in I
            MOI.add_constraint(o, x[i], MOI.Integer())
        end
    end
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai,x), 0.0), MOI.LessThan(bi))
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n),x), 0.0), MOI.GreaterThan(1.0))
    lmo = FrankWolfe.MathOptLMO(o)
    return lmo
end

lmo = prepare_portfolio_lmo()

function f(x)
    return 1/2 * Ωi * dot(x, Mi, x) - dot(ri, x)
end
function grad!(storage, x)
    mul!(storage, Mi, x, Ωi, 0)
    storage .-= ri
    return storage
end

x, _, result_baseline = Boscia.solve(f, grad!, lmo, verbose = true, )
@test dot(ai, x) <= bi + eps()

branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, HiGHS.Optimizer())
MOI.set(branching_strategy.optimizer, MOI.Silent(), true)

lmo = prepare_portfolio_lmo()
x, _, result_strong_branching = Boscia.solve(f, grad!, lmo, verbose = true, branching_strategy=branching_strategy)

plot(result_baseline[:list_time],result_baseline[:list_ub], label="BL"); plot!(result_baseline[:list_time],result_baseline[:list_lb], label="BL")
plot!(result_strong_branching[:list_time], result_strong_branching[:list_ub], label="SB"); plot!(result_strong_branching[:list_time], result_strong_branching[:list_lb], label="SB")

plot(result_baseline[:list_ub], label="BL")
plot!(result_baseline[:list_lb], label="BL")
plot!(result_strong_branching[:list_ub], label="SB")
plot!(result_strong_branching[:list_lb], label="SB")
