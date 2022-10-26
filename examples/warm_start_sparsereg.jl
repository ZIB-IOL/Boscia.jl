using Statistics
using Boscia
using FrankWolfe
using Random
using LinearAlgebra
using SCIP
import Bonobo
import MathOptInterface
const MOI = MathOptInterface
using Dates
using Printf
using Test

using Plots

# For bug hunting:
# seed = rand(UInt64)
seed = 0x190b68d57cdf7d56
@show seed
Random.seed!(seed)

n = 80
const ri = rand(n)
const ai = rand(n)
const Ωi = rand(Float64)
const bi = sum(ai)
Ai = randn(n, n)
Ai = Ai' * Ai
const Mi = (Ai + Ai') / 2
@assert isposdef(Mi)


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
    return 1 / 2 * Ωi * dot(x, Mi, x) - dot(ri, x)
end
function grad!(storage, x)
    mul!(storage, Mi, x, Ωi, 0)
    storage .-= ri
    return storage
end

_, _, result_baseline = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=10)

_, _, result_baseline = Boscia.solve(f, grad!, lmo, verbose=true)
_, _, result_afw = Boscia.solve(f, grad!, lmo, verbose=true, afw=true)
# _, _, result_no_active_set = Boscia.solve(f, grad!, lmo, verbose=true, warmstart_active_set=false, warmstart_shadow_set=true)
# _, _, result_no_shadow = Boscia.solve(f, grad!, lmo, verbose=true, warmstart_active_set=true, warmstart_shadow_set=false)
# _, _, result_no_warmstart = Boscia.solve(f, grad!, lmo, verbose=true, warmstart_active_set=false, warmstart_shadow_set=false)

using JSON
open("results_portfolio_afw_" * string(seed) * ".json", "w") do f
    write(
        f,
        # JSON.json((; result_baseline, result_no_warmstart, result_no_active_set, result_no_shadow))
        JSON.json((; result_baseline, result_afw))
    )
end

# plot(result_baseline[:list_time],result_baseline[:list_ub], label="BL")
# plot!(result_no_active_set[:list_time], result_no_active_set[:list_ub], label="no-a")
# plot!(result_no_shadow[:list_time], result_no_shadow[:list_ub], label="no-s")
# plot!(result_no_warmstart[:list_time], result_no_warmstart[:list_ub], label="no-s")

# plot(result_baseline[:list_time],result_baseline[:list_lb], label="BL", legend=:bottomright, style=:dash)
# plot!(result_no_active_set[:list_time], result_no_active_set[:list_lb], label="NA")
# plot!(result_no_shadow[:list_time], result_no_shadow[:list_lb], label="NS")
# plot!(result_no_warmstart[:list_time], result_no_warmstart[:list_lb], label="NW")

# plot!(result_no_active_set[:list_time], result_no_active_set[:list_ub], label="no-a")
# plot!(result_no_shadow[:list_time], result_no_shadow[:list_ub], label="no-s")
# plot!(result_no_warmstart[:list_time], result_no_warmstart[:list_ub], label="no-s")
# plot(result_baseline[:list_lmo_calls_acc],result_baseline[:list_lb], label="BL", legend=:bottomright)
# plot!(result_no_active_set[:list_lmo_calls_acc], result_no_active_set[:list_lb], label="NA")
# plot!(result_no_shadow[:list_lmo_calls_acc], result_no_shadow[:list_lb], label="NS")
# plot!(result_no_warmstart[:list_lmo_calls_acc], result_no_warmstart[:list_lb], label="NW")