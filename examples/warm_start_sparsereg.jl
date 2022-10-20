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
seed = rand(UInt64)
@show seed
Random.seed!(seed)

const n0 = 20;
const p = 5 * n0;
const k = ceil(n0 / 5);
const lambda_0 = rand(Float64);
const lambda_2 = 10.0 * rand(Float64);
const A = rand(Float64, n0, p)
const y = rand(Float64, n0)
const M = 2 * var(A)

# "Sparse Regression" 

function f(x)
    xv = @view(x[1:p])
    return norm(y - A * xv)^2 + lambda_0 * sum(@view(x[p+1:2p])) + lambda_2 * norm(xv)^2
end

function grad!(storage, x)
    storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
    storage[p+1:2p] .= lambda_0
    return storage
end

o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
MOI.empty!(o)
x = MOI.add_variables(o, 2p)
for i in p+1:2p
    MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
    MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
    MOI.add_constraint(o, x[i], MOI.ZeroOne()) # or MOI.Integer()
end
for i in 1:p
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, M], [x[i], x[i+p]]), 0.0),
        MOI.GreaterThan(0.0),
    )
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -M], [x[i], x[i+p]]), 0.0),
        MOI.LessThan(0.0),
    )
end
MOI.add_constraint(
    o,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p), x[p+1:2p]), 0.0),
    MOI.LessThan(k),
)
lmo = FrankWolfe.MathOptLMO(o)

_, _, result_baseline = Boscia.solve(f, grad!, lmo, verbose=true)
_, _, result_no_active_set = Boscia.solve(f, grad!, lmo, verbose=true, warmstart_active_set=false, warmstart_shadow_set=true)
_, _, result_no_shadow = Boscia.solve(f, grad!, lmo, verbose=true, warmstart_active_set=true, warmstart_shadow_set=false)
_, _, result_no_warmstart = Boscia.solve(f, grad!, lmo, verbose=true, warmstart_active_set=false, warmstart_shadow_set=false)

plot(result_baseline[:list_time],result_baseline[:list_ub], label="BL")
plot!(result_baseline[:list_time],result_baseline[:list_lb], label="BL")
plot!(result_no_active_set[:list_time], result_no_active_set[:list_ub], label="no-a")
plot!(result_no_active_set[:list_time], result_no_active_set[:list_lb], label="no-a")
plot!(result_no_shadow[:list_time], result_no_shadow[:list_ub], label="no-s")
plot!(result_no_shadow[:list_time], result_no_shadow[:list_lb], label="no-s")
plot!(result_no_warmstart[:list_time], result_no_warmstart[:list_ub], label="no-s")
plot!(result_no_warmstart[:list_time], result_no_warmstart[:list_lb], label="no-s")
