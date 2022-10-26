using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
import HiGHS

using Plots

# For bug hunting:
seed = rand(UInt64)
@show seed
Random.seed!(seed)

n = 3
k = 3

# generate random doubly stochastic matrix
const Xstar = rand(n, n)
while norm(sum(Xstar, dims=1) .- 1) > 1e-6 || norm(sum(Xstar, dims=2) .- 1) > 1e-6
    Xstar ./= sum(Xstar, dims=1)
    Xstar ./= sum(Xstar, dims=2)
end

function f(x)
    s = zero(eltype(x))
    for i in eachindex(Xstar)
        s += 0.5 * (sum(x[(j-1)*n^2+i] for j in 1:k) - Xstar[i])^2
    end
    return s
end

# note: reshape gives a reference to the same data, so this is updating storage in-place
function grad!(storage, x)
    storage .= 0
    for j in 1:k
        Sk = reshape(@view(storage[(j-1)*n^2+1:j*n^2]), n, n)
        @. Sk = -Xstar
        for m in 1:k
            Yk = reshape(@view(x[(m-1)*n^2+1:m*n^2]), n, n)
            @. Sk += Yk
        end
    end
    return storage
end

function build_birkhoff_lmo()
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    Y = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
    X = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
    theta = MOI.add_variables(o, k)

    for i in 1:k
        MOI.add_constraint.(o, Y[i], MOI.GreaterThan(0.0))
        MOI.add_constraint.(o, Y[i], MOI.LessThan(1.0))
        MOI.add_constraint.(o, X[i], MOI.ZeroOne())
        MOI.add_constraint(o, theta[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, theta[i], MOI.LessThan(1.0))
        # doubly stochastic constraints
        MOI.add_constraint.(
            o,
            vec(sum(X[i], dims=1, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
            MOI.EqualTo(1.0),
        )
        MOI.add_constraint.(
            o,
            vec(sum(X[i], dims=1, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
            MOI.EqualTo(1.0),
        )
        # 0 ≤ Y_i ≤ X_i
        MOI.add_constraint.(o, 1.0 * Y[i] - X[i], MOI.LessThan(0.0))
        # 0 ≤ θ_i - Y_i ≤ 1 - X_i
        MOI.add_constraint.(o, 1.0 * theta[i] .- Y[i] .+ X[i], MOI.LessThan(1.0))
    end
    MOI.add_constraint(o, sum(theta, init=0.0), MOI.EqualTo(1.0))
    return FrankWolfe.MathOptLMO(o)
end

lmo = build_birkhoff_lmo()
_, _, result_baseline = Boscia.solve(f, grad!, lmo, verbose=true)X
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

plot(result_baseline[:list_ub], label="BL")
plot!(result_baseline[:list_lb], label="BL")
plot!(result_strong_branching[:list_ub], label="SB")
plot!(result_strong_branching[:list_lb], label="SB")
