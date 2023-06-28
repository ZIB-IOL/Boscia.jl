using Boscia
using FrankWolfe
using Random
using SCIP
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
using DataFrames
using CSV

example = "birkhoff"
seed=3
Random.seed!(seed)

# Example on the Birkhoff polytope but using permutation matrices directly
# https://arxiv.org/pdf/2011.02752.pdf
# https://www.sciencedirect.com/science/article/pii/S0024379516001257


# min_{X, θ} 1/2 * || ∑_{i in [k]} θ_i X_i - Xhat ||^2
# θ ∈ Δ_k (simplex)
# X_i ∈ P_n (permutation matrix)

# we linearize the bilinear terms in the objective
# min_{X, Y, θ} 1/2 ||∑_{i in [k]} Y - Xhat ||^2
# θ ∈ Δ_k (simplex)
# X_i ∈ P_n (permutation matrix)
# 0 ≤ Y_i ≤ X_i
# 0 ≤ θ_i - Y_i ≤ 1 - X_i

# The variables are ordered (Y, X, theta) in the MOI model
# the objective only uses the last n^2 variables
# Small dimensions since the size of the problem grows quickly (2 k n^2 + k variables)
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
        s += 0.5 * (sum(x[(j-1) * n^2 + i] for j in 1:k) - Xstar[i])^2
    end
    return s
end

# note: reshape gives a reference to the same data, so this is updating storage in-place
function grad!(storage, x)
    storage .= 0
    for j in 1:k
        Sk = reshape(@view(storage[(j-1) * n^2 + 1 : j * n^2]), n, n)
        @. Sk = - Xstar
        for m in 1:k
            Yk = reshape(@view(x[(m-1) * n^2 + 1 : m * n^2]), n, n)
            @. Sk += Yk
        end
    end
    storage
end

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
        o, vec(sum(X[i], dims=1, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
        MOI.EqualTo(1.0),
    )
    MOI.add_constraint.(
        o, vec(sum(X[i], dims=1, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
        MOI.EqualTo(1.0),
    )
    # 0 ≤ Y_i ≤ X_i
    MOI.add_constraint.(
        o, 1.0 * Y[i] - X[i],
        MOI.LessThan(0.0),
    )
    # 0 ≤ θ_i - Y_i ≤ 1 - X_i
    MOI.add_constraint.(
        o, 1.0 * theta[i] .- Y[i] .+ X[i],
        MOI.LessThan(1.0),
    )
end
MOI.add_constraint(o, sum(theta, init=0.0), MOI.EqualTo(1.0))
lmo = FrankWolfe.MathOptLMO(o)

# min_number_lower = Inf
# dual_gap_decay_factor = 0.7
# fw_epsilon = 1e-3
# x, _, result= Boscia.solve(f, grad!, lmo; verbose=true, dual_gap_decay_factor=0.7, min_number_lower=Inf, fw_epsilon = fw_epsilon, print_iter=1) 

# df = DataFrame(lmo_calls=result[:lmo_calls_per_layer], active_set_size=result[:active_set_size_per_layer], discarded_set_size=result[:discarded_set_size_per_layer])
# file_name = "experiments/csv/" * example * "_per_layer_" * string(n) * "_" * string(k) * "_" * string(seed) * "_" * string(min_number_lower) * "_" * string(dual_gap_decay_factor) * "_" * string(fw_epsilon) * ".csv"
# CSV.write(file_name, df, append=false)

Boscia.solve(f, grad!, lmo; verbose=true)

values = [0.9, 1.0]
fw_epsilon_values = [1e-3, 5e-3, 1e-4, 1e-7]
min_num_lower_values = [20, 40, 60, 80, 100, 200, Inf]
seeds = [1]#,2,3]

iter = 1#3

for (seed_idx, seed_val) in enumerate(seeds)
    for (index,value) in enumerate(values)
        for i in 1:iter
            for (idx,eps) in enumerate(fw_epsilon_values)
                for (idx2, min_num_lower_val) in enumerate(min_num_lower_values)
                    dual_gap_decay_factor = value
                    min_number_lower = min_num_lower_val
                    fw_epsilon = eps
                    seed = seed_val
                    data = @timed _, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, dual_gap_decay_factor=dual_gap_decay_factor, min_number_lower=min_number_lower, fw_epsilon = fw_epsilon, print_iter=1)
                    df = DataFrame(seed=seed, dimension=n, min_number_lower=min_number_lower, adaptive_gap=dual_gap_decay_factor, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size])
                    file_name = "experiments/csv/early_stopping_" * example * "_" * string(n) * "_" * string(k) * "_" * string(seed) * "_" * string(min_number_lower) * "_" * string(dual_gap_decay_factor) * "_" * string(fw_epsilon) * "_" * string(i) *".csv"
                    CSV.write(file_name, df, append=false)
                end
            end
        end
    end
end 
