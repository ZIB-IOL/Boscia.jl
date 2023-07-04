using Boscia
using FrankWolfe
using DataFrames
using CSV
using Test
using Random
using HiGHS
using SCIP
# using Statistics
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface

# The example from  "Optimizing a low-dimensional convex function over a high-dimensional cube"
# by Christoph Hunkenschröder, Sebastian Pokutta, Robert Weismantel
# https://arxiv.org/abs/2204.05266. 

example = "low_dim_high_dim"

seed=2
Random.seed!(seed)


m = 400 # larger dimension
n = 20 # small dimension

alpha = 0.00
const refpoint = 0.5 * ones(n) + Random.rand(n)* alpha * 1/n
W = rand(m,n)
const Ws = transpose(W) * W

function f(x)
    return 0.5 * (dot(x, Ws, x) - dot(refpoint, Ws, x) - dot(x, Ws, refpoint))
end

function grad!(storage, x)
    mul!(storage, Ws, (x - refpoint))
end

o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
MOI.empty!(o)
x = MOI.add_variables(o, n)
for xi in x
    MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
    MOI.add_constraint(o, xi, MOI.LessThan(1.0))
    MOI.add_constraint(o, xi, MOI.ZeroOne())
end
lmo = FrankWolfe.MathOptLMO(o)

min_number_lower = Inf
dual_gap_decay_factor = 0.8
#=Boscia.solve(f, grad!, lmo; verbose=true, min_number_lower=min_number_lower, fw_epsilon = 1e-3, print_iter=10)
data = @timed _, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, min_number_lower=min_number_lower, fw_epsilon = 1e-3, print_iter=10)
df = DataFrame(seed=seed, dimension=n, min_number_lower=min_number_lower, adaptive_gap=dual_gap_decay_factor, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size], node_level = result[:node_level])
file_name = "csv/dual_gap_" * example * "_" * string(n) * "_" * string(seed) * "_" * "most_infeasible" *".csv"
CSV.write(file_name, df, append=false)


# Strong branching
branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, HiGHS.Optimizer())
MOI.set(branching_strategy.optimizer, MOI.Silent(), true)

Boscia.solve(f, grad!, lmo; verbose=true, min_number_lower=min_number_lower, fw_epsilon = 1e-3, print_iter=10, branching_strategy = branching_strategy)
data = @timed _, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, min_number_lower=min_number_lower, fw_epsilon = 1e-3, print_iter=10, branching_strategy = branching_strategy)
df = DataFrame(seed=seed, dimension=n, min_number_lower=min_number_lower, adaptive_gap=dual_gap_decay_factor, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size], node_level = result[:node_level])
file_name = "csv/dual_gap_" * example * "_" * string(n) * "_" * string(seed) * "_" * "strong_branching" *".csv"
CSV.write(file_name, df, append=false)

# Hybrid strong branching
function perform_strong_branch(tree, node)
    node.level <= length(tree.root.problem.integer_variables)/4 ? println("Strong") : println("most infeasible")
    return node.level <= length(tree.root.problem.integer_variables)/4
end
branching_strategy = Boscia.HybridStrongBranching(10, 1e-3, HiGHS.Optimizer(), perform_strong_branch)
MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)

Boscia.solve(f, grad!, lmo; verbose=true, min_number_lower=min_number_lower, fw_epsilon = 1e-3, print_iter=10, branching_strategy = branching_strategy)
data = @timed _, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, min_number_lower=min_number_lower, fw_epsilon = 1e-3, print_iter=10, branching_strategy = branching_strategy)
df = DataFrame(seed=seed, dimension=n, min_number_lower=min_number_lower, adaptive_gap=dual_gap_decay_factor, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size], node_level = result[:node_level])
file_name = "csv/dual_gap_" * example * "_" * string(n) * "_" * string(seed) * "_" * "hybrid_strong_branching" *".csv"
CSV.write(file_name, df, append=false) =#


## Different strong branching

iterations = [5, 10, 50, 100]
iter = 2
for (index,value) in enumerate(iterations)
    for i in 0:iter
        max_iter = value
        branching_strategy = Boscia.PartialStrongBranching(max_iter, 10^i*1e-3, HiGHS.Optimizer())
        MOI.set(branching_strategy.optimizer, MOI.Silent(), true)

        Boscia.solve(f, grad!, lmo; verbose=true, branching_strategy = branching_strategy, dual_gap_decay_factor=dual_gap_decay_factor, min_number_lower=min_number_lower, fw_epsilon = 1e-4, print_iter=10, time_limit=10)
        data = @timed _, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, branching_strategy = branching_strategy, dual_gap_decay_factor=dual_gap_decay_factor, min_number_lower=min_number_lower, fw_epsilon = 1e-4, print_iter=10)
        df = DataFrame(seed=seed, dimension=n, min_number_lower=min_number_lower, adaptive_gap=dual_gap_decay_factor, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size], node_level = result[:node_level])
        file_name = "csv/strong_branching_" * example * "_" * string(n) * "_" * string(seed) * "_" * string(max_iter) * "1e-"*string(3-i)*".csv"
        CSV.write(file_name, df, append=false)
    end
end 


## Different hybrid branching

#=for i in [1, 2, 4, 10]
    function perform_strong_branch(tree, node)
        node.level <= length(tree.root.problem.integer_variables)/i ? println("Strong") : println("most infeasible")
        return node.level <= length(tree.root.problem.integer_variables)/i
    end
    branching_strategy = Boscia.HybridStrongBranching(5, 1e-3, HiGHS.Optimizer(), perform_strong_branch)
    MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)

    Boscia.solve(f, grad!, lmo; verbose=true, branching_strategy = branching_strategy, dual_gap_decay_factor=dual_gap_decay_factor, min_number_lower=min_number_lower, fw_epsilon = 1e-4, print_iter=10)
    data = @timed _, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, branching_strategy = branching_strategy, dual_gap_decay_factor=dual_gap_decay_factor, min_number_lower=min_number_lower, fw_epsilon = 1e-4, print_iter=10)
    df = DataFrame(seed=seed, dimension=n, min_number_lower=min_number_lower, adaptive_gap=dual_gap_decay_factor, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size], node_level = result[:node_level])
    file_name = "csv/hybrid_branching_" * example * "_" * string(n) * "_" * string(seed) * "_num_integer_dividedby" * string(i) * ".csv"
    CSV.write(file_name, df, append=false)
end =#
