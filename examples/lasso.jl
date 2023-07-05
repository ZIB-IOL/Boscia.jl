using Statistics
using Random
using Distributions
using LinearAlgebra
using SCIP, HiGHS
using MathOptInterface
MOI = MathOptInterface
using FrankWolfe
using Boscia
using Bonobo
using DataFrames
using CSV


function build_lasso(dim, fac, seed, use_indicator, time_limit, rtol)
    example = "lasso"
    Random.seed!(seed)

    p = dim
    k = ceil(dim / fac)
    group_size = convert(Int64, floor(p / k))
    M = 5.0
    
    lambda_0_g = 0.0
    lambda_2_g = 0.0
    A_g = rand(Float64, dim, p)
    β_sol = rand(Distributions.Uniform(-M, M), p)
    k_int = convert(Int64, k)
    
    for i in 1:k_int
        for _ in 1:group_size-1
            β_sol[rand(((i-1)*group_size+1):(i*group_size))] = 0
        end
    end
    y_g = A_g * β_sol
    k = count(i -> i != 0, β_sol)
    
    groups = []
    for i in 1:(k_int-1)
        push!(groups, ((i-1)*group_size+1):(i*group_size))
    end
    push!(groups, ((k_int-1)*group_size+1):p)

    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, p)
    z = MOI.add_variables(o, p)
    for i in 1:p
        MOI.add_constraint(o, x[i], MOI.GreaterThan(-M))
        MOI.add_constraint(o, x[i], MOI.LessThan(M))

        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne()) 
    end 
    for i in 1:p
        if use_indicator 
            # Indicator: x[i+p] = 1 => x[i] = 0
            # Beware: SCIP can only handle MOI.ACTIVATE_ON_ONE with LessThan constraints.
            # Hence, in the indicator formulation, we ahve zi = 1 => xi = 0. (In bigM zi = 0 => xi = 0)
            gl = MOI.VectorAffineFunction(
                [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                    MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[i])),],
                [0.0, 0.0], )
            gg = MOI.VectorAffineFunction(
                [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                    MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, x[i])),],
                [0.0, 0.0], )
            MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(0.0)))
            MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(0.0)))
        else
            # big M
            MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,M], [x[i], z[i]]), 0.0), MOI.GreaterThan(0.0))
            MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-M], [x[i], z[i]]), 0.0), MOI.LessThan(0.0))
        end
        
    end
    if use_indicator
        MOI.add_constraint(o, sum(z, init=0.0), MOI.GreaterThan(1.0 * (p-k))) # we want less than k zeros
        for i in 1:k_int
            MOI.add_constraint(o, sum(z[groups[i]], init=0.0), MOI.LessThan(1.0 * group_size - 1))
        end
    else
        MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
        for i in 1:k_int
            MOI.add_constraint(o, sum(z[groups[i]], init=0.0), MOI.GreaterThan(1.0))
        end
    end
    
    lmo = FrankWolfe.MathOptLMO(o)


    function f(x)
        return (sum((y_g - A_g * x[1:p]) .^ 2) +
               lambda_0_g * sum(x[p+1:2p]) +
               lambda_2_g * FrankWolfe.norm(x[1:p])^2)/10000
    end
    function grad!(storage, x)
        storage .= vcat(
            2 * (transpose(A_g) * A_g * x[1:p] - transpose(A_g) * y_g + lambda_2_g * x[1:p]),
            lambda_0_g * ones(p),
        )./10000
        return storage
    end

    iter =1
    x = zeros(2p)
    for i in 1:iter
        indicator = use_indicator ? "indicator" : "bigM"
        data = @timed _, time_lmo, result = Boscia.solve(f, grad!, lmo; print_iter = 100, verbose=true, time_limit = time_limit, rel_dual_gap = rtol, dual_gap = 1e-4, use_postsolve = false, fw_epsilon = 1e-2, min_node_fw_epsilon =1e-5)
        df = DataFrame(seed=seed, dimension=dim, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size])
        file_name = "csv/bigM_vs_indicator_" * example * "_" * indicator * "_" * string(dim) * "_" * string(fac) * "_" * string(seed) * ".csv"
        CSV.write(file_name, df, append=false)
    end
    return x, f(x)
end

# TODO: test
function strong_branching_data()
    example = "lasso"

    seed=28 #20 25 27
    Random.seed!(seed)
    #seed = 0xe9265a85d25f83a6 
    
    # Constant parameters for the sparse regression
    # min norm(y-A β)² + λ_0 ∑ z_i + λ_2 ||β||²
    # s.t. -Mz_i<=β_i <= Mz_i
    # ∑ z_i <= k 
    # z_i ∈ {0,1} for i = 1,..,p 
    
    n=20
    p = 5*n
    k = ceil(n/5)
    group_size = convert(Int64, floor(p/k))
    M_g = 5.0
    
    lambda_0_g = 0.0
    lambda_2_g = 0.0
    A_g = rand(Float64, n, p)
    β_sol = rand(Distributions.Uniform(-M_g, M_g), p)
    k_int = convert(Int64, k)
    
    for i in 1:k_int
        for _ in 1:group_size-1
            β_sol[rand(((i-1)*group_size+1):(i*group_size))] = 0
        end
    end
    y_g = A_g * β_sol
    k=0 # correct k
    for i in 1:p
        if β_sol[i] == 0 
            k += 1
        end
    end
    k = p-k 
    
    groups= []
    for i in 1:(k_int-1)
        push!(groups, ((i-1)*group_size+1):(i*group_size))
    end
    push!(groups,((k_int-1)*group_size+1):p)
    
    
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o,p)
    z = MOI.add_variables(o,p)
    for i in 1:p
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne()) # or MOI.Integer()
    end 
    for i in 1:p
        MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,M_g], [x[i], z[i]]), 0.0), MOI.GreaterThan(0.0))
        MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-M_g], [x[i], z[i]]), 0.0), MOI.LessThan(0.0))
        # Indicator: x[i+p] = 1 => -M_g <= x[i] <= M_g
        gl = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[i])),],
            [0.0, 0.0], )
        gg = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, x[i])),],
            [0.0, 0.0], )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(M_g)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-M_g)))
    end
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p),z), 0.0), MOI.LessThan(1.0*k))
    for i in 1:k_int
        MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(group_size),z[groups[i]]), 0.0), MOI.GreaterThan(1.0))
    end
    lmo = FrankWolfe.MathOptLMO(o)
    global_bounds = Boscia.IntegerBounds()
    for i = 1:p
        push!(global_bounds, (i+p, MOI.GreaterThan(0.0)))
        push!(global_bounds, (i+p, MOI.LessThan(1.0)))
        push!(global_bounds, (i, MOI.GreaterThan(-M_g)))
        push!(global_bounds, (i, MOI.LessThan(M_g)))
    end
    
    function f(x)
        return sum((y_g-A_g*x[1:p]).^2) + lambda_0_g*sum(x[p+1:2p]) + lambda_2_g*FrankWolfe.norm(x[1:p])^2
    end
    function grad!(storage, x)
        storage.=vcat(2*(transpose(A_g)*A_g*x[1:p] - transpose(A_g)*y_g + lambda_2_g*x[1:p]), lambda_0_g*ones(p))
        return storage
    end
       
    min_number_lower = Inf
    dual_gap_decay_factor = 0.8
    
    ## Different strong branching

    iterations = [5, 10, 50, 100]
    iter = 2
    for (index,value) in enumerate(iterations)
        for i in 0:iter
            max_iter = value
            branching_strategy = Boscia.PartialStrongBranching(max_iter, 10^i*1e-3, SCIP.Optimizer())
            MOI.set(branching_strategy.optimizer, MOI.Silent(), true)

            Boscia.solve(f, grad!, lmo; verbose=true, branching_strategy = branching_strategy, dual_gap_decay_factor=dual_gap_decay_factor, min_number_lower=min_number_lower, fw_epsilon = 1e-4, print_iter=10)
            data = @timed _, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, branching_strategy = branching_strategy, dual_gap_decay_factor=dual_gap_decay_factor, min_number_lower=min_number_lower, fw_epsilon = 1e-4, print_iter=10)
            df = DataFrame(seed=seed, dimension=n, min_number_lower=min_number_lower, adaptive_gap=dual_gap_decay_factor, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size], node_level = result[:node_level])
            file_name = "csv/strong_branching_" * example * "_" * string(n) * "_" * string(seed) * "_" * string(max_iter) * "1e-"*string(3-i)*".csv"
            CSV.write(file_name, df, append=false)
        end
    end 
end 

# Lasso
example = "lasso"

seed=28 #20 25 27
Random.seed!(seed)
#seed = 0xe9265a85d25f83a6 

# Constant parameters for the sparse regression
# min norm(y-A β)² + λ_0 ∑ z_i + λ_2 ||β||²
# s.t. -Mz_i<=β_i <= Mz_i
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p 

n=20
p = 5*n
k = ceil(n/5)
group_size = convert(Int64, floor(p/k))
M_g = 5.0

const lambda_0_g = 0.0
const lambda_2_g = 0.0
const A_g = rand(Float64, n, p)
β_sol = rand(Distributions.Uniform(-M_g, M_g), p)
k_int = convert(Int64, k)

for i in 1:k_int
    for _ in 1:group_size-1
        β_sol[rand(((i-1)*group_size+1):(i*group_size))] = 0
    end
end
const y_g = A_g * β_sol
k=0 # correct k
for i in 1:p
    if β_sol[i] == 0 
        global k += 1
    end
end
k = p-k 

groups= []
for i in 1:(k_int-1)
    push!(groups, ((i-1)*group_size+1):(i*group_size))
end
push!(groups,((k_int-1)*group_size+1):p)


o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
MOI.empty!(o)
z = MOI.add_variables(o,p)
x = MOI.add_variables(o,p)
for i in 1:p
    MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
    MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
    MOI.add_constraint(o, z[i], MOI.ZeroOne()) # or MOI.Integer()
end 
for i in 1:p
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,M_g], [x[i], z[i]]), 0.0), MOI.GreaterThan(0.0))
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-M_g], [x[i], z[i]]), 0.0), MOI.LessThan(0.0))
    # Indicator: x[i+p] = 1 => -M_g <= x[i] <= M_g
    gl = MOI.VectorAffineFunction(
        [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
            MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[i])),],
        [0.0, 0.0], )
    gg = MOI.VectorAffineFunction(
        [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
            MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, x[i])),],
        [0.0, 0.0], )
    MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(M_g)))
    MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-M_g)))
end
MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p),z), 0.0), MOI.LessThan(1.0*k))
for i in 1:k_int
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(group_size),z[groups[i]]), 0.0), MOI.GreaterThan(1.0))
end
lmo = FrankWolfe.MathOptLMO(o)
global_bounds = Boscia.IntegerBounds()
for i = 1:p
    push!(global_bounds, (i+p, MOI.GreaterThan(0.0)))
    push!(global_bounds, (i+p, MOI.LessThan(1.0)))
    push!(global_bounds, (i, MOI.GreaterThan(-M_g)))
    push!(global_bounds, (i, MOI.LessThan(M_g)))
end

function f(x)
    return sum((y_g-A_g*x[1:p]).^2) + lambda_0_g*sum(x[p+1:2p]) + lambda_2_g*FrankWolfe.norm(x[1:p])^2
end
function grad!(storage, x)
    storage.=vcat(2*(transpose(A_g)*A_g*x[1:p] - transpose(A_g)*y_g + lambda_2_g*x[1:p]), lambda_0_g*ones(p))
    return storage
end
   
min_number_lower = Inf
dual_gap_decay_factor = 0.8
#=Boscia.solve(f, grad!, lmo; verbose=true, min_number_lower=min_number_lower, fw_epsilon = 1e-3, print_iter=10, )
data = @timed _, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, min_number_lower=min_number_lower, fw_epsilon = 1e-3, print_iter=10)
df = DataFrame(seed=seed, dimension=n, min_number_lower=min_number_lower, adaptive_gap=dual_gap_decay_factor, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size], node_level = result[:node_level])
file_name = "csv/dual_gap_" * example * "_" * string(n) * "_" * string(seed) * "_" * "most_infeasible" *".csv"
CSV.write(file_name, df, append=false)


# Strong branching
branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, HiGHS.Optimizer())
MOI.set(branching_strategy.optimizer, MOI.Silent(), true)

Boscia.solve(f, grad!, lmo; verbose=true, min_number_lower=min_number_lower, fw_epsilon = 1e-3, print_iter=10,  branching_strategy = branching_strategy)
data = @timed _, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, min_number_lower=min_number_lower, fw_epsilon = 1e-3, print_iter=10, branching_strategy = branching_strategy)
df = DataFrame(seed=seed, dimension=n, min_number_lower=min_number_lower, adaptive_gap=dual_gap_decay_factor, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size], node_level = result[:node_level])
file_name = "csv/dual_gap_" * example * "_" * string(n) * "_" * string(seed) * "_" * "strong_branching" *".csv"
CSV.write(file_name, df, append=false)

# Hybrid strong branching
function perform_strong_branch(tree, node)
    node.level <= length(tree.root.problem.integer_variables)/20 ? println("Strong") : println("most infeasible")
    return node.level <= length(tree.root.problem.integer_variables)/20
end
branching_strategy = Boscia.HybridStrongBranching(10, 1e-3, HiGHS.Optimizer(), perform_strong_branch)
MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)

Boscia.solve(f, grad!, lmo; verbose=true, min_number_lower=min_number_lower, fw_epsilon = 1e-3, print_iter=10, branching_strategy = branching_strategy)
data = @timed _, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, min_number_lower=min_number_lower, fw_epsilon = 1e-3, print_iter=10, branching_strategy = branching_strategy)
df = DataFrame(seed=seed, dimension=n, min_number_lower=min_number_lower, adaptive_gap=dual_gap_decay_factor, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size], node_level = result[:node_level])
file_name = "csv/dual_gap_" * example * "_" * string(n) * "_" * string(seed) * "_" * "hybrid_strong_branching" *".csv"
CSV.write(file_name, df, append=false)=#

## Different strong branching

iterations = [5, 10, 50, 100]
iter = 2
for (index,value) in enumerate(iterations)
    for i in 0:iter
        max_iter = value
        branching_strategy = Boscia.PartialStrongBranching(max_iter, 10^i*1e-3, SCIP.Optimizer())
        MOI.set(branching_strategy.optimizer, MOI.Silent(), true)

        Boscia.solve(f, grad!, lmo; verbose=true, branching_strategy = branching_strategy, dual_gap_decay_factor=dual_gap_decay_factor, min_number_lower=min_number_lower, fw_epsilon = 1e-4, print_iter=10)
        data = @timed _, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, branching_strategy = branching_strategy, dual_gap_decay_factor=dual_gap_decay_factor, min_number_lower=min_number_lower, fw_epsilon = 1e-4, print_iter=10)
        df = DataFrame(seed=seed, dimension=n, min_number_lower=min_number_lower, adaptive_gap=dual_gap_decay_factor, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size], node_level = result[:node_level])
        file_name = "csv/strong_branching_" * example * "_" * string(n) * "_" * string(seed) * "_" * string(max_iter) * "1e-"*string(3-i)*".csv"
        CSV.write(file_name, df, append=false)
    end
end 


## Different hybrid branching
#=
for i in [1,5,10,20]
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
end
=#
