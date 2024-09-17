using Statistics
using Boscia
using FrankWolfe
using Random
using LinearAlgebra
using HiGHS
#using HiGHS
import Bonobo
import MathOptInterface
const MOI = MathOptInterface
using Dates
using Printf
using Test

# Sparse regression

# Constant parameters for the sparse regression
# min norm(y-A β)² + λ_0 ∑ z_i + λ_2 ||β||²
# s.t. -Mz_i <= β_i <= Mz_i
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p 

# A - matrix of observations.
# y - vector of results.
# We want to match Aβ as closely as possible to y 
# while having relative few non zero entries in β.
# Each continuous variable β_i is assigned a binary z_i,
# z_i = 0 => β_i = 0

############################################# Problem set up #################################################
n0 = 28
p = 5 * n0;
k = ceil(n0 / 5);
seed = rand(UInt64)
#seed =  0x0c1145c200469bf1
#seed = 0x2c6d6bd3949ad1f0
@show seed

Random.seed!(seed)


const lambda_0 = rand(Float64);
const lambda_2 = 10.0 * rand(Float64);
const A = rand(Float64, n0, p)
const y = rand(Float64, n0)
const M = 2 * var(A)
example_name = "sparse_regression_n0_" * string(n0) * "p_" * string(p)
################################################################################################################
file_name ="sparse_reg_examples_a_c" # for saving results in "./results/"
###### run once to make sure everything is initialized 


o = HiGHS.Optimizer()
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
lmo = Boscia.MathOptBLMO(o)

function f(x)
    xv = @view(x[1:p])
    return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
end

function grad!(storage, x)
    storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
    storage[p+1:2p] .= lambda_0
    return storage
end


x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, fw_epsilon=1e-3, print_iter=100, time_limit=30)


#################### weighted_sum #################################################################


o = HiGHS.Optimizer()
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
lmo = Boscia.MathOptBLMO(o)

function f(x)
    xv = @view(x[1:p])
    return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
end

function grad!(storage, x)
    storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
    storage[p+1:2p] .= lambda_0
    return storage
end
iterations_stable = 4::Int

decision_function = "weighted_sum"
if decision_function == "product"
    μ = 1e-6
else
    μ = 0.1 # μ used in the computation of the branching score
end

x, _, result = Boscia.solve(f, grad!, lmo, branching_strategy=Boscia.PSEUDO_COST(iterations_stable,false, lmo, μ, decision_function),verbose=true, fw_epsilon=1e-3, print_iter=100, time_limit=300)
settings = "PSEUDO_COST_" * decision_function * "_" * string(iterations_stable) * "_μ=" * string(μ)
Boscia.save_results(result, settings, example_name, seed, file_name, true) 



o = HiGHS.Optimizer()
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
lmo = Boscia.MathOptBLMO(o)

function f(x)
    xv = @view(x[1:p])
    return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
end

function grad!(storage, x)
    storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
    storage[p+1:2p] .= lambda_0
    return storage
end
iterations_stable = 10::Int

decision_function = "weighted_sum"
if decision_function == "product"
    μ = 1e-6
else
    μ = 0.1 # μ used in the computation of the branching score
end

x, _, result = Boscia.solve(f, grad!, lmo, branching_strategy=Boscia.PSEUDO_COST(iterations_stable,false, lmo, μ, decision_function),verbose=true, fw_epsilon=1e-3, print_iter=100, time_limit=300)
settings = "PSEUDO_COST_" * decision_function * "_" * string(iterations_stable) * "_μ=" * string(μ)
Boscia.save_results(result, settings, example_name, seed, file_name, false) 



# ############################## product rule ############################################################


o = HiGHS.Optimizer()
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
lmo = Boscia.MathOptBLMO(o)

function f(x)
    xv = @view(x[1:p])
    return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
end

function grad!(storage, x)
    storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
    storage[p+1:2p] .= lambda_0
    return storage
end
iterations_stable = 4::Int

decision_function = "product"
if decision_function == "product"
    μ = 1e-6
else
    μ = 0.7 # μ used in the computation of the branching score
end
x, _, result = Boscia.solve(f, grad!, lmo, branching_strategy=Boscia.PSEUDO_COST(iterations_stable,false, lmo, μ, decision_function),verbose=true, fw_epsilon=1e-3, print_iter=100, time_limit=300)
settings = "PSEUDO_COST_" * decision_function * "_" * string(iterations_stable)
Boscia.save_results(result, settings, example_name, seed, file_name, false) 



o = HiGHS.Optimizer()
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
lmo = Boscia.MathOptBLMO(o)

function f(x)
    xv = @view(x[1:p])
    return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
end

function grad!(storage, x)
    storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
    storage[p+1:2p] .= lambda_0
    return storage
end
iterations_stable = 10::Int

decision_function = "product"
if decision_function == "product"
    μ = 1e-6
else
    μ = 0.7 # μ used in the computation of the branching score
end
x, _, result = Boscia.solve(f, grad!, lmo, branching_strategy=Boscia.PSEUDO_COST(iterations_stable,false, lmo, μ, decision_function),verbose=true, fw_epsilon=1e-3, time_limit=300)
settings = "PSEUDO_COST_" * decision_function * "_" * string(iterations_stable)
Boscia.save_results(result, settings, example_name, seed, file_name, false) 

# ######################### MOST_INFEASIBLE ##########################################################
o = HiGHS.Optimizer()
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
lmo = Boscia.MathOptBLMO(o)

function f(x)
    xv = @view(x[1:p])
    return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
end

function grad!(storage, x)
    storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
    storage[p+1:2p] .= lambda_0
    return storage
end

x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, fw_epsilon=1e-3, time_limit=300)
settings = "MOST_INFEASIBLE"
Boscia.save_results(result, settings, example_name, seed, file_name, false) 





o = HiGHS.Optimizer()

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
function f(x)
    xv = @view(x[1:p])
    return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
end

function grad!(storage, x)
    storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
    storage[p+1:2p] .= lambda_0
    return storage
end
lmo = Boscia.MathOptBLMO(o)
blmo = Boscia.MathOptBLMO(HiGHS.Optimizer())
branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
MOI.set(branching_strategy.bounded_lmo.o, MOI.Silent(), true)

x, _, result_strong_branching =
    Boscia.solve(f, grad!, lmo, verbose=true, branching_strategy=branching_strategy, time_limit=300)
settings = "Strong_Branching"
Boscia.save_results(result_strong_branching, settings, example_name, seed, file_name, false) 