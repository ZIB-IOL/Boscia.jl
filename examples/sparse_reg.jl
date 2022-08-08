using Statistics
using BranchWolfe
using FrankWolfe
using Random
using SCIP
import Bonobo
import MathOptInterface
const MOI = MathOptInterface
using Dates
using Printf

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

Random.seed!(42)
n0=20; p = 5*n0; k = ceil(n0/5);
const lambda_0 = rand(Float64); const lambda_2 = 10.0*rand(Float64);
const A = rand(Float64, n0, p)
const y = rand(Float64, n0)
const M = 2*var(A)

# "Sparse Regression" 
o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
MOI.empty!(o)
x = MOI.add_variables(o,p)
z = MOI.add_variables(o, p)
for i in 1:p
    MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
    MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
    MOI.add_constraint(o, z[i], MOI.ZeroOne()) # or MOI.Integer()
end 
for i in 1:p
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,M], [x[i], z[i]]), 0.0), MOI.GreaterThan(0.0))
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-M], [x[i], z[i]]), 0.0), MOI.LessThan(0.0))
end
MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p),z), 0.0), MOI.LessThan(k))
lmo = FrankWolfe.MathOptLMO(o)

function f(x)
    return sum((y-A*x[1:p]).^2) + lambda_0*sum(x[p+1:2p]) + lambda_2*FrankWolfe.norm(x[1:p])^2
end

function grad!(storage, x)
    storage.=vcat(2*(transpose(A)*A*x[1:p] - transpose(A)*y + lambda_2*x[1:p]), lambda_0*ones(p))
    return storage
end

x, _, result = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = true, fw_epsilon=1e-3, print_iter=1)

# @show result // too large to be output