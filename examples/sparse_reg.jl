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

# Constant parameters for the sparse regression
# min norm(y-A β)² + λ_0 ∑ z_i + λ_2 ||β||²
# s.t. -Mz_i <= β_i <= Mz_i
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p 
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
x = MOI.add_variables(o,2p)
for i in p+1:2p
    MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
    MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
    MOI.add_constraint(o, x[i], MOI.ZeroOne()) # or MOI.Integer()
end 
for i in 1:p
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,M], [x[i], x[i+p]]), 0.0), MOI.GreaterThan(0.0))
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-M], [x[i], x[i+p]]), 0.0), MOI.LessThan(0.0))
end
MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p),x[p+1:2p]), 0.0), MOI.LessThan(k))
lmo = FrankWolfe.MathOptLMO(o)

function f(x)
    return sum((y-A*x[1:p]).^2) + lambda_0*sum(x[p+1:2p]) + lambda_2*FrankWolfe.norm(x[1:p])^2
end

function grad!(storage, x)
    storage.=vcat(2*(transpose(A)*A*x[1:p] - transpose(A)*y + lambda_2*x[1:p]), lambda_0*ones(p))
    return storage
end

x, _ = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = true, fw_epsilon=1e-3, print_iter=1)
