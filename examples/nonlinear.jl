using SCIP
using FrankWolfe
using LinearAlgebra
import MathOptInterface
using Random
using BranchWolfe
import Bonobo
using Printf
using Dates
using PyPlot

const MOI = MathOptInterface

n=5
seed=10

Random.seed!(seed)

o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
MOI.empty!(o)
x = MOI.add_variables(o, n)
for xi in x
    MOI.add_constraint(o, xi, MOI.ZeroOne())
    MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
    MOI.add_constraint(o, xi, MOI.LessThan(1.0))
end

# print(o.variable_info)

lmo = FrankWolfe.MathOptLMO(o)

#const A = LinearAlgebra.Symmetric(randn(n,n), :U)
const A = let
    A = randn(n,n)
    A'*A
end

@assert isposdef(A) == true
# add value on the diagonal 

const y = Random.rand(Bool,n)*0.6.+0.3

function f(x)
    d = x-y
    return dot(d, A, d)
end

function grad!(storage, x)
    # storage = Ax
    mul!(storage, A, x)
    # storage = 2Ax - 2Ay
    mul!(storage, A, y, -2, 2)
end

x, _,_ = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = true, print_iter=1)

@show x
