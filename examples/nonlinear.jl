using FrankWolfe
using LinearAlgebra
import MathOptInterface
using Random
using Boscia
import Bonobo
using Printf
using Dates

# using SCIP
# const MOI = MathOptInterface

n = 40
seed = 10

Random.seed!(seed)

################################################################
# alternative implementation of LMO using MOI and SCIP
################################################################
# o = SCIP.Optimizer()
# MOI.set(o, MOI.Silent(), true)
# MOI.empty!(o)
# x = MOI.add_variables(o, n)
# for xi in x
#     MOI.add_constraint(o, xi, MOI.ZeroOne())
#     MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
#     MOI.add_constraint(o, xi, MOI.LessThan(1.0))
# end
# lmo = FrankWolfe.MathOptLMO(o)


################################################################
# LMO via CubeSimpleBLMO
################################################################
int_vars = collect(1:n)

lbs = zeros(n)
ubs = ones(n)

sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
# wrap the sblmo into a bound manager
lmo = Boscia.ManagedBoundedLMO(sblmo, lbs[int_vars], ubs[int_vars], n, int_vars)

const A = let
    A = randn(n, n)
    A' * A
end

@assert isposdef(A) == true

const y = Random.rand(Bool, n) * 0.6 .+ 0.3

function f(x)
    d = x - y
    return dot(d, A, d)
end

function grad!(storage, x)
    # storage = Ax
    mul!(storage, A, x)
    # storage = 2Ax - 2Ay
    return mul!(storage, A, y, -2, 2)
end

x, _, _ = Boscia.solve(f, grad!, lmo, verbose=true, print_iter=500)

@show x
