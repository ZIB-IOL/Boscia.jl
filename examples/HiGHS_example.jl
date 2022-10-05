using Boscia
using FrankWolfe
using Random
using HiGHS
using SCIP
using LinearAlgebra
import MathOptInterface

const MOI = MathOptInterface

n = 6

const diffw = 0.5 * ones(n)
o = HiGHS.Optimizer()

MOI.set(o, MOI.Silent(), true)

x = MOI.add_variables(o, n)

for xi in x
    MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
    MOI.add_constraint(o, xi, MOI.LessThan(1.0))
    MOI.add_constraint(o, xi, MOI.ZeroOne())
end
lmo = FrankWolfe.MathOptLMO(o)

function f(x)
    return sum(0.5*(x.-diffw).^2)
end

function grad!(storage, x)
    @. storage = x-diffw
end

x, _, result = Boscia.solve(f, grad!, lmo, verbose = true)
