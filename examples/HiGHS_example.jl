using Boscia
using FrankWolfe
using Random
using HiGHS
using LinearAlgebra
import MathOptInterface

const MOI = MathOptInterface

println("\nHiGHS Example")

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
blmo = Boscia.MathOptBLMO(o)

function f(x)
    return 0.5 * sum((x .- diffw) .^ 2)
end

function grad!(storage, x)
    @. storage = x - diffw
end

settings = Boscia.create_default_settings()
settings.branch_and_bound[:verbose] = true
x, _, result = Boscia.solve(f, grad!, blmo, settings=settings)
