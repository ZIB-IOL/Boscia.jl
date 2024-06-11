# This script sets up and solves a linear optimization problem using the Boscia and FrankWolfe packages in Julia. 
# The problem involves a 6-dimensional variable `x` that is constrained to be between 0 and 1 (inclusive) and is also restricted to binary values (0 or 1).
# The objective function `f(x)` is a quadratic function representing the squared Euclidean distance from a vector `diffw` (which is set to 0.5 for all components).
# The gradient of the objective function is computed by `grad!`.
# The `HiGHS.Optimizer` is used to handle the optimization problem and the `FrankWolfe.MathOptLMO` interface connects it to the Frank-Wolfe algorithm.
# The final solution `x` and the result status are obtained using `Boscia.solve`.using Boscia


using FrankWolfe
using Random
using HiGHS
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
    return 0.5 * sum((x .- diffw) .^ 2)
end

function grad!(storage, x)
    @. storage = x - diffw
end

x, _, result = Boscia.solve(f, grad!, lmo, verbose=true)
