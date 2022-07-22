using BranchWolfe
using FrankWolfe
using Random
using SCIP
import Bonobo
import MathOptInterface
const MOI = MathOptInterface
using Dates
using Printf

Random.seed!(1)

const n = 25#25
const diff = Random.rand(Bool,n)*0.6.+0.3

o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
MOI.empty!(o)
x = MOI.add_variables(o, n)
for xi in x
    MOI.add_constraint(o, xi, MOI.ZeroOne())
    MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
    MOI.add_constraint(o, xi, MOI.LessThan(1.0))
end
lmo = FrankWolfe.MathOptLMO(o)


function f(x)
    return sum(0.5*(x.-diff).^2)
end
function grad!(storage, x)
    @. storage = x-diff
end

x, _ = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = false)
