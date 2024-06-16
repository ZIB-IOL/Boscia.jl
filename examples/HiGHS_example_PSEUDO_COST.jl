using Boscia
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
lmo = Boscia.MathOptBLMO(o)

function f(x)
    return 0.5 * sum((x .- diffw) .^ 2)
end

function grad!(storage, x)
    @. storage = x - diffw
end
#pseudos = Dict{Int,Array{Float64}}(idx=>zeros(2) for idx in Boscia.get_integer_variables(lmo))
#branch_tracker = Dict{Int, Float64}(idx=> 0 for idx in Boscia.get_integer_variables(lmo))
iterations_stable = 1::Int
x, _, result = Boscia.solve(f, grad!, lmo, branching_strategy=Boscia.PSEUDO_COST(iterations_stable,false, lmo), verbose=true)