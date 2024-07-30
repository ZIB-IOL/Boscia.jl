using Boscia
using StableRNGs
using SCIP
using MathOptInterface
const MOI = MathOptInterface

seed = 1234
n = 20 #50
rng = StableRNG(seed)
diffi = rand(rng, Bool, n) * 0.6 .+ 0.3
    
function f(x)
    return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
end
function grad!(storage, x)
    @. storage = x - diffi
end
int_vars = unique!(rand(collect(1:n), Int(floor(n / 2))))

solution = copy(diffi)
solution[int_vars] .= round.(solution[int_vars])
@show diffi
@show solution
@show f(solution)

# Cube Simple LMO
lbs = zeros(n)
ubs = ones(n)
lmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
Boscia.solve(f, grad!, lmo, lbs[int_vars], ubs[int_vars], int_vars, n; variant=Boscia.Blended(), verbose=true)

# MOI
o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
MOI.empty!(o)
x = MOI.add_variables(o, n)
for xi in x
    MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
    MOI.add_constraint(o, xi, MOI.LessThan(1.0))
    if xi.value in int_vars
        MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
    end
end
lmo = Boscia.MathOptBLMO(o)
x, _, _ = Boscia.solve(f, grad!, lmo; variant=Boscia.Blended(), verbose=true)
