using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface

n = 10
const ri = rand(n)
const ai = rand(n)
const Ωi = rand(Float64)
const bi = sum(ai)
Ai = randn(n, n)
Ai = Ai' * Ai
const Mi = (Ai + Ai') / 2
@assert isposdef(Mi)

# integer set
const I = 1:(n÷2)

o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
MOI.empty!(o)
x = MOI.add_variables(o, n)

for i in 1:n
    MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
    if i in I
        MOI.add_constraint(o, x[i], MOI.Integer())
    end
end
MOI.add_constraint(
    o,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai, x), 0.0),
    MOI.LessThan(bi),
)
MOI.add_constraint(
    o,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
    MOI.GreaterThan(1.0),
)
lmo = FrankWolfe.MathOptLMO(o)

function f(x)
    return 1 / 2 * Ωi * dot(x, Mi, x) - dot(ri, x)
end
function grad!(storage, x)
    mul!(storage, Mi, x, Ωi, 0)
    storage .-= ri
    return storage
end

x, _, result = Boscia.solve(f, grad!, lmo, verbose=true)
@test dot(ai, x) <= bi + 1e-6
@test f(x) <= f(result[:raw_solution]) + 1e-6


mutable struct GradientCutHandler{F, G, XT} <: SCIP.AbstractConstraintHandler
    o::SCIP.Optimizer
    f::F
    grad!::G
    storage::XT
    epivar::MOI.VariableIndex
    vars::Vector{MOI.VariableIndex}
    ncalls::Int
end

function SCIP.check(ch::GradientCutHandler, constraints::Vector{Ptr{SCIP.SCIP_CONS}}, sol::Ptr{SCIP.SCIP_SOL}, checkintegrality::Bool, checklprows::Bool, printreason::Bool, completely::Bool)
    @assert length(constraints) == 0
    values = SCIP.sol_values(ch.o, ch.vars, sol)
    zval = SCIP.sol_values(ch.o, [ch.epivar], sol)[1]
    if zval < ch.f(values)
        return SCIP.SCIP_INFEASIBLE
    end
    return SCIP.SCIP_FEASIBLE
end

function enforce_epigraph(ch::GradientCutHandler)
    values = SCIP.sol_values(ch.o, ch.vars)
    zval = SCIP.sol_values(ch.o, [ch.epivar])[1]
    fx = ch.f(values)
    ch.grad!(ch.storage, values)
    # f(x̂) + dot(∇f(x̂), x-x̂) - z ≤ 0 <=>
    # dot(∇f(x̂), x) - z ≤ dot(∇f(x̂), x̂) - f(x̂)
    if zval < fx - 1e-6
        f = dot(ch.storage, ch.vars) - ch.epivar
        s = MOI.LessThan(dot(ch.storage, values) - fx)
        fval = MOI.Utilities.eval_variables(vi -> SCIP.sol_values(ch.o, [vi])[1],  f)
        @assert fval > s.upper - 1e-11
        MOI.add_constraint(
            ch.o,
            dot(ch.storage, ch.vars) - ch.epivar,
            MOI.LessThan(dot(ch.storage, values) - fx),
        )
        ch.ncalls += 1
        return SCIP.SCIP_CONSADDED
    end
    return SCIP.SCIP_FEASIBLE
end

function SCIP.enforce_lp_sol(ch::GradientCutHandler, constraints, nusefulconss, solinfeasible)
    @assert length(constraints) == 0
    return enforce_epigraph(ch)
end

function SCIP.enforce_pseudo_sol(
        ch::GradientCutHandler, constraints, nusefulconss,
        solinfeasible, objinfeasible,
    )
    @assert length(constraints) == 0
    return enforce_epigraph(ch)
end

function SCIP.lock(ch::GradientCutHandler, constraint, locktype, nlockspos, nlocksneg)
    z::Ptr{SCIP.SCIP_VAR} = SCIP.var(ch.o, ch.epivar)
    if z != C_NULL
        SCIP.@SCIP_CALL SCIP.SCIPaddVarLocksType(ch.o, z, SCIP.SCIP_LOCKTYPE_MODEL, nlockspos, nlocksneg)
    end
    for x in ch.vars
        xi::Ptr{SCIP.SCIP_VAR} = SCIP.var(ch.o, x)
        xi == C_NULL && continue
        SCIP.@SCIP_CALL SCIP.SCIPaddVarLocksType(ch.o, xi, SCIP.SCIP_LOCKTYPE_MODEL, nlockspos + nlocksneg, nlockspos + nlocksneg)
    end
end

o = SCIP.Optimizer()
MOI.empty!(o)
x = MOI.add_variables(o, n)
for i in 1:n
    MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
    if i in I
        MOI.add_constraint(o, x[i], MOI.Integer())
    end
end
MOI.add_constraint(
    o,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai, x), 0.0),
    MOI.LessThan(bi),
)
MOI.add_constraint(
    o,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
    MOI.GreaterThan(1.0),
)

z = MOI.add_variable(o)

epigraph_ch = GradientCutHandler(o, f, grad!, zeros(length(x)), z, x, 0)
SCIP.include_conshdlr(o, epigraph_ch; needs_constraints=false, name="handler_gradient_cuts")

MOI.set(o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z)

MOI.optimize!(o)

@show epigraph_ch.ncalls
