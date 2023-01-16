using SCIP

mutable struct GradientCutHandler{F, G, XT} <: SCIP.AbstractConstraintHandler
    o::SCIP.Optimizer
    f::F
    grad!::G
    storage::XT
    epivar::MOI.VariableIndex
    vars::Vector{MOI.VariableIndex}
    ncalls::Int
end

function SCIP.check(ch::GradientCutHandler, constraints::Vector{Ptr{SCIP.SCIP_CONS}}, sol::Ptr{SCIP.SCIP_SOL}, checkintegrality::Bool, checklprows::Bool, printreason::Bool, completely::Bool; tol=1e-6)
    @assert length(constraints) == 0
    values = SCIP.sol_values(ch.o, ch.vars, sol)
    zval = SCIP.sol_values(ch.o, [ch.epivar], sol)[1]
    if zval < ch.f(values) - tol
        return SCIP.SCIP_INFEASIBLE
    end
    return SCIP.SCIP_FEASIBLE
end

function enforce_epigraph(ch::GradientCutHandler, tol=1e-6)
    values = SCIP.sol_values(ch.o, ch.vars)
    zval = SCIP.sol_values(ch.o, [ch.epivar])[1]
    fx = ch.f(values)
    ch.grad!(ch.storage, values)
    # f(x̂) + dot(∇f(x̂), x-x̂) - z ≤ 0 <=>
    # dot(∇f(x̂), x) - z ≤ dot(∇f(x̂), x̂) - f(x̂)
    if zval < fx - tol
        f = dot(ch.storage, ch.vars) - ch.epivar
        s = MOI.LessThan(dot(ch.storage, values) - fx)
        fval = MOI.Utilities.eval_variables(vi -> SCIP.sol_values(ch.o, [vi])[1],  f)
        @assert fval > s.upper - 1e-11
        MOI.add_constraint(
            ch.o,
            dot(ch.storage, ch.vars) - ch.epivar,
            MOI.LessThan(dot(ch.storage, values) - fx),
        )
        # print(ch.o) # KeyError: key Ptr{Nothing} @0x000000001421e2b0 not found
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
