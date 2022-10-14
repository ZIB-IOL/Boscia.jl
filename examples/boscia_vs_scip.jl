using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
using Distributions
import MathOptInterface
MOI = MathOptInterface
using CSV
using DataFrames

function boscia_vs_scip(seed=1, dimension=5, iter=3)

    Random.seed!(seed)
    n = dimension
    ri = rand(n)
    ai = rand(n)
    Ωi = rand(Float64)
    bi = sum(ai)
    Ai = randn(n, n)
    Ai = Ai' * Ai
    Mi = (Ai + Ai') / 2
    @assert isposdef(Mi)

    # integer set
    I = 1:(n÷2)
    #I = collect(1:n)
    
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    limit = 1800
    # MOI.set(o, MOI.TimeLimitSec(), limit)
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
    # println("BOSCIA MODEL")
    # print(o)

    function f(x)
        return 1 / 2 * Ωi * dot(x, Mi, x) - dot(ri, x)
    end
    function grad!(storage, x)
        mul!(storage, Mi, x, Ωi, 0)
        storage .-= ri
        return storage
    end

    # x, _, result = Boscia.solve(f, grad!, lmo, verbose=true)
    # @test dot(ai, x) <= bi + 1e-6
    # @test f(x) <= f(result[:raw_solution]) + 1e-6
    # @show MOI.get(o, MOI.SolveTimeSec())

    open("examples/csv/boscia_vs_scip_mixed.csv", "w") do f
        CSV.write(f,[], writeheader=true, header=["seed", "dimension", "time_boscia", "solution_boscia", "termination_boscia", "time_scip", "solution_scip", "termination_scip", "ncalls_scip"])
    end

    intial_status = String(string(MOI.get(o, MOI.TerminationStatus())))
    # SCIP
    time_boscia = -Inf
    for i in 1:iter
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit)
        @show x, f(x)
        time_boscia=result[:total_time_in_sec]
        if result[:status] == "Optimal (tolerance reached)"
            status = "OPTIMAL"
        end
        df = DataFrame(seed=seed, dimension=n, time_boscia=time_boscia, solution_boscia=result[:primal_objective], termination_boscia=status, time_scip=-Inf, solution_scip=Inf, termination_scip=intial_status, ncalls_scip=-Inf)
        file_name = "examples/csv/boscia_vs_scip_mixed.csv"
        CSV.write(file_name, df, append=true)
        # display(df)
    end

    # @show x
    # @show time_boscia

    function build_scip_optimizer()
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
        
        z = MOI.add_variable(o)
        
        epigraph_ch = GradientCutHandler(o, f, grad!, zeros(length(x)), z, x, 0)
        SCIP.include_conshdlr(o, epigraph_ch; needs_constraints=false, name="handler_gradient_cuts")
        
        MOI.set(o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z)    
        # println("SCIP MODEL")
        # print(o)
        return o, epigraph_ch, x
    end

    for i in 1:iter
        o, epigraph_ch, x = build_scip_optimizer()
        MOI.set(o, MOI.TimeLimitSec(), limit)
        # MOI.set(o, MOI.AbsoluteGapTolerance(), 1.000000e-06) #AbsoluteGapTolerance not defined
        # MOI.set(o, MOI.RelativeGapTolerance(), 1.000000e-02)
        MOI.optimize!(o)
        time_scip = MOI.get(o, MOI.SolveTimeSec())
        # @show MOI.get(o, MOI.ObjectiveValue())
        vars_scip = MOI.get(o, MOI.VariablePrimal(), x)
        @assert sum(ai.*vars_scip) <= bi # constraint violated
        solution_scip = f(vars_scip)
        @show solution_scip
        termination_scip = String(string(MOI.get(o, MOI.TerminationStatus())))
        df_temp = DataFrame(CSV.File("examples/csv/boscia_vs_scip_mixed.csv", types=Dict(:seed=>Int64, :dimension=>Int64, :time_boscia=>Float64, :solution_boscia=>Float64, :termination_boscia=>String, :time_scip=>Float64, :solution_scip=>Float64, :termination_scip=>String, :ncalls_scip=>Float64)))
        df_temp[nrow(df_temp)-iter+i, :time_scip] = time_scip
        df_temp[nrow(df_temp)-iter+i, :solution_scip] = solution_scip
        df_temp[nrow(df_temp)-iter+i, :termination_scip] = termination_scip
        ncalls_scip = epigraph_ch.ncalls
        df_temp[nrow(df_temp)-iter+i, :ncalls_scip] = ncalls_scip
        CSV.write("examples/csv/boscia_vs_scip_mixed.csv", df_temp, append=false)
        # display(df_temp)
    end
end

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
    # @show zval, fx
    if zval < fx - 1e-6
        # println(fx - zval)
        f = dot(ch.storage, ch.vars) - ch.epivar
        s = MOI.LessThan(dot(ch.storage, values) - fx)
        fval = MOI.Utilities.eval_variables(vi -> SCIP.sol_values(ch.o, [vi])[1],  f)
        # @show fval - s.upper
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
    # @show ch.ncalls
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
