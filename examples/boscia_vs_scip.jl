using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface
using CSV
using DataFrames

function boscia_vs_scip(mode, seed=1, dimension=5, iter=3; scip_oa=true)

    Random.seed!(seed)
    n = dimension
    ri = rand(n)
    ai = rand(n)
    Ωi = rand()
    bi = sum(ai)
    Ai = randn(n, n)
    Ai = Ai' * Ai
    Mi = (Ai + Ai') / 2
    @assert isposdef(Mi)

    # integer set
    if mode == "integer"
        I = collect(1:n)
    elseif mode == "mixed"
        I = 1:(n÷2)
    end
    
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    limit = 1800
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

    initial_status = String(string(MOI.get(o, MOI.TerminationStatus())))
    # SCIP
    time_boscia = -Inf
    for i in 1:iter
        Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit)
        @show x, f(x)
        @test dot(ai, x) <= bi + 1e-6
        @test f(x) <= f(result[:raw_solution]) + 1e-6
        time_boscia=result[:total_time_in_sec]
        status = result[:status]
        if result[:status] == "Optimal (tolerance reached)"
            status = "OPTIMAL"
        end
        df = DataFrame(seed=seed, dimension=n, time_boscia=time_boscia, solution_boscia=result[:primal_objective], termination_boscia=status, time_scip=-Inf, solution_scip=Inf, termination_scip=initial_status, ncalls_scip=-Inf)
        if mode == "integer"
            file_name = "examples/csv/boscia_vs_scip_integer_50.csv"
        elseif mode == "mixed"
            file_name = "examples/csv/boscia_vs_scip_mixed_50.csv"
        end
        if !isfile(file_name)
            CSV.write(file_name, df, append=true, writeheader=true)
        else 
            CSV.write(file_name, df, append=true)
        end
    end

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

    if scip_oa
        for i in 1:iter
            o, epigraph_ch, x = build_scip_optimizer()
            MOI.set(o, MOI.TimeLimitSec(), limit)
            # MOI.set(o, MOI.AbsoluteGapTolerance(), 1.000000e-06) #AbsoluteGapTolerance not defined
            # MOI.set(o, MOI.RelativeGapTolerance(), 1.000000e-02)
            MOI.optimize!(o)
            time_scip = MOI.get(o, MOI.SolveTimeSec())
            vars_scip = MOI.get(o, MOI.VariablePrimal(), x)
            @assert sum(ai.*vars_scip) <= bi + 1e-6 # constraint violated
            solution_scip = f(vars_scip)
            @show solution_scip
            termination_scip = String(string(MOI.get(o, MOI.TerminationStatus())))
            if mode == "integer"
                file_name = "examples/csv/boscia_vs_scip_integer_50.csv"
            elseif mode == "mixed"
                file_name = "examples/csv/boscia_vs_scip_mixed_50.csv"
            end
            df_temp = DataFrame(CSV.File(file_name, types=Dict(:seed=>Int64, :dimension=>Int64, :time_boscia=>Float64, :solution_boscia=>Float64, :termination_boscia=>String, :time_scip=>Float64, :solution_scip=>Float64, :termination_scip=>String, :ncalls_scip=>Float64)))
            df_temp[nrow(df_temp)-iter+i, :time_scip] = time_scip
            df_temp[nrow(df_temp)-iter+i, :solution_scip] = solution_scip
            df_temp[nrow(df_temp)-iter+i, :termination_scip] = termination_scip
            ncalls_scip = epigraph_ch.ncalls
            df_temp[nrow(df_temp)-iter+i, :ncalls_scip] = ncalls_scip
            CSV.write(file_name, df_temp, append=false)
        end
    end
end

# mutable struct GradientCutHandler{F, G, XT} <: SCIP.AbstractConstraintHandler
#     o::SCIP.Optimizer
#     f::F
#     grad!::G
#     storage::XT
#     epivar::MOI.VariableIndex
#     vars::Vector{MOI.VariableIndex}
#     ncalls::Int
# end

# function SCIP.check(ch::GradientCutHandler, constraints::Vector{Ptr{SCIP.SCIP_CONS}}, sol::Ptr{SCIP.SCIP_SOL}, checkintegrality::Bool, checklprows::Bool, printreason::Bool, completely::Bool)
#     @assert length(constraints) == 0
#     values = SCIP.sol_values(ch.o, ch.vars, sol)
#     zval = SCIP.sol_values(ch.o, [ch.epivar], sol)[1]
#     if zval < ch.f(values) - 1e-6
#         return SCIP.SCIP_INFEASIBLE
#     end
#     return SCIP.SCIP_FEASIBLE
# end

# function enforce_epigraph(ch::GradientCutHandler)
#     values = SCIP.sol_values(ch.o, ch.vars)
#     zval = SCIP.sol_values(ch.o, [ch.epivar])[1]
#     fx = ch.f(values)
#     ch.grad!(ch.storage, values)
#     # f(x̂) + dot(∇f(x̂), x-x̂) - z ≤ 0 <=>
#     # dot(∇f(x̂), x) - z ≤ dot(∇f(x̂), x̂) - f(x̂)
#     if zval < fx - 1e-6
#         f = dot(ch.storage, ch.vars) - ch.epivar
#         s = MOI.LessThan(dot(ch.storage, values) - fx)
#         fval = MOI.Utilities.eval_variables(vi -> SCIP.sol_values(ch.o, [vi])[1],  f)
#         @assert fval > s.upper - 1e-11
#         MOI.add_constraint(
#             ch.o,
#             dot(ch.storage, ch.vars) - ch.epivar,
#             MOI.LessThan(dot(ch.storage, values) - fx),
#         )
#         # print(ch.o) # KeyError: key Ptr{Nothing} @0x000000001421e2b0 not found
#         ch.ncalls += 1
#         return SCIP.SCIP_CONSADDED
#     end
#     return SCIP.SCIP_FEASIBLE
# end

# function SCIP.enforce_lp_sol(ch::GradientCutHandler, constraints, nusefulconss, solinfeasible)
#     @assert length(constraints) == 0
#     return enforce_epigraph(ch)
# end

# function SCIP.enforce_pseudo_sol(
#         ch::GradientCutHandler, constraints, nusefulconss,
#         solinfeasible, objinfeasible,
#     )
#     @assert length(constraints) == 0
#     return enforce_epigraph(ch)
# end

# function SCIP.lock(ch::GradientCutHandler, constraint, locktype, nlockspos, nlocksneg)
#     z::Ptr{SCIP.SCIP_VAR} = SCIP.var(ch.o, ch.epivar)
#     if z != C_NULL
#         SCIP.@SCIP_CALL SCIP.SCIPaddVarLocksType(ch.o, z, SCIP.SCIP_LOCKTYPE_MODEL, nlockspos, nlocksneg)
#     end
#     for x in ch.vars
#         xi::Ptr{SCIP.SCIP_VAR} = SCIP.var(ch.o, x)
#         xi == C_NULL && continue
#         SCIP.@SCIP_CALL SCIP.SCIPaddVarLocksType(ch.o, xi, SCIP.SCIP_LOCKTYPE_MODEL, nlockspos + nlocksneg, nlockspos + nlocksneg)
#     end
# end

function boscia_vs_scip_birkhoff(seed=1, dimension=4, iter=3, k=3)
    limit = 1800
    Random.seed!(seed)
    n = dimension
    Xstar = rand(n, n)
    while norm(sum(Xstar, dims=1) .- 1) > 1e-6 || norm(sum(Xstar, dims=2) .- 1) > 1e-6
        Xstar ./= sum(Xstar, dims=1)
        Xstar ./= sum(Xstar, dims=2)
    end
    
    function f(x)
        s = zero(eltype(x))
        for i in eachindex(Xstar)
            s += 0.5 * (sum(x[(j-1)*n^2+i] for j in 1:k) - Xstar[i])^2
        end
        return s
    end
    
    # note: reshape gives a reference to the same data, so this is updating storage in-place
    function grad!(storage, x)
        storage .= 0
        for j in 1:k
            Sk = reshape(@view(storage[(j-1)*n^2+1:j*n^2]), n, n)
            @. Sk = -Xstar
            for m in 1:k
                Yk = reshape(@view(x[(m-1)*n^2+1:m*n^2]), n, n)
                @. Sk += Yk
            end
        end
        return storage
    end

    function build_birkhoff_lmo()
        o = SCIP.Optimizer()
        MOI.set(o, MOI.Silent(), true)
        MOI.empty!(o)
        Y = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
        X = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
        theta = MOI.add_variables(o, k)
    
        for i in 1:k
            MOI.add_constraint.(o, Y[i], MOI.GreaterThan(0.0))
            MOI.add_constraint.(o, Y[i], MOI.LessThan(1.0))
            MOI.add_constraint.(o, X[i], MOI.ZeroOne())
            MOI.add_constraint(o, theta[i], MOI.GreaterThan(0.0))
            MOI.add_constraint(o, theta[i], MOI.LessThan(1.0))
            # doubly stochastic constraints
            MOI.add_constraint.(
                o,
                vec(sum(X[i], dims=1, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
                MOI.EqualTo(1.0),
            )
            MOI.add_constraint.(
                o,
                vec(sum(X[i], dims=1, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
                MOI.EqualTo(1.0),
            )
            # 0 ≤ Y_i ≤ X_i
            MOI.add_constraint.(o, 1.0 * Y[i] - X[i], MOI.LessThan(0.0))
            # 0 ≤ θ_i - Y_i ≤ 1 - X_i
            MOI.add_constraint.(o, 1.0 * theta[i] .- Y[i] .+ X[i], MOI.LessThan(1.0))
        end
        MOI.add_constraint(o, sum(theta, init=0.0), MOI.EqualTo(1.0))
        return FrankWolfe.MathOptLMO(o)
    end

    initial_status = String(string(MOI.get(SCIP.Optimizer(), MOI.TerminationStatus())))
    # SCIP
    time_boscia = -Inf
    for i in 1:iter
        lmo = build_birkhoff_lmo()
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit)
        @show x, f(x)
        @test f(x) <= f(result[:raw_solution]) + 1e-5
        time_boscia=result[:total_time_in_sec]
        status = result[:status]
        if result[:status] == "Optimal (tolerance reached)"
            status = "OPTIMAL"
        end
        df = DataFrame(seed=seed, dimension=n, time_boscia=time_boscia, solution_boscia=result[:primal_objective], termination_boscia=status, time_scip=-Inf, solution_scip=Inf, termination_scip=initial_status, ncalls_scip=-Inf)
        file_name = joinpath(@__DIR__, "csv/boscia_vs_scip_birkhoff_$dimension.csv")
        if !isfile(file_name)
            CSV.write(file_name, df, append=true, writeheader=true)
        else 
            CSV.write(file_name, df, append=true)
        end
    end

    function build_scip_optimizer()
        o = SCIP.Optimizer()
        MOI.set(o, MOI.Silent(), true)
        MOI.empty!(o)
        Y = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
        X = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
        theta = MOI.add_variables(o, k)
    
        for i in 1:k
            MOI.add_constraint.(o, Y[i], MOI.GreaterThan(0.0))
            MOI.add_constraint.(o, Y[i], MOI.LessThan(1.0))
            MOI.add_constraint.(o, X[i], MOI.ZeroOne())
            MOI.add_constraint(o, theta[i], MOI.GreaterThan(0.0))
            MOI.add_constraint(o, theta[i], MOI.LessThan(1.0))
            # doubly stochastic constraints
            MOI.add_constraint.(
                o,
                vec(sum(X[i], dims=1, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
                MOI.EqualTo(1.0),
            )
            MOI.add_constraint.(
                o,
                vec(sum(X[i], dims=1, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
                MOI.EqualTo(1.0),
            )
            # 0 ≤ Y_i ≤ X_i
            MOI.add_constraint.(o, 1.0 * Y[i] - X[i], MOI.LessThan(0.0))
            # 0 ≤ θ_i - Y_i ≤ 1 - X_i
            MOI.add_constraint.(o, 1.0 * theta[i] .- Y[i] .+ X[i], MOI.LessThan(1.0))
        end
        MOI.add_constraint(o, sum(theta, init=0.0), MOI.EqualTo(1.0))
        x = MOI.get(o, MOI.ListOfVariableIndices())
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
        MOI.optimize!(o)
        time_scip = MOI.get(o, MOI.SolveTimeSec())
        # @show MOI.get(o, MOI.ObjectiveValue())
        vars_scip = MOI.get(o, MOI.VariablePrimal(), x)
        solution_scip = f(vars_scip)
        @show solution_scip
        termination_scip = String(string(MOI.get(o, MOI.TerminationStatus())))
        df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_vs_scip_birkhoff_$dimension.csv"), types=Dict(:seed=>Int64, :dimension=>Int64, :time_boscia=>Float64, :solution_boscia=>Float64, :termination_boscia=>String, :time_scip=>Float64, :solution_scip=>Float64, :termination_scip=>String, :ncalls_scip=>Float64)))
        df_temp[nrow(df_temp)-iter+i, :time_scip] = time_scip
        df_temp[nrow(df_temp)-iter+i, :solution_scip] = solution_scip
        df_temp[nrow(df_temp)-iter+i, :termination_scip] = termination_scip
        ncalls_scip = epigraph_ch.ncalls
        df_temp[nrow(df_temp)-iter+i, :ncalls_scip] = ncalls_scip
        CSV.write(joinpath(@__DIR__, "csv/boscia_vs_scip_birkhoff_$dimension.csv"), df_temp, append=false)
    end
end

function boscia_vs_scip_grouped(seed=1, iter=3; n=50, p=5n, k=n ÷ 2)
    limit = 1800
    Random.seed!(seed)
    lambda_0 = rand()
    lambda_2 = 10.0 * rand()
    A = rand(n, p)
    y = rand(n)
    M = 2 * var(A)

    function build_lmo()
        o = SCIP.Optimizer()
        MOI.set(o, MOI.Silent(), true)
        MOI.empty!(o)
        x = MOI.add_variables(o, 2p)
        for i in p+1:2p
            MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
            MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
            MOI.add_constraint(o, x[i], MOI.ZeroOne())
        end
        for i in 1:p
            MOI.add_constraint(
                o,
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, M], [x[i], x[i+p]]), 0.0),
                MOI.GreaterThan(0.0),
            )
            MOI.add_constraint(
                o,
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -M], [x[i], x[i+p]]), 0.0),
                MOI.LessThan(0.0),
            )
        end
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p), x[p+1:2p]), 0.0),
            MOI.LessThan(1.0 * k),
        )
        lmo = FrankWolfe.MathOptLMO(o)
        return lmo
    end
    function f(x)
        xv = @view(x[1:p])
        return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
    end

    function grad!(storage, x)
        xv = @view(x[1:p])
        storage[1:p] .= 2 * (transpose(A) * A * xv - transpose(A) * y + lambda_2 * xv)
        storage[p+1:2p] .= lambda_0
        return storage
    end


    function build_scip_optimizer()
        o = SCIP.Optimizer()
        MOI.set(o, MOI.Silent(), true)
        MOI.empty!(o)
        x = MOI.add_variables(o, 2p)
        for i in p+1:2p
            MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
            MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
            MOI.add_constraint(o, x[i], MOI.ZeroOne())
        end
        for i in 1:p
            MOI.add_constraint(
                o,
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, M], [x[i], x[i+p]]), 0.0),
                MOI.GreaterThan(0.0),
            )
            MOI.add_constraint(
                o,
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -M], [x[i], x[i+p]]), 0.0),
                MOI.LessThan(0.0),
            )
        end
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p), x[p+1:2p]), 0.0),
            MOI.LessThan(1.0 * k),
        )
        z = MOI.add_variable(o)
        
        epigraph_ch = GradientCutHandler(o, f, grad!, zeros(length(x)), z, x, 0)
        SCIP.include_conshdlr(o, epigraph_ch; needs_constraints=false, name="handler_gradient_cuts")
        
        MOI.set(o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z)    
        return o, epigraph_ch, x
    end

    initial_status = String(string(MOI.get(SCIP.Optimizer(), MOI.TerminationStatus())))
    # SCIP
    time_boscia = -Inf
    for i in 1:iter
        lmo = build_lmo()
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit)
        @show x, f(x)
        @test f(x) <= f(result[:raw_solution]) + 1e-5
        time_boscia=result[:total_time_in_sec]
        status = result[:status]
        if occursin("Optimal", result[:status])
            status = "OPTIMAL"
        end
        df = DataFrame(seed=seed, dimension=n, time_boscia=time_boscia, solution_boscia=result[:primal_objective], termination_boscia=status, time_scip=-Inf, solution_scip=Inf, termination_scip=initial_status, ncalls_scip=-Inf)
        file_name = joinpath(@__DIR__, "csv/boscia_vs_scip_sparsereg_$n.csv")
        if !isfile(file_name)
            CSV.write(file_name, df, append=true, writeheader=true)
        else
            CSV.write(file_name, df, append=true)
        end
    end

    for i in 1:iter
        o, epigraph_ch, x = build_scip_optimizer()
        MOI.set(o, MOI.TimeLimitSec(), limit)
        MOI.optimize!(o)
        time_scip = MOI.get(o, MOI.SolveTimeSec())
        # @show MOI.get(o, MOI.ObjectiveValue())
        vars_scip = MOI.get(o, MOI.VariablePrimal(), x)
        solution_scip = f(vars_scip)
        @show solution_scip
        termination_scip = String(string(MOI.get(o, MOI.TerminationStatus())))
        df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_vs_scip_sparsereg_$n.csv"), types=Dict(:seed=>Int64, :dimension=>Int64, :time_boscia=>Float64, :solution_boscia=>Float64, :termination_boscia=>String, :time_scip=>Float64, :solution_scip=>Float64, :termination_scip=>String, :ncalls_scip=>Float64)))
        df_temp[nrow(df_temp)-iter+i, :time_scip] = time_scip
        df_temp[nrow(df_temp)-iter+i, :solution_scip] = solution_scip
        df_temp[nrow(df_temp)-iter+i, :termination_scip] = termination_scip
        ncalls_scip = epigraph_ch.ncalls
        df_temp[nrow(df_temp)-iter+i, :ncalls_scip] = ncalls_scip
        CSV.write(joinpath(@__DIR__, "csv/boscia_vs_scip_sparsereg_$n.csv"), df_temp, append=false)
    end
end
