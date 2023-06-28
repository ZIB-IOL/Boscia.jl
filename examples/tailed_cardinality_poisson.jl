using Statistics
using Boscia
using FrankWolfe
using Random
using LinearAlgebra
using SCIP
using Distributions
import MathOptInterface
const MOI = MathOptInterface
using Dates
using Printf
using DataFrames
using CSV
include("scip_oa.jl")

# Two-tailed cardinality-constrained regression

# Constant parameters for the sparse regression
# min_{x,s,z} ∑_i f_i(x) - λ ∑_j z_j + μ ||x||²
# s.t. z_j = 1 => s ≤ 0
#      s ≥ x_j  - τ_j
#      s ≥ -x_j - τ_j
#      x, s ∈ X, 
#      z_j ∈ {0,1}^n

# f_i - contributions to the loss function.
# x - predictors.
# λ - cardinality penalty
# μ - ℓ₂ penalty

# modified from
# Tractable Continuous Approximations for Constraint Selection via Cardinality Minimization, Ahn, Gangammanavar†1, Troxell

function sparse_regression(seed=1, dimension=10; bo_mode="boscia") 
    limit = 1800
    f, grad!, τ, M = build_objective_gradient(seed, dimension)
    o = SCIP.Optimizer()
    lmo, _ = build_twotailed_optimizer(o, τ, M)
    # println("BOSCIA MODEL")
    # print(lmo.o)

    # "Sparse Regression" 

    x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=10, afw=true)

    if bo_mode == "afw"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, afw=true)
    elseif bo_mode == "as_ss"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=false, warmstart_shadow_set=false)
    elseif bo_mode == "as"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=false, warmstart_shadow_set=true)
    elseif bo_mode == "ss"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=true, warmstart_shadow_set=false)
    elseif bo_mode == "boscia"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit)
    end             

    total_time_in_sec=result[:total_time_in_sec]
    status = result[:status]
    if occursin("Optimal", result[:status])
        status = "OPTIMAL"
    end
    df = DataFrame(seed=seed, dimension=dimension, M=M, time=total_time_in_sec, solution=result[:primal_objective], dual_gap=result[:dual_gap], termination=status, ncalls=result[:lmo_calls])
    if bo_mode ==  "afw"
        file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_tailed_cardinality_poisson.csv")
    elseif bo_mode == "boscia"
        file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_tailed_cardinality_poisson.csv")
    else 
        file_name = joinpath(@__DIR__,"csv/no_warm_start_" * bo_mode * "_tailed_cardinality_poisson.csv")
    end
    if !isfile(file_name)
        CSV.write(file_name, df, append=true, writeheader=true)
    else 
        CSV.write(file_name, df, append=true)
    end
    # @show x
    # @show f(x) 
    # return f, x

    # xv = @view(x[1:p])
    # predictions = [p > 0.5 ? 1 : -1 for p in A*xv]
    # @show (predictions, y)
    return f(x), x
end

function sparse_reg_scip(seed=1, dimension=10)
    limit = 1800
    f, grad!, τ, M = build_objective_gradient(seed, dimension)
    lmo, epigraph_ch, (x, z, s), lmo_check = build_scip_optimizer(τ, M, limit, f, grad!)
    MOI.set(lmo.o, MOI.TimeLimitSec(), limit)
    # MOI.set(o, MOI.AbsoluteGapTolerance(), 1.000000e-06) #AbsoluteGapTolerance not defined
    # MOI.set(o, MOI.RelativeGapTolerance(), 1.000000e-02)
    MOI.optimize!(lmo.o)
    termination_scip = String(string(MOI.get(lmo.o, MOI.TerminationStatus())))
    
    println(termination_scip)
    if termination_scip != "INFEASIBLE" && termination_scip != "DUAL_INFEASIBLE"
        time_scip = MOI.get(lmo.o, MOI.SolveTimeSec())
        var_x_scip = MOI.get(lmo.o, MOI.VariablePrimal(), x)
        var_z_scip = MOI.get(lmo.o, MOI.VariablePrimal(), z)
        var_s_scip = MOI.get(lmo.o, MOI.VariablePrimal(), s)
        vars_scip = vcat(var_x_scip, var_z_scip, var_s_scip)
        @assert Boscia.is_linear_feasible(lmo_check.o, vars_scip)
        #@assert sum(ai.*vars_scip) <= bi + 1e-6 # constraint violated
        solution_scip = f(vars_scip)
        ncalls_scip = epigraph_ch.ncalls

        df = DataFrame(seed=seed, dimension=dimension, M=M, time=time_scip, solution=solution_scip, termination=termination_scip, calls=ncalls_scip)
    else
        time_scip = MOI.get(lmo.o, MOI.SolveTimeSec())
        #@assert sum(ai.*vars_scip) <= bi + 1e-6 # constraint violated
        ncalls_scip = epigraph_ch.ncalls

        df = DataFrame(seed=seed, dimension=dimension, M=M, time=time_scip, solution=Inf, termination=termination_scip, calls=ncalls_scip)

    end
    file_name = joinpath(@__DIR__,"csv/scip_oa_tailed_cardinality_poisson.csv")
    if !isfile(file_name)
        CSV.write(file_name, df, append=true, writeheader=true)
    else 
        CSV.write(file_name, df, append=true)
    end
    if termination_scip != "INFEASIBLE" && termination_scip != "DUAL_INFEASIBLE"
        return f(vars_scip), vars_scip
    end
end

function build_twotailed_optimizer(o, τ, M)
    MOI.set(o, MOI.Silent(), true)
    n = length(τ)
    x = MOI.add_variables(o, n)
    z = MOI.add_variables(o, n)
    s = MOI.add_variables(o, n)
    MOI.add_constraint.(o, z, MOI.ZeroOne())
    MOI.add_constraint.(o, s, MOI.GreaterThan(0.0))
    MOI.add_constraint.(o, x, MOI.GreaterThan(-M))
    MOI.add_constraint.(o, x, MOI.LessThan(M))
    MOI.add_constraint.(o, s, MOI.LessThan(M))
    for j in 1:n
        MOI.add_constraint(o, 1.0 * s[j] - x[j], MOI.GreaterThan(-τ[j]))
        MOI.add_constraint(o, 1.0 * s[j] + x[j], MOI.GreaterThan(-τ[j]))
        MOI.add_constraint(o,
            MOI.VectorAffineFunction(
                [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[j])), MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, s[j]))],
                [0.0, 0.0],
            ),
            MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(0.0)),
        )
    end
    lmo = FrankWolfe.MathOptLMO(o)

    return lmo, (x, z, s)
end

function build_scip_optimizer(τ, M, limit, f, grad!)
    o = SCIP.Optimizer()
    MOI.set(o, MOI.TimeLimitSec(), limit)
    MOI.set(o, MOI.Silent(), true)
    lmo, (x, z, s) = build_twotailed_optimizer(o, τ, M)
    z_i = MOI.add_variable(lmo.o)
    n = length(τ)
    epigraph_ch = GradientCutHandler(lmo.o, f, grad!, zeros(n+n+n), z_i, vcat(x, z, s), 0)
    SCIP.include_conshdlr(lmo.o, epigraph_ch; needs_constraints=false, name="handler_gradient_cuts")
    MOI.set(lmo.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    
    # lmo to verify feasibility of solution after optimization
    o_check = SCIP.Optimizer()
    lmo_check, _ = build_twotailed_optimizer(o_check, τ, M)
    z_i = MOI.add_variable(lmo_check.o)
    MOI.set(lmo_check.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    
    # println("SCIP MODEL")
    # print(lmo.o)
    return lmo, epigraph_ch, (x, z, s), lmo_check
end

function build_objective_gradient(seed, n)
    Random.seed!(seed)
    p = n
    τ = 6 * rand(p)
    M = 20.0

    # underlying true weights
    ws = rand(Float64, p)
    # set 50 entries to 0
    for _ in 1:20
        ws[rand(1:p)] = 0
    end
    bs = rand(Float64)
    Xs = randn(Float64, n, p)
    ys = map(1:n) do idx
        a = dot(Xs[idx, :], ws) + bs
        return rand(Distributions.Poisson(exp(a)))
    end

    α = 1.3
    function f(θ)
        w = @view(θ[1:p])
        b = θ[end]
        s = sum(1:n) do i
            a = dot(w, Xs[:, i]) + b
            return 1 / n * (exp(a) - ys[i] * a)
        end
        return s + α * norm(w)^2
    end
    function grad!(storage, θ)
        w = @view(θ[1:p])
        b = θ[end]
        storage[1:p] .= 2α .* w
        storage[p+1:2p] .= 0
        storage[end] = 0
        for i in 1:n
            xi = @view(Xs[:, i])
            a = dot(w, xi) + b
            storage[1:p] .+= 1 / n * xi * exp(a)
            storage[1:p] .-= 1 / n * ys[i] * xi
            storage[end] += 1 / n * (exp(a) - ys[i])
        end
        storage ./= norm(storage)
        return storage
    end

    return f, grad!, τ, M
end