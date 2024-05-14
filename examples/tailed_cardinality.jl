using Boscia
using FrankWolfe
using Random
using SCIP
using LinearAlgebra
using Distributions
using DataFrames, CSV
using JuMP
using Ipopt
import MathOptInterface
const MOI = MathOptInterface
using Pavito
using AmplNLWriter, SHOT_jll

include("scip_oa.jl")
include("BnB_Ipopt.jl")

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

function build_objective_gradient(seed, m0)
    @assert m0 >= 10
    Random.seed!(seed)
    n0 = Int(round(m0 / 10))
    @assert n0 > 0
    λ = rand()
    μ = 10.0 * rand()
    A = rand(m0, n0)
    y = rand(m0)
    τ = 6 * rand(n0)
    n = length(τ)
    M = 20.0
    function f(x)
        xv = @view(x[1:n])
        zv = @view(x[n+1:2n])
        return norm(y - A * xv)^2 - λ * sum(zv) + μ * norm(xv)^2
    end

    function grad!(storage, x)
        xv = @view(x[1:n])
        storage .= 0
        @view(storage[1:n]) .= 2 * (transpose(A) * A * xv - transpose(A) * y + μ * xv)
        @view(storage[n+1:2n]) .= -λ
        return storage
    end
    return (f, grad!, n0, m0, τ, M)
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

function tailed_cardinality_sparse_reg_boscia(seed=1, dimension=10, full_callback= false; bo_mode="default", depth=1) 
    limit = 1800
    f, grad!, n0, m0, τ, M = build_objective_gradient(seed, dimension)
    o = SCIP.Optimizer()
    lmo, _ = build_twotailed_optimizer(o, τ, M)

    x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=10, afw=true)

    if bo_mode == "afw"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, variant=Boscia.AwayFrankWolfe())
    ### warmstart_active_set no longer defined on master branch
    elseif bo_mode == "no_as_no_ss"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warm_start=false, use_shadow_set=false)
    elseif bo_mode == "no_as"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warm_start=false, use_shadow_set=true)
    elseif bo_mode == "no_ss"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=true, time_limit=limit, use_shadow_set=false)
    elseif bo_mode == "default"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit)
    elseif bo_mode == "local_tightening"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=true, global_dual_tightening=false) 
    elseif bo_mode == "global_tightening"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=true) 
    elseif bo_mode == "no_tightening"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=false) 
    elseif bo_mode == "strong_branching"
        branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, HiGHS.Optimizer())
        MOI.set(branching_strategy.optimizer, MOI.Silent(), true)

        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, branching_strategy = branching_strategy)
    elseif bo_mode == "hybrid_branching"
        function perform_strong_branch(tree, node)
            return node.level <= length(tree.root.problem.integer_variables)/depth
        end
        branching_strategy = Boscia.HybridStrongBranching(10, 1e-3, HiGHS.Optimizer(), perform_strong_branch)
        MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)

        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, branching_strategy = branching_strategy)
    else
        error("Mode not known!")
    end             

    total_time_in_sec=result[:total_time_in_sec]
    status = result[:status]
    if occursin("Optimal", result[:status])
        status = "OPTIMAL"
    end
    if full_callback
        lb_list = result[:list_lb]
        ub_list = result[:list_ub]
        time_list = result[:list_time]
        list_lmo_calls = result[:list_lmo_calls_acc]
        list_open_nodes = result[:open_nodes]
    end

    if full_callback
        df = DataFrame(seed=seed, dimension=m0, n0=n0, M=M, time= time_list, lowerBound=lb_list, upperBound = ub_list, termination=status, LMOcalls = list_lmo_calls, openNodes=list_open_nodes)
        file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_tailed_cardinality_" * string(dimension) * "_" *string(seed) *".csv")
        CSV.write(file_name, df, append=false)
    else
        df = DataFrame(seed=seed, n0=n0, m0=m0, M=M, time=total_time_in_sec, solution=result[:primal_objective], dual_gap=result[:dual_gap], rel_dual_gap=result[:rel_dual_gap], termination=status, ncalls=result[:lmo_calls])
        if bo_mode == "default" || bo_mode == "local_tightening" || bo_mode == "global_tightening" || bo_mode == "no_tightening" || bo_mode == "afw" || bo_mode == "strong_branching"
            file_name = joinpath(@__DIR__, "csv/boscia_" * bo_mode * "_tailed_cardinality_" * string(dimension) * "_" *string(seed) * ".csv")
        elseif bo_mode == "hybrid_branching"
            file_name = joinpath(@__DIR__, "csv/boscia_" * bo_mode * "_" * string(depth) * "_tailed_cardinality_" * string(dimension) * "_" *string(seed) * ".csv")
        else 
            file_name = joinpath(@__DIR__,"csv/no_warm_start_" * bo_mode * "_tailed_cardinality_" * string(dimension) * "_" *string(seed) * ".csv")
        end
    end
    CSV.write(file_name, df, append=false, writeheader=true)

    # @show x
    # @show f(x) 
    # return f, x

    # xv = @view(x[1:p])
    # predictions = [p > 0.5 ? 1 : -1 for p in A*xv]
    # @show (predictions, y)
    return f(x), x
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

function tailed_cardinality_sparse_reg_scip(seed=1, dimension=10)
    limit = 1800
    f, grad!, n0, m0, τ, M = build_objective_gradient(seed, dimension)
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

        df = DataFrame(seed=seed, n0=n0, m0=m0, M=M, time=time_scip, solution=solution_scip, termination=termination_scip, calls=ncalls_scip)
    else
        time_scip = MOI.get(lmo.o, MOI.SolveTimeSec())
        #@assert sum(ai.*vars_scip) <= bi + 1e-6 # constraint violated
        ncalls_scip = epigraph_ch.ncalls

        df = DataFrame(seed=seed, n0=n0, m0=m0, M=M, time=time_scip, solution=Inf, termination=termination_scip, calls=ncalls_scip)

    end
    file_name = joinpath(@__DIR__,"csv/scip_oa_tailed_cardinality_" * string(dimension) * "_" *string(seed) * ".csv")
    CSV.write(file_name, df, append=false, writeheader=true)
end

# function build_shot_model(τ, M; time_limit=1800)
#     o = Model(() -> AmplNLWriter.Optimizer(SHOT_jll.amplexe))
#     # set_silent(m)
#     set_optimizer_attribute(o, "Termination.TimeLimit", time_limit)
#     set_optimizer_attribute(o, "Output.Console.LogLevel", 3)
#     set_optimizer_attribute(o, "Output.File.LogLevel", 6)
#     set_optimizer_attribute(o, "Termination.ObjectiveGap.Absolute", 1e-6)
#     set_optimizer_attribute(o, "Termination.ObjectiveGap.Relative", 1e-6)

#     n = length(τ)
#     @variable(o, x[1:n])
#     @variable(o, z[1:n], Bin)
#     @variable(o, s[1:n])

#     @constraint(o, s .>= 0)
#     @constraint(o, x .>= -M)
#     @constraint(o, x .<= M)
#     @constraint(o, s .<= M)
#     # MOI.add_constraint.(o, z, MOI.ZeroOne())
#     # MOI.add_constraint.(o, s, MOI.GreaterThan(0.0))
#     # MOI.add_constraint.(o, x, MOI.GreaterThan(-M))
#     # MOI.add_constraint.(o, x, MOI.LessThan(M))
#     # MOI.add_constraint.(o, s, MOI.LessThan(M))
#     for j in 1:n
#         @constraint(o, s[j] - x[j] >= -τ[j])
#         @constraint(o, s[j] + x[j] >= -τ[j])
#         @constraint(o, z[j] --> {s[j] <= 0})
#         # MOI.add_constraint(o, 1.0 * s[j] - x[j], MOI.GreaterThan(-τ[j]))
#         # MOI.add_constraint(o, 1.0 * s[j] + x[j], MOI.GreaterThan(-τ[j]))
#         # MOI.add_constraint(o,
#         #     MOI.VectorAffineFunction(
#         #         [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[j])), MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, s[j]))],
#         #         [0.0, 0.0],
#         #     ),
#         #     MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(0.0)),
#         # )
#     end

#     # TODO: set objective
#     # expr1 = @expression(m, A*x[1:p])
#     # expr2 = @expression(m, dot(x[1:p], x[1:p]))
#     # expr = @expression(m, sum((y[i] - expr1[i])^2 for i in 1:n) + lambda_0*sum(x[i] for i in p+1:2p) + lambda_2*expr2)
#     # @objective(m, Min, expr)

#     return o, o[:x]
# end

# function tailed_cardinality_sparse_reg_shot(seed=1, n=5; time_limit=1800)
#     f, grad!, n0, m0, τ, M = build_objective_gradient(seed, n)
#     # @show f
#     m, x = build_shot_model(τ, M; time_limit=time_limit)
#     @show objective_sense(m)
#     optimize!(m)
#     termination_shot = String(string(MOI.get(m, MOI.TerminationStatus())))

#     if termination_shot != "TIME_LIMIT" && termination_shot != "OPTIMIZE_NOT_CALLED"
#         time_shot = MOI.get(m, MOI.SolveTimeSec())
#         vars_shot = value.(x)
        
#         o_check = SCIP.Optimizer()
#         lmo_check, _ = build_twotailed_optimizer(o_check, τ, M)
#         @assert Boscia.is_linear_feasible(lmo_check.o, vars_shot)

#         solution_shot = f(vars_shot)
#     else 
#         solution_shot = NaN
#         time_shot = time_limit
#     end

#     @show termination_shot, solution_shot


#     df = DataFrame(seed=seed, dimension=n, p=p, k=k, time=time_shot, solution=solution_shot, termination=termination_shot)
#     file_name = joinpath(@__DIR__,"csv/shot_tailed_cardinality_sparse_reg_" * string(seed) * "_" * string(n) * ".csv")
#     CSV.write(file_name, df, append=false, writeheader=true)
# end