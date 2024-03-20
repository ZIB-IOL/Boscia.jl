using Statistics
using Boscia
using FrankWolfe
using Random
using LinearAlgebra
using SCIP
using Pavito
using AmplNLWriter, SHOT_jll
import Bonobo
import MathOptInterface
const MOI = MathOptInterface
using Dates
using Printf
using Test
using DataFrames
using CSV
using Random
include("scip_oa.jl")
include("BnB_Ipopt.jl")

# Sparse logistic regression

# Constant parameters for the sparse regression
# min 1/N ∑ log(1 + exp(-y_i * β @ a_i)) + λ_0 ∑ z_i + μ/2 * ||β||²
# s.t. -Mz_i <= β_i <= Mz_i
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p 

# A - matrix of observations.
# y - vector of results.
# while having relative few non zero entries in β.
# Each continuous variable β_i is assigned a binary z_i,
# z_i = 0 => β_i = 0

function build_function(seed, dimension, var_A)
    Random.seed!(seed)
    n0 = dimension
    p = 5 * n0;
    #k = 5.0#ceil(n0 / 5);
    A = randn(Float64, n0, p)
    y = Random.bitrand(n0)
    y = [i == 0 ? -1 : 1 for i in y]
    for (i,val) in enumerate(y)
        A[i,:] = var_A * A[i,:] * y[i]
    end
    #M = 2 * var(A)
    mu = 10.0 * rand(Float64);
    function build_objective_gradient(A, y, mu)
        # just flexing with unicode
        # reusing notation from Bach 2010 Self-concordant analyis for LogReg
        ℓ(u) = log(exp(u/2) + exp(-u/2))
        dℓ(u) = -1/2 + inv(1 + exp(-u))
        n = length(y)
        invn = inv(n)
        function f(x)
            xv = @view(x[1:p])
            err_term = invn * sum(eachindex(y)) do i # 1/N
                dtemp = dot(A[i,:], xv) # prediction
                ℓ(dtemp) - y[i] * dtemp / 2
            end
            pen_term = mu * dot(xv, xv) / 2
            err_term + pen_term
        end
        function grad!(storage, x)
            storage .= 0 # zeros(length(x))
            xv = @view(x[1:p])
            # @show storage
            for i in eachindex(y)
                dtemp = dot(A[i,:], xv)
                @. storage[1:p] += invn * A[i,:] * (dℓ(dtemp) - y[i] / 2)
            end
            @. storage[1:p] += mu * xv
            storage
        end
        return f, grad!
    end

    f, grad! = build_objective_gradient(A, y, mu)
    # @show A, y, mu

    return f, grad!, p, A, y, mu
end

function build_optimizer(o, p, k, M)
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
        MOI.LessThan(k),
    )
    lmo = FrankWolfe.MathOptLMO(o)

    return lmo, x
end

function sparse_log_reg_boscia(seed=1, dimension=5, M=3, k=5.0, var_A=1.0, full_callback = false; bo_mode="default") 
    limit = 1800

    f, grad!, p, A, y, mu = build_function(seed, dimension, var_A)
    o = SCIP.Optimizer()
    lmo, _ = build_optimizer(o, p, k, M)
    # println("BOSCIA MODEL")
    # print(o)

    x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=10, afw=true)

    if bo_mode == "afw"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, variant=Boscia.AwayFrankWolfe())
    ### warmstart_active_set no longer defined on master branch
    # elseif bo_mode == "no_as_no_ss"
    #     x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=false, use_shadow_set=false)
    # elseif bo_mode == "no_as"
    #     x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=false, warmstart_shadow_set=true)
    elseif bo_mode == "no_ss"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=true, time_limit=limit, use_shadow_set=false)
    elseif bo_mode == "default"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, print_iter=1)
    elseif bo_mode == "local_tightening"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=true, global_dual_tightening=false, print_iter=1) 
    elseif bo_mode == "global_tightening"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=true, print_iter=1) 
    elseif bo_mode == "no_tightening"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=false, print_iter=1) 
    elseif bo_mode == "local_tightening_no_ss"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=true, global_dual_tightening=false, use_shadow_set=false, print_iter=1) 
    elseif bo_mode == "global_tightening_no_ss"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=true, use_shadow_set=false,print_iter=1) 
    elseif bo_mode == "no_tightening_no_ss"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=false, use_shadow_set=false, print_iter=1) 
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
        list_local_tightening = result[:local_tightenings]
        list_global_tightening = result[:global_tightenings]
    end

    if full_callback
        df = DataFrame(seed=seed, dimension=dimension, var_A=var_A, p=p, k=k, M=M, time= time_list, lowerBound= lb_list, upperBound = ub_list, termination=status, LMOcalls = list_lmo_calls, openNodes=list_open_nodes, localTighteings=list_local_tightening, globalTightenings=list_global_tightening)
        file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_sparse_log_regression_" * string(dimension) * "_" * string(M) * "-" * string(var_A) * "_" *string(seed) *".csv")
        CSV.write(file_name, df, append=false)
    else
        df = DataFrame(seed=seed, dimension=dimension, var_A=var_A, p=p, k=k, M=M, time=total_time_in_sec, solution=result[:primal_objective], dual_gap = result[:dual_gap], rel_dual_gap=result[:rel_dual_gap], termination=status, ncalls=result[:lmo_calls])
        file_name = joinpath(@__DIR__,"csv/boscia_" * bo_mode * "_sparse_log_reg_" * string(seed) * "_" * string(dimension) * "_" * string(var_A) * "_" * string(p) * "_" * string(M) * ".csv")
    end
    CSV.write(file_name, df, append=false, writeheader=true)
    return f(x), x
end

function build_pavito_model(n, k, var_A, M, p, A, y, mu; time_limit=1800)
    m = Model(
        optimizer_with_attributes(
            Pavito.Optimizer,
            "mip_solver" => optimizer_with_attributes(
                SCIP.Optimizer, 
                "limits/maxorigsol" => 10000,
                "numerics/feastol" => 1e-6,
                "display/verblevel" => 0,
            ),
            "cont_solver" => optimizer_with_attributes(
                Ipopt.Optimizer, 
                "print_level" => 0,
                "tol" => 1e-6,
            ),
        ),
    )  
    MOI.set(m, MOI.TimeLimitSec(), time_limit)
    set_silent(m)

    @variable(m, x[1:2p])
    for i in p+1:2p
        #@constraint(m, 1 >= x[i] >= 0)
        set_binary(x[i])
    end

    for i in 1:p
        @constraint(m, x[i] + M*x[i+p] >= 0)
        @constraint(m, x[i] - M*x[i+p] <= 0)
    end
    lbs = vcat(fill(-M, p), fill(0.0,p))
    ubs = vcat(fill(M, p), fill(1.0, p))

    @constraint(m, sum(x[p+1:2p]) <= k)

    expr1 = @expression(m, mu/2*sum(x[i]^2 for i in 1:p))
    lexprs = []
    for i in 1:n
        push!(lexprs, @expression(m, dot(x[1:p], A[i, :])))
    end
    exprs = []
    for i in 1:n
        push!(exprs, @NLexpression(m, exp(lexprs[i]/2) + exp(-lexprs[i]/2)))
    end 
    expr = @NLexpression(m, 1/n * sum(log(exprs[i]) - y[i] * lexprs[i] * 1/2 for i in 1:n ) + expr1)
    @NLobjective(m, Min, expr)

    return m, m[:x]
end

function sparse_log_reg_pavito(seed=1, dimension=10, M=3, k=5.0, var_A=1.0; print_models=false, time_limit=1800)
    f, grad!, p, A, y, mu = build_function(seed, dimension, var_A)
    # @show f
    m, x = build_pavito_model(dimension, k, var_A, M, p, A, y, mu; time_limit=time_limit)
    if print_models
        println("PAVITO")
        println(m)
    end
    @show objective_sense(m)
    optimize!(m)
    termination_pavito = String(string(MOI.get(m, MOI.TerminationStatus())))

    if termination_pavito != "TIME_LIMIT" && termination_pavito != "OPTIMIZE_NOT_CALLED"
        time_pavito = MOI.get(m, MOI.SolveTimeSec())
        vars_pavito = value.(x)
        
        o_check = SCIP.Optimizer()
        lmo_check, _ = build_optimizer(o_check,p, k, M)
        @assert Boscia.is_linear_feasible(lmo_check.o, vars_pavito)

        solution_pavito = f(vars_pavito)
    else 
        solution_pavito = NaN
        time_pavito = time_limit
    end

    @show termination_pavito, solution_pavito

    df = DataFrame(seed=seed, dimension=dimension, var_A=var_A, p=p, k=k, M=M, time=time_pavito, solution=solution_pavito, termination=termination_pavito)
    file_name = joinpath(@__DIR__,"csv/pavito_sparse_log_reg_" * string(seed) * "_" * string(dimension) * "_" * string(var_A) * "_" * string(p) * "_" * string(M) * ".csv")
    CSV.write(file_name, df, append=false, writeheader=true)
end

function build_shot_model(seed, n, k, var_A, M, p, A, y, mu; time_limit=1800)
    m = Model(() -> AmplNLWriter.Optimizer(SHOT_jll.amplexe))
    # set_silent(m)
    set_optimizer_attribute(m, "Termination.TimeLimit", time_limit)
    set_optimizer_attribute(m, "Output.Console.LogLevel", 3)
    set_optimizer_attribute(m, "Output.File.LogLevel", 6)
    set_optimizer_attribute(m, "Termination.ObjectiveGap.Absolute", 1e-6)
    set_optimizer_attribute(m, "Termination.ObjectiveGap.Relative", 1e-6)

    @variable(m, x[1:2p])
    for i in p+1:2p
        #@constraint(m, 1 >= x[i] >= 0)
        set_binary(x[i])
    end

    for i in 1:p
        @constraint(m, x[i] + M*x[i+p] >= 0)
        @constraint(m, x[i] - M*x[i+p] <= 0)
    end
    lbs = vcat(fill(-M, p), fill(0.0,p))
    ubs = vcat(fill(M, p), fill(1.0, p))

    @constraint(m, sum(x[p+1:2p]) <= k)

    expr1 = @expression(m, mu/2*sum(x[i]^2 for i in 1:p))
    lexprs = []
    for i in 1:n
        push!(lexprs, @expression(m, dot(x[1:p], A[i, :])))
    end
    exprs = []
    for i in 1:n
        push!(exprs, @NLexpression(m, exp(lexprs[i]/2) + exp(-lexprs[i]/2)))
    end 
    expr = @NLexpression(m, 1/n * sum(log(exprs[i]) - y[i] * lexprs[i] * 1/2 for i in 1:n ) + expr1)
    @NLobjective(m, Min, expr)

    return m, m[:x]
end

function sparse_log_reg_shot(seed=1, dimension=10, M=3, k=5.0, var_A=1.0; time_limit=1800)
    f, grad!, p, A, y, mu = build_function(seed, dimension, var_A)
    # @show f
    m, x = build_shot_model(seed, dimension, k, var_A, M, p, A, y, mu; time_limit=time_limit)
    @show objective_sense(m)
    optimize!(m)
    termination_shot = String(string(MOI.get(m, MOI.TerminationStatus())))

    if termination_shot != "TIME_LIMIT" && termination_shot != "OPTIMIZE_NOT_CALLED"
        time_shot = MOI.get(m, MOI.SolveTimeSec())
        vars_shot = value.(x)
        
        o_check = SCIP.Optimizer()
        lmo_check, _ = build_optimizer(o_check,p, k, M)
        @assert Boscia.is_linear_feasible(lmo_check.o, vars_shot)

        solution_shot = f(vars_shot)
    else 
        solution_shot = NaN
        time_shot = time_limit
    end

    @show termination_shot, solution_shot

    df = DataFrame(seed=seed, dimension=dimension, var_A=var_A, p=p, k=k, M=M, time=time_shot, solution=solution_shot, termination=termination_shot)
    file_name = joinpath(@__DIR__,"csv/shot_sparse_log_reg_" * string(seed) * "_" * string(dimension) * "_" * string(var_A) * "_" * string(p) * "_" * string(M) * ".csv")
    CSV.write(file_name, df, append=false, writeheader=true)
end

function build_scip_optimizer(p, k, M, limit, f, grad!)
    o = SCIP.Optimizer()
    MOI.set(o, MOI.TimeLimitSec(), limit)
    MOI.set(o, MOI.Silent(), true)
    lmo, x = build_optimizer(o, p, k, M)
    z_i = MOI.add_variable(lmo.o)
    epigraph_ch = GradientCutHandler(o, f, grad!, zeros(length(x)), z_i, x, 0)
    SCIP.include_conshdlr(lmo.o, epigraph_ch; needs_constraints=false, name="handler_gradient_cuts")
    MOI.set(lmo.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    
    # lmo to verify feasibility of solution after optimization
    o_check = SCIP.Optimizer()
    lmo_check, _ = build_optimizer(o_check, p, k, M)
    z_i = MOI.add_variable(lmo_check.o)
    MOI.set(lmo_check.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    return lmo, epigraph_ch, x, lmo_check
end

function sparse_log_reg_scip(seed=1, dimension=10, M=3, k=5.0, var_A=0.5)
    limit = 1800
    f, grad!, p, A, y, mu  = build_function(seed, dimension, var_A)
    lmo, epigraph_ch, x, lmo_check = build_scip_optimizer(p, k, M, limit, f, grad!)

    MOI.set(lmo.o, MOI.TimeLimitSec(), limit)
    MOI.optimize!(lmo.o)
    termination_scip = String(string(MOI.get(lmo.o, MOI.TerminationStatus())))
    
    if termination_scip != "INFEASIBLE"
        time_scip = MOI.get(lmo.o, MOI.SolveTimeSec())
        vars_scip = MOI.get(lmo.o, MOI.VariablePrimal(), x)
        @assert Boscia.is_linear_feasible(lmo_check.o, vars_scip)
        solution_scip = f(vars_scip)
        ncalls_scip = epigraph_ch.ncalls

        df = DataFrame(seed=seed, dimension=dimension, var_A=var_A, p=p, k=k, M=M, time=time_scip, x=[vars_scip], solution=solution_scip, termination=termination_scip, calls=ncalls_scip)
    else
        time_scip = MOI.get(o, MOI.SolveTimeSec())
        ncalls_scip = epigraph_ch.ncalls

        df = DataFrame(seed=seed, dimension=dimension, var_A=var_A, p=p, k=k, M=M, time=time_scip, x=[Inf], solution=Inf, termination=termination_scip, calls=ncalls_scip)

    end
    file_name = joinpath(@__DIR__,"csv/scip_oa_sparse_log_reg_" * string(seed) * "_" * string(dimension) * "_" * string(var_A) * "_" * string(p) * "_" * string(M) * ".csv")
    CSV.write(file_name, df, append=false, writeheader=true, delim=";")
    return f(vars_scip), vars_scip
end

# build tree 
function build_bnb_ipopt_model(n, M, k, var_A, p, A, y, mu; time_limit)
    m = Model(Ipopt.Optimizer)
    set_silent(m)

    @variable(m, x[1:2p])
    for i in p+1:2p
        @constraint(m, 1 >= x[i] >= 0)
    end

    for i in 1:p
        @constraint(m, x[i] + M*x[i+p] >= 0)
        @constraint(m, x[i] - M*x[i+p] <= 0)
    end
    lbs = vcat(fill(-M, p), fill(0.0,p))
    ubs = vcat(fill(M, p), fill(1.0, p))

    @constraint(m, sum(x[p+1:2p]) <= k)

    expr1 = @expression(m, mu/2*sum(x[i]^2 for i in 1:p))
    lexprs = []
    for i in 1:n
        push!(lexprs, @expression(m, dot(x[1:p], A[i, :])))
    end
    exprs = []
    for i in 1:n
        push!(exprs, @NLexpression(m, exp(lexprs[i]/2) + exp(-lexprs[i]/2)))
    end 
    expr = @NLexpression(m, 1/n * sum(log(exprs[i]) - y[i] * lexprs[i] * 1/2 for i in 1:n ) + expr1)
    @NLobjective(m, Min, expr)

    model = IpoptOptimizationProblem(collect(p+1:2p), m, Boscia.SOLVING, time_limit, lbs, ubs)
    bnb_model = BB.initialize(;
    traverse_strategy = BB.BestFirstSearch(),
    Node = MIPNode,
    root = model,
    sense = objective_sense(m) == MOI.MAX_SENSE ? :Max : :Min,
    rtol = 1e-2,
    )
    BB.set_root!(bnb_model, (
    lbs = fill(-Inf, length(x)),#zeros(length(x)),
    ubs = fill(Inf, length(x)),
    status = MOI.OPTIMIZE_NOT_CALLED)
    )
    return bnb_model, expr, p, k
end

# BnB tree with Ipopt
function sparse_log_reg_ipopt(seed=1, n=10, M=3, k=5.0, var_A=1.0; time_limit=1800)
    f, grad!, p, A, y, mu = build_function(seed, n, var_A)

    # build tree
    bnb_model, expr = build_bnb_ipopt_model(n, M, k, var_A, p, A, y, mu; time_limit=time_limit)
    list_lb = []
    list_ub = []
    list_time = []
    list_number_nodes = []
    callback = build_callback(list_lb, list_ub, list_time, list_number_nodes)
    time_ref = Dates.now()
    data = @timed BB.optimize!(bnb_model, callback=callback)
    push!(list_lb, bnb_model.lb)
    push!(list_ub, bnb_model.incumbent)
    push!(list_time, float(Dates.value(Dates.now()-time_ref)))
    push!(list_number_nodes, bnb_model.num_nodes)
    total_time_in_sec= list_time[end]
    status = ""
    if bnb_model.root.solving_stage == Boscia.TIME_LIMIT_REACHED
        status = "Time limit reached"
    else
        status = "Optimal"
    end    

    df = DataFrame(seed=seed, dimension=n, var_A=var_A, p=p, k=k, M=M, time=total_time_in_sec, num_nodes = bnb_model.num_nodes, solution=bnb_model.incumbent, termination=status)
    file_name = joinpath(@__DIR__,"csv/ipopt_sparse_log_reg_" * string(seed) * "_" * string(n) * "_" * string(var_A) * "_" * string(p) * "_" * string(M) * ".csv")
    CSV.write(file_name, df, append=false, writeheader=true)
end



