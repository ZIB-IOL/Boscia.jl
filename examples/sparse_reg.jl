using Boscia
using FrankWolfe
using Random
using SCIP
using LinearAlgebra
using Distributions
using DataFrames, CSV
using JuMP
using Ipopt
using HiGHS
import MathOptInterface
const MOI = MathOptInterface
using Pavito
using AmplNLWriter, SHOT_jll

include("scip_oa.jl")
include("BnB_Ipopt.jl")

# Sparse regression

# Constant parameters for the sparse regression
# min norm(y-A β)² + λ_0 ∑ z_i + λ_2 ||β||²
# s.t. -Mz_i <= β_i <= Mz_i
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p 

# A - matrix of observations.
# y - vector of results.
# We want to match Aβ as closely as possible to y 
# while having relative few non zero entries in β.
# Each continuous variable β_i is assigned a binary z_i,
# z_i = 0 => β_i = 0

function build_function(seed, n)
    Random.seed!(seed)
    p = 5 * n;
    k = ceil(n / 5);
    lambda_0 = rand(Float64);
    lambda_2 = 10.0 * rand(Float64);
    A = rand(Float64, n, p)
    y = rand(Float64, n)
    M = 2 * var(A)
    # @show A, y, M, lambda_0, lambda_2

    function f(x)
        xv = @view(x[1:p])
        return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
    end

    function grad!(storage, x)
        storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
        storage[p+1:2p] .= lambda_0
        return storage
    end

    return f, grad!, p, k, M, A, y, lambda_0, lambda_2
end

function build_optimizer(o, p, k, M)
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, 2p)
    for i in p+1:2p
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, x[i], MOI.ZeroOne()) # or MOI.Integer()
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

function sparse_reg_boscia(seed=1, n=5, full_callback = false; bo_mode="default", depth=1, write=true)
    limit = 1800

    f, grad!, p, k, M, A, y, lambda_0, lambda_2 = build_function(seed, n)
    o = SCIP.Optimizer()
    lmo, _ = build_optimizer(o, p, k, M)
    # println(o)
    println("precompile")
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
   # print(lmo.o)
    println("actual run")
    if bo_mode == "afw"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, variant=Boscia.AwayFrankWolfe(), use_postsolve=false)
    ### warmstart_active_set no longer defined on master branch
    elseif bo_mode == "no_as_no_ss"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warm_start=false, use_shadow_set=false, use_postsolve=false)
    elseif bo_mode == "no_as"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warm_start=false, use_shadow_set=true, use_postsolve=false)
    elseif bo_mode == "no_ss"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=true, time_limit=limit, use_shadow_set=false, use_postsolve=false)
    elseif bo_mode == "default"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=true, time_limit=limit, print_iter=1, use_postsolve=false)
    elseif bo_mode == "local_tightening"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=true, global_dual_tightening=false, print_iter=1, use_postsolve=false) 
    elseif bo_mode == "global_tightening"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=true, print_iter=1, use_postsolve=false) 
    elseif bo_mode == "no_tightening"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=false, print_iter=1, use_postsolve=false) 
    elseif bo_mode == "local_tightening_no_ss"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=true, global_dual_tightening=false, use_shadow_set=false, print_iter=1) 
    elseif bo_mode == "global_tightening_no_ss"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=true, use_shadow_set=false,print_iter=1) 
    elseif bo_mode == "no_tightening_no_ss"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=false, use_shadow_set=false, print_iter=1) 
    elseif bo_mode == "strong_branching"
        blmo = Boscia.MathOptBLMO(HiGHS.Optimizer())
        branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
        MOI.set(branching_strategy.bounded_lmo.o, MOI.Silent(), true)

        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, branching_strategy = branching_strategy)
    elseif bo_mode == "hybrid_branching"
        function perform_strong_branch(tree, node)
            return node.level <= length(tree.root.problem.integer_variables)/depth
        end
        blmo = Boscia.MathOptBLMO(HiGHS.Optimizer())
        branching_strategy = Boscia.HybridStrongBranching(10, 1e-3, blmo, perform_strong_branch)
        MOI.set(branching_strategy.pstrong.bounded_lmo.o, MOI.Silent(), true)

        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, branching_strategy = branching_strategy, use_postsolve=false)
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
        list_active_set_size_cb = result[:list_active_set_size] 
        list_discarded_set_size_cb = result[:list_discarded_set_size]
        list_local_tightening = result[:local_tightenings]
        list_global_tightening = result[:global_tightenings]
        df = DataFrame(seed=seed, dimension=n, time=time_list, lowerBound= lb_list, upperBound = ub_list, termination=status, LMOcalls = list_lmo_calls, localTighteings=list_local_tightening, globalTightenings=list_global_tightening, list_active_set_size_cb=list_active_set_size_cb,list_discarded_set_size_cb=list_discarded_set_size_cb)
        file_name = joinpath(@__DIR__, "final_csvs/boscia_" * bo_mode * "_" * string(n) * "_" *string(seed) * "_sparse_reg.csv")
        CSV.write(file_name, df, append=false)
    else
        if write
        @show result[:primal_objective]
        df = DataFrame(seed=seed, dimension=n, p=p, k=k, time=total_time_in_sec, solution=result[:primal_objective], dual_gap =result[:dual_gap], rel_dual_gap=result[:rel_dual_gap], termination=status, ncalls=result[:lmo_calls])
        if bo_mode=="default" || bo_mode=="local_tightening" || bo_mode=="global_tightening" || bo_mode=="no_tightening" || bo_mode=="afw" || bo_mode == "strong_branching"
            file_name = joinpath(@__DIR__,"csv/boscia_" * bo_mode * "_sparse_reg_" * string(seed) * "_" * string(n) * ".csv")
        elseif bo_mode == "hybrid_branching"
            file_name = joinpath(@__DIR__,"csv/boscia_" * bo_mode * "_" * string(depth) * "_sparse_reg_" * string(seed) * "_" * string(n) * ".csv")
        else 
            file_name = joinpath(@__DIR__,"csv/no_warm_start_" * bo_mode * "_sparse_reg_" * string(seed) * "_" * string(n) * ".csv")
        end
        CSV.write(file_name, df, append=false, writeheader=true)
    end
    end
end

function build_pavito_model(n, p, k, M, A, y, lambda_0, lambda_2; time_limit = 1800)
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

    expr1 = @expression(m, A*x[1:p])
    expr2 = @expression(m, dot(x[1:p], x[1:p]))
    expr = @expression(m, sum((y[i] - expr1[i])^2 for i in 1:n) + lambda_0*sum(x[i] for i in p+1:2p) + lambda_2*expr2)
    @objective(m, Min, expr)

    return m, m[:x]
end

function sparse_reg_pavito(seed=1, n=5; time_limit=1800)
    f, grad!, p, k, M, A, y, lambda_0, lambda_2 = build_function(seed, n)
    # @show f
    m, x = build_pavito_model(n, p, k, M, A, y, lambda_0, lambda_2; time_limit=time_limit)

    @show objective_sense(m)
    optimize!(m)
    termination_pavito = String(string(MOI.get(m, MOI.TerminationStatus())))

    if termination_pavito != "TIME_LIMIT" && termination_pavito != "OPTIMIZE_NOT_CALLED"
        time_pavito = MOI.get(m, MOI.SolveTimeSec())
        vars_pavito = value.(x)
        @assert Boscia.is_linear_feasible(m.moi_backend, vars_pavito)    
        solution_pavito = f(vars_pavito)
    else 
        solution_pavito = NaN
        time_pavito = time_limit
    end

    @show termination_pavito, solution_pavito

    df = DataFrame(seed=seed, dimension=n, p=p, k=k, time=time_pavito, solution=solution_pavito, termination=termination_pavito)
    file_name = joinpath(@__DIR__,"csv/pavito_sparse_reg_" * string(seed) * "_" * string(n) * ".csv")

    CSV.write(file_name, df, append=false, writeheader=true)
end

function build_shot_model(n, p, k, M, A, y, lambda_0, lambda_2; time_limit = 1800)
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

    expr1 = @expression(m, A*x[1:p])
    expr2 = @expression(m, dot(x[1:p], x[1:p]))
    expr = @expression(m, sum((y[i] - expr1[i])^2 for i in 1:n) + lambda_0*sum(x[i] for i in p+1:2p) + lambda_2*expr2)
    @objective(m, Min, expr)

    return m, m[:x]
end

function sparse_reg_shot(seed=1, n=5; time_limit=1800)
    f, grad!, p, k, M, A, y, lambda_0, lambda_2 = build_function(seed, n)
    # @show f
    m, x = build_shot_model(n, p, k, M, A, y, lambda_0, lambda_2; time_limit=time_limit)
    @show objective_sense(m)
    optimize!(m)
    termination_shot = String(string(MOI.get(m, MOI.TerminationStatus())))

    if termination_shot != "TIME_LIMIT" && termination_shot != "OPTIMIZE_NOT_CALLED"
        time_shot = MOI.get(m, MOI.SolveTimeSec())
        vars_shot = value.(x)
        
        o_check = SCIP.Optimizer()
        lmo_check, _ = build_optimizer(o_check, p, k, M)
        @assert Boscia.is_linear_feasible(lmo_check.o, vars_shot)

        solution_shot = f(vars_shot)
    else 
        solution_shot = NaN
        time_shot = time_limit
    end

    @show termination_shot, solution_shot


    df = DataFrame(seed=seed, dimension=n, p=p, k=k, time=time_shot, solution=solution_shot, termination=termination_shot)
    file_name = joinpath(@__DIR__,"csv/shot_sparse_reg_" * string(seed) * "_" * string(n) * ".csv")
    CSV.write(file_name, df, append=false, writeheader=true)
end

function build_scip_optimizer(p, k, M, limit, f, grad!)
    o = SCIP.Optimizer()
    MOI.set(o, MOI.TimeLimitSec(), limit)
    MOI.set(o, MOI.Silent(), true)
    lmo, x = build_optimizer(o, p, k, M)
    print(lmo.o)
    z_i = MOI.add_variable(lmo.o)
    epigraph_ch = GradientCutHandler(lmo.o, f, grad!, zeros(length(x)), z_i, x, 0)
    SCIP.include_conshdlr(lmo.o, epigraph_ch; needs_constraints=false, name="handler_gradient_cuts")
    MOI.set(lmo.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    
    # lmo to verify feasibility of solution after optimization
    o_check = SCIP.Optimizer()
    lmo_check, _ = build_optimizer(o_check, p, k, M)
    z_i = MOI.add_variable(lmo_check.o)
    MOI.set(lmo_check.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    
    return lmo, epigraph_ch, x, lmo_check
end

function sparse_reg_scip(seed=1, n=5; tol=1e-6)
    limit = 1800
    f, grad!, p, k, M, A, y, lambda_0, lambda_2 = build_function(seed, n)

    lmo, epigraph_ch, x, lmo_check = build_scip_optimizer(p, k, M, limit, f, grad!)

    MOI.optimize!(lmo.o)
    time_scip = MOI.get(lmo.o, MOI.SolveTimeSec())
    vars_scip = MOI.get(lmo.o, MOI.VariablePrimal(), x)
    @assert Boscia.is_linear_feasible(lmo_check.o, vars_scip)
    solution_scip = f(vars_scip)
    @show solution_scip
    termination_scip = String(string(MOI.get(lmo.o, MOI.TerminationStatus())))
    ncalls_scip = epigraph_ch.ncalls

    df = DataFrame(seed=seed, dimension=n, p=p, k=k, time=time_scip, solution=solution_scip, termination=termination_scip, calls=ncalls_scip)
    file_name = joinpath(@__DIR__,"csv/scip_oa_sparse_reg_" * string(seed) * "_" * string(n) * ".csv")
    CSV.write(file_name, df, append=false, writeheader=true)
end

function build_bnb_ipopt_model(n, p, k, M, A, y, lambda_0, lambda_2)
    time_limit = 1800
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

    expr1 = @expression(m, A*x[1:p])
    expr2 = @expression(m, dot(x[1:p], x[1:p]))
    expr = @expression(m, sum((y[i] - expr1[i])^2 for i in 1:n) + lambda_0*sum(x[i] for i in p+1:2p) + lambda_2*expr2)
    @objective(m, Min, expr)

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
    return bnb_model, expr
end

function sparse_reg_ipopt(seed=1, n=5; full_callback=false)
    f, grad!, p, k, M, A, y, lambda_0, lambda_2 = build_function(seed, n)
    # build tree
    bnb_model, expr = build_bnb_ipopt_model(n, p, k, M, A, y, lambda_0, lambda_2)
    # print(bnb_model.root.m)    
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

    @show status, bnb_model.incumbent
    if full_callback
        df = DataFrame(seed=seed, num_v=num_v,number_nodes = bnb_model.num_nodes, time=list_time, lowerBound = list_lb, upperBound = list_ub, termination=status,)
        file_name =joinpath(@__DIR__,"csv/ipopt_sparse_reg_" * string(seed) * "_" * string(n) *  ".csv")
        CSV.write(file_name, df, append=false)
    else
    df = DataFrame(seed=seed, dimension=n, p=p, k=k, time=total_time_in_sec, num_nodes = bnb_model.num_nodes, solution=bnb_model.incumbent, termination=status)
    file_name = joinpath(@__DIR__,"csv/ipopt_sparse_reg_" * string(seed) * "_" * string(n) * ".csv")
    CSV.write(file_name, df, append=false, writeheader=true)
    end
end

function sparse_reg_grid_search()
    example = "int_sparsereg"

    seed=2 # 19 (29 is too long!) 30
    Random.seed!(seed)
    # n=10 seed = 1 produces good example

    n =50
    m = 80
    l = 5
    k = 15

    sol_x = rand(1:l, n)
    for _ in 1:(n-k)
        sol_x[rand(1:n)] = 0
    end

    #=k=0 # correct k
    for i in 1:n
        if sol_x[i] == 0 
            global k += 1
        end
    end
    k = n-k =#

    D = rand(m,n)
    y_d = D*sol_x

    # @testset "Integer sparse regression" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o,n)
    z = MOI.add_variables(o,n)
    for i in 1:n
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0*l))
        MOI.add_constraint(o, x[i], MOI.Integer())

        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())

        MOI.add_constraint(o, 1.0 * x[i] - 1.0 * l * z[i], MOI.LessThan(0.0))
    end 
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0*k))
    # MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(zeros(n),x), sum(Float64.(iszero.(x)))), MOI.GreaterThan(1.0*(n-k)))
    # MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n),z), 0.0), MOI.GreaterThan(1.0*k))
    lmo = FrankWolfe.MathOptLMO(o)

    function f(x)
        xv = @view(x[1:n])
        return 1/2 * sum(abs2, y_d - D * xv)  #+ lambda_2*FrankWolfe.norm(x)^2 + lambda_0*sum(x[p+1:2p])
    end

    function grad!(storage, x)
        storage .= 0
        @view(storage[1:n]) .= transpose(D)* (D*@view(x[1:n]) - y_d)
        return storage
    end

    #= function perform_strong_branch(tree, node)
        return node.level <= length(tree.root.problem.integer_variables)
    end
    branching_strategy = Boscia.HybridStrongBranching(10, 1e-3, HiGHS.Optimizer(), perform_strong_branch)
    MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)=#

    #val_min, x_min = Boscia.sparse_min_via_enum(f, n, k, fill(0:l, n))
    #@show x_min
    # @show x[1:n]
    # @show x_min
    # @test val_min == f(x)
    # @test isapprox(x[1:n], x_min)
    # @test isapprox(f(x), f(result[:raw_solution]), atol = 1e-6, rtol = 1e-6)

    values = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
    fw_epsilon_values = [1e-3, 5e-3, 1e-4, 1e-7]
    min_num_lower_values = [20, 40, 60, 80, 100, 200, Inf]
    seeds = [30]#,2,3]

    Boscia.solve(f, grad!, lmo; verbose=true)

    iter = 3
    for (seed_idx, seed_val) in enumerate(seeds)
        for (index,value) in enumerate(values)
            for i in 1:iter
                for (idx,eps) in enumerate(fw_epsilon_values)
                    for (idx2, min_num_lower_val) in enumerate(min_num_lower_values)
                        seed = seed_val
                        dual_gap_decay_factor = value
                        min_number_lower = min_num_lower_val
                        fw_epsilon = eps
                        data = @timed sol, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, dual_gap_decay_factor=dual_gap_decay_factor, min_number_lower=min_number_lower, fw_epsilon = fw_epsilon, print_iter=1)
                        @show f(sol)
                        df = DataFrame(seed=seed, dimension=n, min_number_lower=min_number_lower, adaptive_gap=dual_gap_decay_factor, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size])
                        file_name = "experiments/final_csvs/early_stopping_" * example * "_" * string(n) * "_" * string(seed) * "_" * string(min_number_lower) * "_" * string(dual_gap_decay_factor) * "_" * string(fw_epsilon) * "_" * string(i) *".csv"
                        CSV.write(file_name, df, append=false)
                    end
                end
            end
        end
    end
end