using Boscia
using FrankWolfe
using Random
using SCIP
using Pavito
# using Statistics
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface
using CSV
using DataFrames
using Ipopt
include("scip_oa.jl")
include("BnB_Ipopt.jl")

function sparse_reg(seed=1, n=20, iter = 1, full_callback = false; bo_mode)
    limit = 1800
    @show seed, n, iter

    f, grad!, p, k, M = build_function(seed, n)
    @show f
    o = SCIP.Optimizer()
    lmo, _ = build_optimizer(o, p, k, M)
    # println(lmo.o)
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)

    for i in 1:iter
        if bo_mode == "afw"
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=true, time_limit=limit, afw=true)
        elseif bo_mode == "as_ss"
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=false, warmstart_shadow_set=false)
        elseif bo_mode == "as"
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=false, warmstart_shadow_set=true)
        elseif bo_mode == "ss"
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=true, warmstart_shadow_set=false)
        elseif bo_mode == "boscia"
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit)
        elseif bo_mode == "local_tightening"
            x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=true, global_dual_tightening=false) 
        elseif bo_mode == "global_tightening"
            x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=true) 
        elseif bo_mode == "no_tightening"
            x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=false) 
        end             
        # @show x, f(x)
        # @test dot(ai, x) <= bi + 1e-6
        # @test f(x) <= f(result[:raw_solution]) + 1e-6
        total_time_in_sec=result[:total_time_in_sec]
        status = result[:status]
        @show status, result[:primal_objective]
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
            df = DataFrame(seed=seed, dimension=n, p=p, k=k, time= time_list, lowerBound= lb_list, upperBound = ub_list, termination=status, LMOcalls = list_lmo_calls, openNodes=list_open_nodes, localTighteings=list_local_tightening, globalTightenings=list_global_tightening)
            file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_sparse_reg_" * string(n) * "_" *string(seed) *".csv")
            CSV.write(file_name, df, append=false)
        else
            df = DataFrame(seed=seed, dimension=n, p=p, k=k, time=total_time_in_sec, solution=result[:primal_objective], dual_gap =result[:dual_gap], rel_dual_gap=result[:rel_dual_gap], termination=status, ncalls=result[:lmo_calls])
            if bo_mode ==  "afw"
                file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_sparse_reg.csv")
            elseif bo_mode == "boscia" || bo_mode == "local_tightening" || bo_mode == "global_tightening" || bo_mode == "no_tightening"
                file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_sparse_reg.csv")
            else 
                file_name = joinpath(@__DIR__,"csv/no_warm_start_" * bo_mode * "_sparse_reg.csv")
            end
            if !isfile(file_name)
                CSV.write(file_name, df, append=true, writeheader=true)
            else 
                CSV.write(file_name, df, append=true)
            end
        end
        # display(df)
    end
end

function sparse_reg_scip(seed=1, n=20, iter = 1; tol=1e-6)
    limit = 1800

    f, grad!, p, k, M = build_function(seed, n)

    for i in 1:iter
        lmo, epigraph_ch, x, lmo_check = build_scip_optimizer(p, k, M, limit, f, grad!)

        # MOI.set(o, MOI.AbsoluteGapTolerance(), 1.000000e-06) #AbsoluteGapTolerance not defined
        # MOI.set(o, MOI.RelativeGapTolerance(), 1.000000e-02)
        MOI.optimize!(lmo.o)
        time_scip = MOI.get(lmo.o, MOI.SolveTimeSec())
        vars_scip = MOI.get(lmo.o, MOI.VariablePrimal(), x)
        @assert Boscia.is_linear_feasible(lmo_check.o, vars_scip)
        #@assert sum(ai.*vars_scip) <= bi + 1e-6 # constraint violated
        solution_scip = f(vars_scip)
        termination_scip = String(string(MOI.get(lmo.o, MOI.TerminationStatus())))
        ncalls_scip = epigraph_ch.ncalls

        df = DataFrame(seed=seed, dimension=n, p=p, k=k, time=time_scip, solution=solution_scip, termination=termination_scip, calls=ncalls_scip)
        file_name = joinpath(@__DIR__,"csv/scip_oa_sparse_reg_ " * string(tol) * ".csv")
        if !isfile(file_name)
            CSV.write(file_name, df, append=true, writeheader=true)
        else 
            CSV.write(file_name, df, append=true)
        end
    end
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

function build_scip_optimizer(p, k, M, limit, f, grad!)
    o = SCIP.Optimizer()
    MOI.set(o, MOI.TimeLimitSec(), limit)
    MOI.set(o, MOI.Silent(), true)
    lmo, x = build_optimizer(o, p, k, M)
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

function build_function(seed, n)
    Random.seed!(seed)
    p = 5 * n;
    k = ceil(n / 5);
    lambda_0 = rand(Float64);
    lambda_2 = 10.0 * rand(Float64);
    A = rand(Float64, n, p)
    y = rand(Float64, n)
    M = 2 * var(A)

    function f(x)
        xv = @view(x[1:p])
        return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
    end

    function grad!(storage, x)
        storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
        storage[p+1:2p] .= lambda_0
        return storage
    end

    return f, grad!, p, k, M
end

# BnB tree with Ipopt
function sparse_reg_ipopt(seed = 1, n = 20, iter = 1)
    # build tree
    bnb_model, expr, p, k = build_bnb_ipopt_model(seed, n)
    print(bnb_model.root.m)    
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
    df = DataFrame(seed=seed, dimension=n, p=p, k=k, time=total_time_in_sec, num_nodes = bnb_model.num_nodes, solution=bnb_model.incumbent, termination=status)
    file_name = joinpath(@__DIR__,"csv/ipopt_sparse_reg_ " * ".csv")
    if !isfile(file_name)
        CSV.write(file_name, df, append=true, writeheader=true)
    else 
        CSV.write(file_name, df, append=true)
    end
end

# build tree 
function build_bnb_ipopt_model(seed, n)
    Random.seed!(seed)
    time_limit = 1800

    p = 5 * n;
    k = ceil(n / 5);
    lambda_0 = rand(Float64);
    lambda_2 = 10.0 * rand(Float64);
    A = rand(Float64, n, p)
    y = rand(Float64, n)
    M = 2 * var(A)

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
    return bnb_model, expr, p, k

end

# compare big M with indicator formulation
function build_sparse_reg(dim, fac, seed, use_indicator, time_limit, rtol)
    example = "sparse_reg"
    Random.seed!(seed)

    p = dim # p continuous
   # integral = 2*p # num integer vars
    k = ceil(p/fac) 
    lambda_0 = rand(Float64); lambda_2 = 10.0*rand(Float64);
    A = rand(Float64, dim, p)
    y = rand(Float64, dim)
    M = 2*var(A)

    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, p)
    z = MOI.add_variables(o, p)
    for i in 1:p
        MOI.add_constraint(o, x[i], MOI.GreaterThan(-M))
        MOI.add_constraint(o, x[i], MOI.LessThan(M))

        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne()) 
    end 
    for i in 1:p
        if use_indicator 
            # Indicator: x[i+p] = 1 => x[i] = 0
            # Beware: SCIP can only handle MOI.ACTIVATE_ON_ONE with LessThan constraints.
            # Hence, in the indicator formulation, we ahve zi = 1 => xi = 0. (In bigM zi = 0 => xi = 0)
            gl = MOI.VectorAffineFunction(
                [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                    MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[i])),],
                [0.0, 0.0], )
            gg = MOI.VectorAffineFunction(
                [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                    MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, x[i])),],
                [0.0, 0.0], )
            MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(0.0)))
            MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(0.0)))
        else
            # big M
            MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,M], [x[i], z[i]]), 0.0), MOI.GreaterThan(0.0))
            MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-M], [x[i], z[i]]), 0.0), MOI.LessThan(0.0))
        end
        
    end
    if use_indicator
        MOI.add_constraint(o, sum(z, init=0.0), MOI.GreaterThan(1.0 * (p-k))) # we want less than k zeros
    else
        MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    end
    lmo = FrankWolfe.MathOptLMO(o)


    function f(x)
        return (sum((y-A*x[1:p]).^2) + lambda_0*sum(x[p+1:2p]) + lambda_2*FrankWolfe.norm(x[1:p])^2)/10000
    end
    function grad!(storage, x)
        storage.=vcat(2*(transpose(A)*A*x[1:p] - transpose(A)*y + lambda_2*x[1:p]), lambda_0*ones(p))./10000
        return storage
    end

    iter =1
    x = zeros(2p)
    for i in 1:iter
        indicator = use_indicator ? "indicator" : "bigM"
        data = @timed x, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, print_iter = 100, time_limit = time_limit, rel_dual_gap = rtol, dual_gap = 1e-4, use_postsolve = false, fw_epsilon = 1e-2, min_node_fw_epsilon =1e-5)
        df = DataFrame(seed=seed, dimension=dim, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size])
        file_name = "csv/bigM_vs_indicator_" * example * "_" * indicator * "_" * string(dim) * "_" * string(fac) * "_" * string(seed) * ".csv"
        CSV.write(file_name, df, append=false)
    end
    return x, f(x)
end

function sparse_reg_grid_search_data()
    example = "int_sparsereg"

    seed=2 # 19 (29 is too long!) 30
    Random.seed!(seed)
    # n=10 seed = 1 produces good example

    n = 40
    m = 60
    l = 3
    k = 10

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
                        file_name = "csv/early_stopping_" * example * "_" * string(n) * "_" * string(seed) * "_" * string(min_number_lower) * "_" * string(dual_gap_decay_factor) * "_" * string(fw_epsilon) * "_" * string(i) *".csv"
                        CSV.write(file_name, df, append=false)
                        @show file_name
                    end
                end
            end
        end
    end
end


function strong_branching_data()
    example = "int_sparse_reg"

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
    
    D = rand(m,n)
    y_d = D*sol_x
    
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
    
        MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-1.0*l], [x[i], z[i]]), 0.0), MOI.LessThan(0.0))
    end 
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n),z), 0.0), MOI.LessThan(1.0*k))
    lmo = FrankWolfe.MathOptLMO(o)
    
    function f(x)
        return sum((y_d-D*x[1:n]).^2)  #+ lambda_2*FrankWolfe.norm(x)^2 + lambda_0*sum(x[p+1:2p]) 
    end
        
    function grad!(storage, x)
        storage.=vcat(2*(transpose(D)*D*x[1:n] - transpose(D)*y_d), zeros(n))  #vcat(..,zeros(n))
        return storage
    end
    
    
    min_number_lower = Inf
    dual_gap_decay_factor = 0.8
    Boscia.solve(f, grad!, lmo; verbose=true, min_number_lower=min_number_lower, fw_epsilon = 1e-3, print_iter=10)
    
    ## Different strong branching
    iterations = [5, 10, 50, 100]
    iter = 2
    for (index,value) in enumerate(iterations)
        for i in 0:iter
            max_iter = value
            branching_strategy = Boscia.PartialStrongBranching(max_iter, 10^i*1e-3, SCIP.Optimizer())
            MOI.set(branching_strategy.optimizer, MOI.Silent(), true)

            Boscia.solve(f, grad!, lmo; verbose=true, branching_strategy = branching_strategy, dual_gap_decay_factor=dual_gap_decay_factor, min_number_lower=min_number_lower, fw_epsilon = 1e-4, print_iter=10)
            data = @timed _, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, branching_strategy = branching_strategy, dual_gap_decay_factor=dual_gap_decay_factor, min_number_lower=min_number_lower, fw_epsilon = 1e-4, print_iter=10)
            df = DataFrame(seed=seed, dimension=n, min_number_lower=min_number_lower, adaptive_gap=dual_gap_decay_factor, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size], node_level = result[:node_level])
            file_name = "csv/strong_branching_" * example * "_" * string(n) * "_" * string(seed) * "_" * string(max_iter) * "1e-"*string(3-i)*".csv"
            CSV.write(file_name, df, append=false)
        end
    end 
end

function hybrid_branching_data()
    example = "int_sparse_reg"

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
    
    D = rand(m,n)
    y_d = D*sol_x
    
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
    
        MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-1.0*l], [x[i], z[i]]), 0.0), MOI.LessThan(0.0))
    end 
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n),z), 0.0), MOI.LessThan(1.0*k))
    lmo = FrankWolfe.MathOptLMO(o)
    
    function f(x)
        return sum((y_d-D*x[1:n]).^2)  #+ lambda_2*FrankWolfe.norm(x)^2 + lambda_0*sum(x[p+1:2p]) 
    end
        
    function grad!(storage, x)
        storage.=vcat(2*(transpose(D)*D*x[1:n] - transpose(D)*y_d), zeros(n))  #vcat(..,zeros(n))
        return storage
    end
    
    
    min_number_lower = Inf
    dual_gap_decay_factor = 0.8
    Boscia.solve(f, grad!, lmo; verbose=true, min_number_lower=min_number_lower, fw_epsilon = 1e-3, print_iter=10, time_limit=10)
    
    ## Different hybrid branching
    for i in [1, 5, 10, 20]
        function perform_strong_branch(tree, node)
            node.level <= length(tree.root.problem.integer_variables)/i ? println("Strong") : println("most infeasible")
            return node.level <= length(tree.root.problem.integer_variables)/i
        end
        branching_strategy = Boscia.HybridStrongBranching(5, 1e-3, SCIP.Optimizer(), perform_strong_branch)
        MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)

        Boscia.solve(f, grad!, lmo; verbose=true, branching_strategy = branching_strategy, dual_gap_decay_factor=dual_gap_decay_factor, min_number_lower=min_number_lower, fw_epsilon = 1e-4, print_iter=10, time_limit=10)
        data = @timed _, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, branching_strategy = branching_strategy, dual_gap_decay_factor=dual_gap_decay_factor, min_number_lower=min_number_lower, fw_epsilon = 1e-4, print_iter=10)
        df = DataFrame(seed=seed, dimension=n, min_number_lower=min_number_lower, adaptive_gap=dual_gap_decay_factor, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size], node_level = result[:node_level])
        file_name = "csv/hybrid_branching_" * example * "_" * string(n) * "_" * string(seed) * "_num_integer_dividedby" * string(i) * ".csv"
        @show file_name
        CSV.write(file_name, df, append=false)
    end
end 

function sparse_reg_pavito(seed=1, n=20; print_models=false)
    f, grad!, p, k, M = build_function(seed, n)
    # @show f
    m, x = build_pavito_model(n, p, k, seed)
    if print_models
        println("PAVITO")
        println(m)
    end
    @show objective_sense(m)
    optimize!(m)
    time_pavito = MOI.get(m, MOI.SolveTimeSec())
    vars_pavito = value.(x)
    @assert Boscia.is_linear_feasible(m.moi_backend, vars_pavito)    
    #@assert sum(ai.*vars_scip) <= bi + 1e-6 # constraint violated
    solution_pavito = f(vars_pavito)
    termination_pavito = String(string(MOI.get(m, MOI.TerminationStatus())))

    @show termination_pavito, solution_pavito
    df = DataFrame(seed=seed, dimension=n, p=p, k=k, time=time_pavito, solution=solution_pavito, termination=termination_pavito)
    file_name = joinpath(@__DIR__,"csv/pavito_sparse_reg_" * string(seed) * "_" * string(n) * ".csv")

    # # check feasibility in Ipopt model
    # ipopt_model, _, _, _ = build_bnb_ipopt_model(seed, n)
    # if print_models
    #     println("IPOPT")
    #     print(ipopt_model.root.m)
    # end
    # @show objective_sense(ipopt_model.root.m)
    # key_vector = ipopt_model.root.m[:x]
    # point = Dict(key_vector .=> vars_pavito)
    # report = primal_feasibility_report(ipopt_model.root.m, point, atol=1e-6)
    # @assert isempty(report)
    # BB.optimize!(ipopt_model)
    # @show ipopt_model.incumbent
    # # writedlm("report.txt", report)

    # check feasibility in Boscia model
    o = SCIP.Optimizer()
    lmo, _ = build_optimizer(o, p, k, M)
    if print_models
        println("BOSCIA")
        print(o)
    end
    # check linear feasiblity
    @assert Boscia.is_linear_feasible(lmo, vars_pavito)
    # check integer feasibility
    integer_variables = Vector{Int}()
    for cidx in MOI.get(lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}())
        push!(integer_variables, cidx.value)
    end
    for idx in integer_variables
        @assert isapprox(vars_pavito[idx], round(vars_pavito[idx]); atol=1e-6, rtol=1e-6)
    end
    # check feasibility of rounded solution
    vars_pavito_polished = vars_pavito
    for i in integer_variables
        vars_pavito_polished[i] = round(vars_pavito_polished[i])
    end
    @assert Boscia.is_linear_feasible(lmo, vars_pavito_polished)
    # solve Boscia
    x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=1800, dual_tightening=true, global_dual_tightening=true, rel_dual_gap=1e-6, fw_epsilon=1e-6)
    @show result[:dual_bound]

    # evaluate soluton of each solver
    # vids = MOI.get(ipopt_model.root.m, MOI.ListOfVariableIndices())
    # vars = VariableRef.(ipopt_model.root.m, vids)
    # solution_ipopt = value.(vars)
    solution_boscia = result[:raw_solution]
    #@show f(vars_pavito), f(solution_ipopt), f(solution_boscia)
    @show f(vars_pavito), f(solution_boscia)
    if occursin("Optimal", result[:status])
        @assert result[:dual_bound] <= f(vars_pavito) + 1e-5
    end

    if !isfile(file_name)
        CSV.write(file_name, df, append=true, writeheader=true)
    else 
        CSV.write(file_name, df, append=true)
    end
end

function build_pavito_model(n, p, k, seed)
    Random.seed!(seed)
    time_limit = 1800 

    lambda_0 = rand(Float64);
    lambda_2 = 10.0 * rand(Float64);
    A = rand(Float64, n, p)
    y = rand(Float64, n)
    M = 2 * var(A)

    m = Model(
        optimizer_with_attributes(
            Pavito.Optimizer,
            "mip_solver" => optimizer_with_attributes(
                SCIP.Optimizer, 
                "limits/maxorigsol" => 10000,
                "numerics/feastol" => 1e-6,
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
