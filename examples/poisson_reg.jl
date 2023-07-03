using Boscia
using FrankWolfe
using Random
using SCIP
# using Statistics
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface
using CSV
using DataFrames
include("scip_oa.jl")
include("BnB_Ipopt.jl")

# Poisson sparse regression

# min_{w, b, z} ∑_i exp(w x_i + b) - y_i (w x_i + b) + α norm(w)^2

# y_i    - data points, poisson distributed 
# X_i, b - coefficient for the linear estimation of the expected value of y_i
# w_i    - continuous variables
# z_i    - binary variables s.t. z_i = 0 => w_i = 0
# k      - max number of non zero entries in w

# In a poisson regression, we want to model count data.
# It is assumed that y_i is poisson distributed and that the log 
# of its expected value can be computed linearly.

# poisson(1,10,1.0,1,true;bo_mode="boscia") assertionError
function poisson(seed=1, n=20, Ns=0.1, iter = 1, full_callback=false; bo_mode)
    limit = 1800

    f, grad!, p = build_function(seed, n)
    k = n/2
    o = SCIP.Optimizer()
    lmo, _ = build_optimizer(o, p, k, Ns)

    for i in 1:iter
        if bo_mode == "afw"
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, afw=true)
        elseif bo_mode == "as_ss"
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=false, warmstart_shadow_set=false)
        elseif bo_mode == "as"
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=false, warmstart_shadow_set=true)
        elseif bo_mode == "ss"
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=true, warmstart_shadow_set=false)
        elseif bo_mode == "boscia"
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=true, time_limit=limit)
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
            df = DataFrame(seed=seed, dimension=n, p=p, k=k, Ns=Ns, time= time_list, lowerBound= lb_list, upperBound = ub_list, termination=status, LMOcalls = list_lmo_calls, openNodes=list_open_nodes, localTighteings=list_local_tightening, globalTightenings=list_global_tightening)
            file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_poisson_" * string(n) * "_" * string(Ns) * "-" * string(p) * "_"  * string(k) * "_" * string(seed) * ".csv")
            CSV.write(file_name, df, append=false)
        else
            df = DataFrame(seed=seed, dimension=n, p=p, k=k, Ns=Ns, time=total_time_in_sec, solution=result[:primal_objective], dual_gap =result[:dual_gap], rel_dual_gap=result[:rel_dual_gap], termination=status, ncalls=result[:lmo_calls])
            if bo_mode ==  "afw"
                file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_poisson.csv")
            elseif bo_mode == "boscia" || bo_mode == "local_tightening" || bo_mode == "global_tightening" || bo_mode == "no_tightening"
                file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_poisson.csv")
            else 
                file_name = joinpath(@__DIR__,"csv/no_warm_start_" * bo_mode * "_poisson.csv")
            end
        end
        if !isfile(file_name) & !full_callback
            CSV.write(file_name, df, append=true, writeheader=true)
        else 
            CSV.write(file_name, df, append=true)
        end
        # display(df)
    end
end

function poisson_scip(seed=1, n=20, Ns=0.1,iter = 1;)
    limit = 1800
    f, grad!, p = build_function(seed, n)
    k = n/2

    for i in 1:iter
        lmo, epigraph_ch, x, lmo_check = build_scip_optimizer(p, k, Ns, limit, f, grad!)
        MOI.set(lmo.o, MOI.TimeLimitSec(), limit)
        # MOI.set(o, MOI.AbsoluteGapTolerance(), 1.000000e-06) #AbsoluteGapTolerance not defined
        # MOI.set(o, MOI.RelativeGapTolerance(), 1.000000e-02)
        MOI.optimize!(lmo.o)
        termination_scip = String(string(MOI.get(lmo.o, MOI.TerminationStatus())))
        if termination_scip != "INFEASIBLE"
            time_scip = MOI.get(lmo.o, MOI.SolveTimeSec())
            vars_scip = MOI.get(lmo.o, MOI.VariablePrimal(), x)
            #@assert sum(ai.*vars_scip) <= bi + 1e-6 # constraint violated
            solution_scip = f(vars_scip)
            ncalls_scip = epigraph_ch.ncalls
            @assert Boscia.is_linear_feasible(lmo_check.o, vars_scip)

            df = DataFrame(seed=seed, dimension=n, p=p, k=k, Ns=Ns, time=time_scip, solution=solution_scip, termination=termination_scip, calls=ncalls_scip)
        else
            time_scip = MOI.get(lmo.o, MOI.SolveTimeSec())
            #@assert sum(ai.*vars_scip) <= bi + 1e-6 # constraint violated
            ncalls_scip = epigraph_ch.ncalls
 
            df = DataFrame(seed=seed, dimension=n, p=p, k=k, Ns=Ns, time=time_scip, solution=Inf, termination=termination_scip, calls=ncalls_scip)

        end
        file_name = joinpath(@__DIR__,"csv/scip_oa_poisson.csv")
        if !isfile(file_name)
            CSV.write(file_name, df, append=true, writeheader=true)
        else 
            CSV.write(file_name, df, append=true)
        end
    end
end

function build_optimizer(o, p, k, Ns)
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    w = MOI.add_variables(o, p)
    z = MOI.add_variables(o, p)
    b = MOI.add_variable(o)
    # z_i ∈ {0,1} for i = 1,..,p
    for i in 1:p
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())
    end
    for i in 1:p
        # s.t. -N z_i <= w_i <= N z_i
        MOI.add_constraint(o, Ns * z[i] + w[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, -Ns * z[i] + w[i], MOI.LessThan(0.0))
        # Indicator: z[i] = 1 => -N <= w[i] <= N
        #=gl = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),],
            [0.0, 0.0], )
        gg = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),],
            [0.0, 0.0], )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(Ns)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-Ns))) =#
    end
    # ∑ z_i <= k 
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(o, sum(z, init=0.0), MOI.GreaterThan(1.0))
    # b ∈ [-N, N]
    MOI.add_constraint(o, b, MOI.LessThan(Ns))
    MOI.add_constraint(o, b, MOI.GreaterThan(-Ns))
    lmo = FrankWolfe.MathOptLMO(o)
    return lmo, (w,z,b)
end

function build_scip_optimizer(p, k, M, limit, f, grad!)
    o = SCIP.Optimizer()
    MOI.set(o, MOI.TimeLimitSec(), limit)
    MOI.set(o, MOI.Silent(), true)
    lmo, (w,z,b) = build_optimizer(o, p, k, M)
    z_i = MOI.add_variable(lmo.o)
    epigraph_ch = GradientCutHandler(o, f, grad!, zeros(p+p+1), z_i, vcat(w,z,b), 0)
    SCIP.include_conshdlr(lmo.o, epigraph_ch; needs_constraints=false, name="handler_gradient_cuts")
    MOI.set(lmo.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    
    # lmo to verify feasibility of solution after optimization
    o_check = SCIP.Optimizer()
    lmo_check, _ = build_optimizer(o_check, p, k, M)
    z_i = MOI.add_variable(lmo_check.o)
    MOI.set(lmo_check.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    
    return lmo, epigraph_ch, vcat(w,z,b), lmo_check
end

function build_function(seed, n)
    Random.seed!(seed)
    p = n

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

    return f, grad!, p
end

# BnB tree with Ipopt
function poisson_ipopt(seed = 1, n = 20, Ns = 1.0, iter = 1)
    # build tree
    bnb_model, expr, p, k = build_bnb_ipopt_model(seed, n, Ns)
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

    df = DataFrame(seed=seed, dimension=n, p=p, Ns=Ns, k=k, time=total_time_in_sec, num_nodes = bnb_model.num_nodes, solution=bnb_model.incumbent, termination=status)
    file_name = joinpath(@__DIR__,"csv/ipopt_poisson_reg_ " * ".csv")
    if !isfile(file_name)
        CSV.write(file_name, df, append=true, writeheader=true)
    else 
        CSV.write(file_name, df, append=true)
    end
end

# build tree 
function build_bnb_ipopt_model(seed, n, Ns)
    Random.seed!(seed)
    time_limit = 1800

    p = n
    k = n/2

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

    m = Model(Ipopt.Optimizer)
    set_silent(m)

    @variable(m, x[1:2p+1])
    for i in p+1:2p
        @constraint(m, 1 >= x[i] >= 0)
    end

    for i in 1:p
        @constraint(m, x[i] + Ns*x[i+p] >= 0)
        @constraint(m, x[i] - Ns*x[i+p] <= 0)
    end

    @constraint(m, x[2p+1] <= Ns)
    @constraint(m, x[2p+1] >= -Ns)
    lbs = vcat(fill(-Ns, p), fill(0.0,p), [-Ns])
    ubs = vcat(fill(Ns, p), fill(1.0, p), [Ns])

    @constraint(m, sum(x[p+1:2p]) <= k)
    @constraint(m, sum(x[p+1:2p]) >= 1.0)

    expr1 = @expression(m, α*sum(x[i]^2 for i in 1:p))
    exprs = []
    for i in 1:p
        push!(exprs, @expression(m, dot(x[1:p], Xs[:, i]) + x[end]))
    end
    expr = @NLexpression(m, 1/n * sum(exp(exprs[i]) - ys[i] * exprs[i] for i in 1:p ) + expr1)
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

function poisson_scip(seed=1, n=20, Ns=0.1,iter = 1;)
    limit = 1800
    Random.seed!(seed)
    p = n
    k = n/2
    α = 1.3
    function build_function()
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
        return f, grad!
    end
    f, grad! = build_function()
    function build_scip_optimizer()
        o = SCIP.Optimizer()
        MOI.set(o, MOI.Silent(), true)
        MOI.empty!(o)
        w = MOI.add_variables(o, p)
        z = MOI.add_variables(o, p)
        b = MOI.add_variable(o)
        for i in 1:p
            MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
            MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
            MOI.add_constraint(o, z[i], MOI.ZeroOne())
        end
        for i in 1:p
            MOI.add_constraint(o, Ns * z[i] + w[i], MOI.GreaterThan(0.0))
            MOI.add_constraint(o, -Ns * z[i] + w[i], MOI.LessThan(0.0))
            # Indicator: z[i] = 1 => -N <= w[i] <= N
            #=gl = MOI.VectorAffineFunction(
                [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                    MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),],
                [0.0, 0.0], )
            gg = MOI.VectorAffineFunction(
                [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                    MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),],
                [0.0, 0.0], )
            MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(Ns)))
            MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-Ns))) =#
        end
        MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
        MOI.add_constraint(o, sum(z, init=0.0), MOI.GreaterThan(1.0))
        MOI.add_constraint(o, b, MOI.LessThan(Ns))
        MOI.add_constraint(o, b, MOI.GreaterThan(-Ns))
        lmo = FrankWolfe.MathOptLMO(o)
        
        z_i = MOI.add_variable(o)
        
        epigraph_ch = GradientCutHandler(o, f, grad!, zeros(p+p+1), z_i, vcat(w,z,b), 0)
        SCIP.include_conshdlr(o, epigraph_ch; needs_constraints=false, name="handler_gradient_cuts")
        
        MOI.set(o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
        # println("SCIP MODEL")
        # print(o)
        return o, epigraph_ch, vcat(w,z,b)
    end

    for i in 1:iter
        o, epigraph_ch, x = build_scip_optimizer()
        MOI.set(o, MOI.TimeLimitSec(), limit)
        # MOI.set(o, MOI.AbsoluteGapTolerance(), 1.000000e-06) #AbsoluteGapTolerance not defined
        # MOI.set(o, MOI.RelativeGapTolerance(), 1.000000e-02)
        MOI.optimize!(o)
        termination_scip = String(string(MOI.get(o, MOI.TerminationStatus())))
        if termination_scip != "INFEASIBLE"
            time_scip = MOI.get(o, MOI.SolveTimeSec())
            vars_scip = MOI.get(o, MOI.VariablePrimal(), x)
            #@assert sum(ai.*vars_scip) <= bi + 1e-6 # constraint violated
            solution_scip = f(vars_scip)
            ncalls_scip = epigraph_ch.ncalls

            df = DataFrame(seed=seed, dimension=n, p=p, k=k, Ns=Ns, time=time_scip, solution=solution_scip, termination=termination_scip, calls=ncalls_scip)
        else
            time_scip = MOI.get(o, MOI.SolveTimeSec())
            #@assert sum(ai.*vars_scip) <= bi + 1e-6 # constraint violated
            ncalls_scip = epigraph_ch.ncalls

            df = DataFrame(seed=seed, dimension=n, p=p, k=k, Ns=Ns, time=time_scip, solution=Inf, termination=termination_scip, calls=ncalls_scip)

        end
        file_name = joinpath(@__DIR__,"csv/scip_oa_poisson.csv")
        if !isfile(file_name)
            CSV.write(file_name, df, append=true, writeheader=true)
        else 
            CSV.write(file_name, df, append=true)
        end
    end
end

# poisson(1, 30, bo_mode="afw")
# poisson_scip(1, 30)

# compare big M with indicator formulation
function build_poisson_reg(dim, fac, seed, use_indicator, time_limit, rtol)
    example = "poisson_reg"
    Random.seed!(seed)

    p = dim
    n = p
    k = ceil(p/fac)

    # underlying true weights
    ws = rand(Float64, p)
    # set 50 entries to 0
    for _ in 1:(p-k)
        ws[rand(1:p)] = 0
    end
    bs = rand(Float64)
    Xs = randn(Float64, n, p)
    ys = map(1:n) do idx
        a = dot(Xs[idx, :], ws) + bs
        return rand(Distributions.Poisson(exp(a)))
    end
    M = 1.0

    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, p)
    z = MOI.add_variables(o, p)
    b = MOI.add_variable(o)
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
    MOI.add_constraint(o, b, MOI.LessThan(M))
    MOI.add_constraint(o, b, MOI.GreaterThan(-M))
    lmo = FrankWolfe.MathOptLMO(o)


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

    iter =1
    x = zeros(2p+1)
    for i in 1:iter
        indicator = use_indicator ? "indicator" : "bigM"
        data = @timed x, time_lmo, result = Boscia.solve(f, grad!, lmo; print_iter = 100, verbose=true, time_limit = time_limit, rel_dual_gap = rtol, dual_gap = 1e-4, use_postsolve = false, fw_epsilon = 1e-2, min_node_fw_epsilon =1e-5, min_fw_iterations = 2)
        df = DataFrame(seed=seed, dimension=dim, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size])
        file_name = "csv/bigM_vs_indicator_" * example * "_" * indicator * "_" * string(dim) * "_" * string(fac) * "_" * string(seed) * ".csv"
        CSV.write(file_name, df, append=false)
    end
    result = f(x)
    return x, result
end

