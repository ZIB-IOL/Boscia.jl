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

function poisson(seed=1, n=20, Ns=0.1, iter = 1; bo_mode)
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
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit)
        end             
        # @show x, f(x)
        # @test dot(ai, x) <= bi + 1e-6
        # @test f(x) <= f(result[:raw_solution]) + 1e-6
        total_time_in_sec=result[:total_time_in_sec]
        status = result[:status]
        if occursin("Optimal", result[:status])
            status = "OPTIMAL"
        end
        df = DataFrame(seed=seed, dimension=n, p=p, k=k, Ns=Ns, time=total_time_in_sec, solution=result[:primal_objective], termination=status, ncalls=result[:lmo_calls])
        if bo_mode ==  "afw"
            file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_poisson.csv")
        elseif bo_mode == "boscia"
            file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_poisson.csv")
        else 
            file_name = joinpath(@__DIR__,"csv/no_warm_start_" * bo_mode * "_poisson.csv")
        end
        if !isfile(file_name)
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
    data = @timed BB.optimize!(bnb_model, callback=callback)
    time_ref = Dates.now()
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

    df = DataFrame(seed=seed, dimension=n, p=p, k=k, time=total_time_in_sec, num_nodes = bnb_model.num_nodes, solution=bnb_model.incumbent, termination=status)
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

    #expr1 = @expression(m, α*sum(x[i]^2 for i in 1:p))
    #expr2 = @expression(m, exp(dot(x[1:p], Xs(:, i)) + x[end]) for i in 1:p)
    #expr = @NLexpression(m, sum(1/n * (exp(dot(x[1:p], Xs(:, i)) + x[end]) - ys[i]*(dot(x[1:p], Xs[:, i] + x[end]))) for i in 1:p) + expr1)
    @NLobjective(m, Min, expr)

    model = IpoptOptimizationProblem(collect(p+1:2p), m, Boscia.SOLVING, time_limit, lbs, ubs)
    bnb_model = BB.initialize(;
    traverse_strategy = BB.BFS(),
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
