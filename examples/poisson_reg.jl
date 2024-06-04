using Boscia
using FrankWolfe
using Random
using SCIP
using Pavito
using HiGHS
using AmplNLWriter, SHOT_jll
# using Statistics
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface
using CSV
using DataFrames

include("scip_oa.jl")
include("BnB_Ipopt.jl")

# min_{w, b, z} ∑_i exp(w x_i + b) - y_i (w x_i + b) + α norm(w)^2
# s.t. -N z_i <= w_i <= N z_i
# b ∈ [-N, N]
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p

# y_i    - data points, poisson distributed 
# X_i, b - coefficient for the linear estimation of the expected value of y_i
# w_i    - continuous variables
# z_i    - binary variables s.t. z_i = 0 => w_i = 0
# k      - max number of non zero entries in w

# In a poisson regression, we want to model count data.
# It is assumed that y_i is poisson distributed and that the log 
# of its expected value can be computed linearly.

function build_function(seed, n; Ns=0.0, use_scale=false)
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
    for j in 1:p 
        Xs[:,j] ./= (maximum(Xs[:,j]) - minimum(Xs[:,j]))
        if Ns == 10.0
            Xs[:,j] .*= 0.1
        end
    end
    ys = map(1:n) do idx
        a = dot(Xs[idx, :], ws) + bs
        return rand(Distributions.Poisson(exp(a)))
    end

    α = 1.3
    scale = exp(n/2)
    function f(θ)
        #θ = BigFloat.(θ)
        w = @view(θ[1:p])
        b = θ[end]
        s = sum(1:n) do i
            a = dot(w, Xs[:, i]) + b
            return 1 / n * (exp(a) - ys[i] * a)
        end
        if use_scale
            return 1/scale * (s + α * norm(w)^2)
        end
        return s + α * norm(w)^2
    end
    function grad!(storage, θ)
        #θ = BigFloat.(θ)
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
        if use_scale
            storage .*= 1/scale
        end
        return storage
    end
    # @show bs, Xs, ys, ws

    return f, grad!, p, α, bs, Xs, ys, ws
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

function poisson_reg_boscia(seed=1, n=20, Ns=0.1, full_callback=false; bo_mode="default", depth=1, limit=1800)
    #limit = 1800
use_scale = false
    f, grad!, p, α, bs, Xs, ys, ws = build_function(seed, n; Ns=Ns)
    k = n/2
    o = SCIP.Optimizer()
    lmo, _ = build_optimizer(o, p, k, Ns)
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)

    if bo_mode == "afw"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=true, time_limit=limit, variant=Boscia.AwayFrankWolfe())
    ### warmstart_active_set no longer defined on master branch
    elseif bo_mode == "no_as_no_ss"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warm_start=false, use_shadow_set=false)
    elseif bo_mode == "no_as"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warm_start=false, use_shadow_set=true)
    elseif bo_mode == "no_ss"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=true, time_limit=limit, use_shadow_set=false)
    elseif bo_mode == "default"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=true, time_limit=limit, print_iter=100, use_postsolve=false, fw_verbose=false)
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
        list_local_tightening = result[:local_tightenings]
        list_global_tightening = result[:global_tightenings]
    end

    if full_callback
        df = DataFrame(seed=seed, dimension=n, p=p, k=k, Ns=Ns, time= time_list, lowerBound= lb_list, upperBound = ub_list, termination=status, LMOcalls = list_lmo_calls, openNodes=list_open_nodes, localTighteings=list_local_tightening, globalTightenings=list_global_tightening)
        file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_poisson_" * string(n) * "_" * string(Ns) * "-" * string(p) * "_"  * string(k) * "_" * string(seed) * ".csv")
        CSV.write(file_name, df, append=false)
    else
        primal = use_scale ? result[:primal_objective] * exp(n/2) : result[:primal_objective]
        dual_gap = use_scale ? result[:dual_gap] * exp(n/2) : result[:dual_gap]
        @show primal, dual_gap, primal - dual_gap
        df = DataFrame(seed=seed, dimension=n, p=p, k=k, Ns=Ns, time=total_time_in_sec, solution=primal, dual_gap =dual_gap, rel_dual_gap=result[:rel_dual_gap], termination=status, ncalls=result[:lmo_calls])
        if bo_mode == "default" || bo_mode == "local_tightening" || bo_mode == "global_tightening" || bo_mode == "no_tightening" || bo_mode=="afw" || bo_mode == "strong_branching"
            file_name = joinpath(@__DIR__, "csv/boscia_" * bo_mode * "_poisson_reg_"  * string(seed) * "_" * string(n) *  "_" * string(k) * "_"  * string(Ns) * ".csv")
        elseif bo_mode == "hybrid_branching"
            file_name = joinpath(@__DIR__, "csv/boscia_" * bo_mode * "_" * string(depth) * "_poisson_reg_"  * string(seed) * "_" * string(n) *  "_" * string(k) * "_"  * string(Ns) * ".csv")
        else
            file_name = joinpath(@__DIR__,"csv/no_warm_start_" * bo_mode * "_poisson_reg_" * string(seed) * "_" * string(n) *  "_" * string(k) * "_"  * string(Ns) * ".csv")
        end
    end
    println(file_name)
    CSV.write(file_name, df, append=false, writeheader=true)
end

function build_pavito_model(n, Ns, p, k, α, bs, Xs, ys, ws; time_limit=1800)
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

    @variable(m, x[1:2p+1])
    for i in p+1:2p
        #@constraint(m, 1 >= x[i] >= 0)
        set_binary(x[i])
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

    return m, m[:x]
end

function poisson_reg_pavito(seed=1, n=20, Ns=0.1; print_models=false, time_limit=1800)
    f, grad!, p, α, bs, Xs, ys, ws = build_function(seed, n, Ns=Ns)
    k = n/2
    # @show f
    m, x = build_pavito_model(n, Ns, p, k, α, bs, Xs, ys, ws; time_limit=time_limit)
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
        lmo_check, _ = build_optimizer(o_check, p, k, Ns)
        @assert Boscia.is_linear_feasible(lmo_check.o, vars_pavito)

        solution_pavito = f(vars_pavito)
    else 
        solution_pavito = NaN
        time_pavito = time_limit
    end

    @show termination_pavito, solution_pavito
    df = DataFrame(seed=seed, dimension=n, p=p, k=k, Ns=Ns, time=time_pavito, solution=solution_pavito, termination=termination_pavito)
    file_name = joinpath(@__DIR__,"csv/pavito_poisson_reg_" * string(seed) * "_" * string(n) *  "_" * string(k) * "_"  * string(Ns) * ".csv")
    CSV.write(file_name, df, append=false, writeheader=true)
end


function build_shot_model(n, Ns, p, k, α, bs, Xs, ys, ws; time_limit=1800)
    m = Model(() -> AmplNLWriter.Optimizer(SHOT_jll.amplexe))
    # set_silent(m)
    set_optimizer_attribute(m, "Termination.TimeLimit", time_limit)
    set_optimizer_attribute(m, "Output.Console.LogLevel", 3)
    set_optimizer_attribute(m, "Output.File.LogLevel", 6)
    set_optimizer_attribute(m, "Termination.ObjectiveGap.Absolute", 1e-6)
    set_optimizer_attribute(m, "Termination.ObjectiveGap.Relative", 1e-6)

    @variable(m, x[1:2p+1])
    for i in p+1:2p
        #@constraint(m, 1 >= x[i] >= 0)
        set_binary(x[i])
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

    return m, m[:x]
end

function poisson_reg_shot(seed=1, n=20, Ns=0.1; time_limit=1800)
    f, grad!, p, α, bs, Xs, ys, ws = build_function(seed, n, Ns=Ns)
    k = n/2
    # @show f
    m, x = build_shot_model(n, Ns, p, k, α, bs, Xs, ys, ws; time_limit=time_limit)
    @show objective_sense(m)
    optimize!(m)
    termination_shot = String(string(MOI.get(m, MOI.TerminationStatus())))
    if termination_shot != "TIME_LIMIT" && termination_shot != "OPTIMIZE_NOT_CALLED" && termination_shot != "OTHER_ERROR"
        time_shot = MOI.get(m, MOI.SolveTimeSec())
        vars_shot = value.(x)

        @show vars_shot

        o_check = SCIP.Optimizer()
        lmo_check, _ = build_optimizer(o_check, p, k, Ns)
        @assert Boscia.is_linear_feasible(lmo_check.o, vars_shot)
        
        solution_shot = f(vars_shot)
        @show solution_shot
        ind = findall(x-> isapprox(0.0,x,rtol=1e-6,atol=1e-10), vars_shot)
        vars_shot[ind] = round.(vars_shot[ind])
        solution_round = f(vars_shot)
        @show solution_round
    else 
        solution_shot = NaN
        time_shot = time_limit
    end

    @show termination_shot, solution_shot
    df = DataFrame(seed=seed, dimension=n, p=p, k=k, Ns=Ns, time=time_shot, solution=solution_shot, termination=termination_shot)
    file_name = joinpath(@__DIR__,"csv/shot_poisson_reg_" * string(seed) * "_" * string(n) *  "_" * string(k) * "_"  * string(Ns) * ".csv")
    CSV.write(file_name, df, append=false, writeheader=true)
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

function poisson_reg_scip(seed=1, n=20, Ns=0.1)
    limit = 1800
    f, grad!, p, α, bs, Xs, ys, ws = build_function(seed, n, Ns=Ns)
    k = n/2

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
        @show solution_scip
        ncalls_scip = epigraph_ch.ncalls
        @assert Boscia.is_linear_feasible(lmo_check.o, vars_scip)

        df = DataFrame(seed=seed, dimension=n, p=p, k=k, Ns=Ns, time=time_scip, solution=solution_scip, termination=termination_scip, calls=ncalls_scip)
    else
        time_scip = MOI.get(lmo.o, MOI.SolveTimeSec())
        #@assert sum(ai.*vars_scip) <= bi + 1e-6 # constraint violated
        ncalls_scip = epigraph_ch.ncalls

        df = DataFrame(seed=seed, dimension=n, p=p, k=k, Ns=Ns, time=time_scip, solution=Inf, termination=termination_scip, calls=ncalls_scip)
    end
    file_name = joinpath(@__DIR__,"csv/scip_oa_poisson_reg_" * string(seed) * "_" * string(n) *  "_" * string(k) * "_"  * string(Ns) * ".csv")
    CSV.write(file_name, df, append=false, writeheader=true)
end

# build tree 
function build_bnb_ipopt_model(n, Ns, p, k, α, bs, Xs, ys, ws; time_limit)
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

function poisson_reg_ipopt(seed=1, n=20, Ns=0.1; time_limit=1800)
    # build tree
    f, grad!, p, α, bs, Xs, ys, ws = build_function(seed, n; Ns=Ns)
    k = n/2

    bnb_model, expr, p, k = build_bnb_ipopt_model(n, Ns, p, k, α, bs, Xs, ys, ws; time_limit=time_limit)
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
@show status
    #@show bnb_model.incumbent_solution.solution
    @show bnb_model.incumbent
    if bnb_model.incumbent_solution !== nothing
        o_check = SCIP.Optimizer()
        lmo_check, _ = build_optimizer(o_check, p, k, Ns)
        @assert Boscia.is_linear_feasible(lmo_check.o, bnb_model.incumbent_solution.solution)
    end

    df = DataFrame(seed=seed, dimension=n, p=p, Ns=Ns, k=k, time=total_time_in_sec, num_nodes = bnb_model.num_nodes, solution=bnb_model.incumbent, termination=status)
    file_name = joinpath(@__DIR__,"csv/ipopt_poisson_reg_" * string(seed) * "_" * string(n) *  "_" * string(k) * "_"  * string(Ns) * ".csv")
    CSV.write(file_name, df, append=false, writeheader=true)
end