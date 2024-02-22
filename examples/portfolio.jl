using Boscia
using FrankWolfe
using Random
using SCIP
using Pavito
using AmplNLWriter, SHOT_jll
# using Statistics
using LinearAlgebra
using Distributions
import MathOptInterface
MOI = MathOptInterface
using CSV
using DataFrames

include("scip_oa.jl")
include("BnB_Ipopt.jl")

function portfolio(seed=1, dimension=5, full_callback=false; mode, bo_mode)
    limit = 1800

    f, grad!, n = build_function(seed, dimension)
    o = SCIP.Optimizer()
    lmo, _ = build_optimizer(o, mode, n)
    println(o)
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
    
    if bo_mode == "afw"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=true, time_limit=limit, afw=true)
    elseif bo_mode == "as_ss"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=false, warmstart_shadow_set=false)
    elseif bo_mode == "as"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=false, warmstart_shadow_set=true)
    elseif bo_mode == "ss"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=true, time_limit=limit, warmstart_active_set=true, warmstart_shadow_set=false)
    elseif bo_mode == "boscia"
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, print_iter=1)
    elseif bo_mode == "local_tightening"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=true, global_dual_tightening=false, print_iter=1) 
    elseif bo_mode == "global_tightening"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=true, print_iter=1) 
    elseif bo_mode == "no_tightening"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=false, print_iter=1) 
    elseif bo_mode == "ss_local_tightening"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=true, global_dual_tightening=false, warmstart_shadow_set=false, print_iter=1) 
    elseif bo_mode == "ss_global_tightening"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=true, warmstart_shadow_set=false,print_iter=1) 
    elseif bo_mode == "ss_no_tightening"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, dual_tightening=false, global_dual_tightening=false, warmstart_shadow_set=false, print_iter=1) 
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
        df = DataFrame(seed=seed, dimension=n, time=time_list, lowerBound= lb_list, upperBound = ub_list, termination=status, LMOcalls = list_lmo_calls, openNodes=list_open_nodes, localTighteings=list_local_tightening, globalTightenings=list_global_tightening)
        file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_" * string(n) * "_" *string(seed) * "_" * mode * "_portfolio.csv")
        CSV.write(file_name, df, append=false)
    else
        @show result[:primal_objective]
        df = DataFrame(seed=seed, dimension=n, time=total_time_in_sec, solution=result[:primal_objective], dual_gap =result[:dual_gap], rel_dual_gap=result[:rel_dual_gap], termination=status, ncalls=result[:lmo_calls])
        if bo_mode ==  "afw"
            file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_" * mode * "_portfolio.csv")
        elseif bo_mode == "boscia" || bo_mode == "local_tightening" || bo_mode == "global_tightening" || bo_mode == "no_tightening"
            file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_" * mode * "_portfolio.csv")
        else 
            file_name = joinpath(@__DIR__,"csv/no_warm_start_" * bo_mode * "_" * mode * "_portfolio.csv")
        end
        if !isfile(file_name)
            CSV.write(file_name, df, append=true, writeheader=true)
        else 
            CSV.write(file_name, df, append=true)
        end
    end
end

function portfolio_scip(seed=1, dimension=5; mode)
    limit = 1800
    f, grad!, n = build_function(seed, dimension)

    lmo, epigraph_ch, x, lmo_check = build_scip_optimizer(mode, limit, n, f, grad!)
    MOI.optimize!(lmo.o)
    time_scip = MOI.get(lmo.o, MOI.SolveTimeSec())
    vars_scip = MOI.get(lmo.o, MOI.VariablePrimal(), x)
    @assert Boscia.is_linear_feasible(lmo_check.o, vars_scip)
    #@assert sum(ai.*vars_scip) <= bi + 1e-6 # constraint violated
    solution_scip = f(vars_scip)
    termination_scip = String(string(MOI.get(lmo.o, MOI.TerminationStatus())))
    ncalls_scip = epigraph_ch.ncalls

    df = DataFrame(seed=seed, dimension=n, time=time_scip, solution=solution_scip, termination=termination_scip, calls=ncalls_scip)
    file_name = joinpath(@__DIR__,"csv/scip_oa_portfolio_" * mode * ".csv")
    if !isfile(file_name)
        CSV.write(file_name, df, append=true, writeheader=true)
    else 
        CSV.write(file_name, df, append=true)
    end
end

function build_optimizer(o, mode, n)
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)

    ai = rand(n)
    bi = sum(ai)

    # MOI.set(o, MOI.TimeLimitSec(), limit)
    x = MOI.add_variables(o, n)
    
    # integer set
    if mode == "integer"
        I = collect(1:n)
    elseif mode == "mixed"
        I = 1:(n÷2)
    end

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
    return lmo, x
end

function build_scip_optimizer(mode, limit, n, f, grad!)
    o = SCIP.Optimizer()
    MOI.set(o, MOI.TimeLimitSec(), limit)
    MOI.set(o, MOI.Silent(), true)
    lmo, x = build_optimizer(o, mode, n)
    z_i = MOI.add_variable(lmo.o)
    epigraph_ch = GradientCutHandler(lmo.o, f, grad!, zeros(length(x)), z_i, x, 0)
    SCIP.include_conshdlr(lmo.o, epigraph_ch; needs_constraints=false, name="handler_gradient_cuts")
    MOI.set(lmo.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    
    # lmo to verify feasibility of solution after optimization
    o_check = SCIP.Optimizer()
    lmo_check, _ = build_optimizer(o_check, mode, n)
    z_i = MOI.add_variable(lmo_check.o)
    MOI.set(lmo_check.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    
    return lmo, epigraph_ch, x, lmo_check
end

function build_function(seed, dimension)
    Random.seed!(seed)
    n = dimension
    ri = rand(n)
    Ωi = rand()
    Ai = randn(n, n)
    Ai = Ai' * Ai
    Mi = (Ai + Ai') / 2
    @assert isposdef(Mi)

    function f(x)
        return 1 / 2 * Ωi * dot(x, Mi, x) - dot(ri, x)
    end
    function grad!(storage, x)
        mul!(storage, Mi, x, Ωi, 0)
        storage .-= ri
        return storage
    end
    return f, grad!, n
end

# BnB tree with Ipopt
function portfolio_ipopt(seed = 1, n = 20; mode="mixed")
    # build tree
    bnb_model, expr = build_bnb_ipopt_model(seed, n; mode=mode)
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

    o_check = SCIP.Optimizer()
    lmo_check, _ = build_optimizer(o_check, mode, n)
    x = Bonobo.get_solution(bnb_model)
    
    @assert Boscia.is_linear_feasible(lmo_check.o, x)

    df = DataFrame(seed=seed, dimension=n, time=total_time_in_sec, num_nodes = bnb_model.num_nodes, solution=bnb_model.incumbent, termination=status)
    file_name = joinpath(@__DIR__,"csv/ipopt_portfolio_ " * mode * ".csv")
    if !isfile(file_name)
        CSV.write(file_name, df, append=true, writeheader=true)
    else 
        CSV.write(file_name, df, append=true)
    end
end

# build tree 
function build_bnb_ipopt_model(seed, n; mode="mixed")
    Random.seed!(seed)
    time_limit = 1800
    
    ri = rand(n)
    Ωi = rand()
    Ai = randn(n, n)
    Ai = Ai' * Ai
    Mi = (Ai + Ai') / 2
    @assert isposdef(Mi)

    m = Model(Ipopt.Optimizer)
    set_silent(m)

    ai = rand(n)
    bi = sum(ai)
    # integer set
    if mode == "integer"
        I = collect(1:n)
    elseif mode == "mixed"
        I = collect(1:(n÷2))
    end

    @variable(m, x[1:n])
    for i in 1:n
        @constraint(m, x[i] >= 0)
    end
    lbs = fill(0.0, n)
    ubs = fill(Inf, n)

    @constraint(m, dot(ai, x) <= bi)
    @constraint(m, dot(x, ones(n)) >= 1.0)

    expr = @expression(m, 1/2 * Ωi * dot(x, Mi, x) - dot(ri, x))
    @objective(m, Min, expr)

    model = IpoptOptimizationProblem(I, m, Boscia.SOLVING, time_limit, lbs, ubs)
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

function build_pavito_model(seed, n; mode="mixed", time_limit=1800)
    Random.seed!(seed)
    
    ri = rand(n)
    Ωi = rand()
    Ai = randn(n, n)
    Ai = Ai' * Ai
    Mi = (Ai + Ai') / 2
    @assert isposdef(Mi)

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
    set_silent(m)
    MOI.set(m, MOI.TimeLimitSec(), time_limit)

    ai = rand(n)
    bi = sum(ai)
    # integer set
    if mode == "integer"
        I = collect(1:n)
    elseif mode == "mixed"
        I = collect(1:(n÷2))
    end

    @variable(m, x[1:n])
    for i in 1:n
        @constraint(m, x[i] >= 0)
        if i in I
            set_integer(x[i])
        end
    end

    @constraint(m, dot(ai, x) <= bi)
    @constraint(m, dot(x, ones(n)) >= 1.0)

    expr = @expression(m, 1/2 * Ωi * dot(x, Mi, x) - dot(ri, x))
    @objective(m, Min, expr)

    return m, x

end

function portfolio_pavito(seed=1, dimension=5; mode, time_limit=1800)
    @show seed, dimension, mode
    f, grad!, n = build_function(seed, dimension)

    m, x = build_pavito_model(seed, dimension; mode=mode, time_limit=time_limit)
    # println(m)

    # dest = MOI.FileFormats.Model(format = MOI.FileFormats.FORMAT_NL)
    # MOI.copy_to(dest, m)
    # MOI.write_to_file(dest, "file_pavito.nl")
    # # read file 
    # print(read("file_pavito.nl", String))
    # m = read_from_file("file_pavito.nl")

    optimize!(m)
    termination_pavito = String(string(MOI.get(m, MOI.TerminationStatus())))
    if termination_pavito != "TIME_LIMIT" && termination_pavito != "OPTIMIZE_NOT_CALLED"
        vars_pavito = value.(x)
        @assert Boscia.is_linear_feasible(m.moi_backend, vars_pavito)    
        #@assert sum(ai.*vars_scip) <= bi + 1e-6 # constraint violated
        solution_pavito = f(vars_pavito)
        time_pavito = MOI.get(m, MOI.SolveTimeSec())
    else 
        solution_pavito = NaN
        time_pavito = time_limit
    end
    
    # check linear feasiblity
    if termination_pavito != "TIME_LIMIT" && termination_pavito != "OPTIMIZE_NOT_CALLED"
        o = SCIP.Optimizer()
        lmo, _ = build_optimizer(o, mode, n)
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
        solution_boscia = result[:raw_solution]
        # @show f(vars_pavito), f(solution_boscia)
        if occursin("Optimal", result[:status])
            @assert result[:dual_bound] <= f(vars_pavito) + 1e-4
        end

        termination_pavito = String(string(termination_pavito))
        @show solution_pavito, termination_pavito
    end

    df = DataFrame(seed=seed, dimension=n, time=time_pavito, solution=solution_pavito, termination=termination_pavito)
        file_name = joinpath(@__DIR__,"csv/pavito_portfolio_" * mode * "_" * string(dimension) * "_" * string(seed) * ".csv")    

    if !isfile(file_name)
        CSV.write(file_name, df, append=true, writeheader=true)
    else 
        CSV.write(file_name, df, append=true)
    end

end

function build_shot_model(seed, n; mode="mixed", time_limit=1800)
    Random.seed!(seed)
    
    ri = rand(n)
    Ωi = rand()
    Ai = randn(n, n)
    Ai = Ai' * Ai
    Mi = (Ai + Ai') / 2
    @assert isposdef(Mi)

    m = Model(() -> AmplNLWriter.Optimizer(SHOT_jll.amplexe)) 
    # set_silent(m)
    # MOI.set(m, MOI.TimeLimitSec(), time_limit)

    ai = rand(n)
    bi = sum(ai)
    # integer set
    if mode == "integer"
        I = collect(1:n)
    elseif mode == "mixed"
        I = collect(1:(n÷2))
    end

    @variable(m, x[1:n])
    for i in 1:n
        @constraint(m, x[i] >= 0)
        if i in I
            set_integer(x[i])
        end
    end

    @constraint(m, dot(ai, x) <= bi)
    @constraint(m, dot(x, ones(n)) >= 1.0)

    expr = @expression(m, 1/2 * Ωi * dot(x, Mi, x) - dot(ri, x))
    @objective(m, Min, expr)

    return m, x

end

function portfolio_shot(seed=1, dimension=5; mode, time_limit=1800)
    @show seed, dimension, mode
    f, grad!, n = build_function(seed, dimension)

    m, x = build_shot_model(seed, dimension; mode=mode, time_limit=time_limit)
    # write model to .nl file
    # println(m)
    # dest = MOI.FileFormats.Model(format = MOI.FileFormats.FORMAT_NL)
    # MOI.copy_to(dest, m)
    # MOI.write_to_file(dest, "file.nl")
    # # read file 
    # m = read_from_file("file.nl")
    # set_optimizer(m, () -> AmplNLWriter.Optimizer(SHOT_jll.amplexe))
    # println(m)

    optimize!(m)
    termination_shot = String(string(MOI.get(m, MOI.TerminationStatus())))
    @show termination_shot
    if termination_shot != "TIME_LIMIT" && termination_shot != "OPTIMIZE_NOT_CALLED"
        vars_shot = value.(x)
        @assert Boscia.is_linear_feasible(m.moi_backend, vars_shot)    
        #@assert sum(ai.*vars_scip) <= bi + 1e-6 # constraint violated
        solution_shot = f(vars_shot)
        time_shot = MOI.get(m, MOI.SolveTimeSec())
    else 
        solution_shot = NaN
        time_shot = time_limit
    end
    
    # check linear feasiblity
    if termination_shot != "TIME_LIMIT" && termination_shot != "OPTIMIZE_NOT_CALLED"
        o = SCIP.Optimizer()
        lmo, _ = build_optimizer(o, mode, n)
        @assert Boscia.is_linear_feasible(lmo, vars_shot)
        # check integer feasibility
        integer_variables = Vector{Int}()
        for cidx in MOI.get(lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}())
            push!(integer_variables, cidx.value)
        end
        for idx in integer_variables
            @assert isapprox(vars_shot[idx], round(vars_shot[idx]); atol=1e-6, rtol=1e-6)
        end
        # check feasibility of rounded solution
        vars_shot_polished = vars_shot
        for i in integer_variables
            vars_shot_polished[i] = round(vars_shot_polished[i])
        end
        @assert Boscia.is_linear_feasible(lmo, vars_shot_polished)
        # solve Boscia
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=1800, dual_tightening=true, global_dual_tightening=true, rel_dual_gap=1e-6, fw_epsilon=1e-6)
        @show result[:dual_bound]
        solution_boscia = result[:raw_solution]
        # @show f(vars_shot), f(solution_boscia)
        if occursin("Optimal", result[:status])
            @assert result[:dual_bound] <= f(vars_shot) + 1e-4
        end

        termination_shot = String(string(termination_shot))
        @show solution_shot, termination_shot
    end

    df = DataFrame(seed=seed, dimension=n, time=time_shot, solution=solution_shot, termination=termination_shot)
        file_name = joinpath(@__DIR__,"csv/shot_portfolio_" * mode * "_" * string(dimension) * "_" * string(seed) * ".csv")    

    if !isfile(file_name)
        CSV.write(file_name, df, append=true, writeheader=true)
    else 
        CSV.write(file_name, df, append=true)
    end
end