using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using JuMP
using Ipopt
using Pavito
using HiGHS
using AmplNLWriter, SHOT_jll
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
using DataFrames
using CSV

include("scip_oa.jl")
include("BnB_Ipopt.jl")

# A MIPLIB instance: 22433
# https://miplib.zib.de/instance_details_22433.html
# Objective function: Minimize the distance to randomely picked vertices
# Number of variables   429
# Number of integers      0
# Number of binaries    231
# Number of constraints 198

function build_function(o, example, num_v, seed)
    Random.seed!(seed)
    file_name = string(example, ".mps")
    src = MOI.FileFormats.Model(filename=file_name)
    MOI.read_from_file(src, joinpath(@__DIR__, string("mps-examples/mps-files/", file_name)))

    MOI.copy_to(o, src)
    MOI.set(o, MOI.Silent(), true)
    n = MOI.get(o, MOI.NumberOfVariables())
    lmo = FrankWolfe.MathOptLMO(o)

    #trick to push the optimum towards the interior
    vs = [FrankWolfe.compute_extreme_point(lmo, randn(n)) for _ in 1:num_v]
    unique!(vs)

    @assert !isempty(vs)
    b_mps = randn(n)
    # @show b_mps
    max_norm = maximum(norm.(vs))

    function f(x)
        r = dot(b_mps, x)
        for v in vs
            r += 1 / (2 * max_norm) * norm(x - v)^2
        end
        return r
    end

    function grad!(storage, x)
        mul!(storage, length(vs)/max_norm * I, x)
        storage .+= b_mps
        for v in vs
            @. storage -= 1/max_norm * v
        end
    end

    return f, grad!, max_norm, vs, b_mps
end 

function build_optimizer(o, example)
    file_name = string(example, ".mps")
    src = MOI.FileFormats.Model(filename=file_name)
    MOI.read_from_file(src, joinpath(@__DIR__, string("mps-examples/mps-files/", file_name)))

    MOI.copy_to(o, src)
    MOI.set(o, MOI.Silent(), true)
    n = MOI.get(o, MOI.NumberOfVariables())
    lmo = FrankWolfe.MathOptLMO(o)

    return lmo
end

function miplib_boscia(seed=1, num_v=5, full_callback=false; example, bo_mode="default", depth=1)
    limit = 600

    o = SCIP.Optimizer()
    f, grad!, max_norm, vs, b_mps = build_function(o, example, num_v, seed)
    lmo = build_optimizer(o, example)
    Boscia.solve(f, grad!, lmo; verbose=false, time_limit=10, afw=true)
    
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
    elseif bo_mode == "strong_convexity"
        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=limit, strong_convexity = 1/max_norm) 
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
    number_nodes = result[:number_nodes]
    if full_callback
        lb_list = result[:list_lb]
        ub_list = result[:list_ub]
        time_list = result[:list_time]
        list_lmo_calls = result[:list_lmo_calls_acc]
        list_open_nodes = result[:open_nodes]
        list_local_tightening = result[:local_tightenings]
        list_global_tightening = result[:global_tightenings]
    end
    # if occursin("Optimal", result[:status])
    #     status = "OPTIMAL"
    # end

    if full_callback
        df = DataFrame(seed=seed, num_v=num_v, time= time_list, lowerBound= lb_list, upperBound = ub_list, termination=status, LMOcalls = list_lmo_calls, openNodes=list_open_nodes, localTighteings=list_local_tightening, globalTightenings=list_global_tightening)
        file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_mip_lib_" * example * "_" * string(num_v) * "_" *string(seed) *".csv")
        CSV.write(file_name, df, append=false)
    else
        df = DataFrame(seed=seed, num_v=num_v, time=total_time_in_sec, solution=result[:primal_objective], dual_gap=result[:dual_gap], rel_dual_gap=result[:rel_dual_gap], number_nodes=number_nodes, termination=status, ncalls=result[:lmo_calls])
        if bo_mode == "default" || bo_mode == "local_tightening" || bo_mode == "global_tightening" || bo_mode == "no_tightening" || bo_mode == "strong_convexity" || bo_mode == "afw" || bo_mode == "strong_branching"
            file_name = joinpath(@__DIR__, "csv/boscia_" * bo_mode * "_mip_lib_" * example * "_" * string(num_v) * "_" * string(seed) * ".csv")
        elseif bo_mode == "hybrid_branching"
            file_name = joinpath(@__DIR__, "csv/boscia_" * bo_mode * "_" * string(depth) * "_mip_lib_" * example * "_" * string(num_v) * "_" * string(seed) * ".csv")
        else 
            file_name = joinpath(@__DIR__,"csv/no_warm_start_" * bo_mode * "_mip_lib_" * example * "_" * string(num_v) * "_" * string(seed) * ".csv")
        end
        CSV.write(file_name, df, append=false, writeheader=true)
    end

    return f(x), x
end

function build_pavito_model(example, max_norm, vs, b_mps; time_limit=1800)
    o = Model(
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
    set_silent(o)
    MOI.set(o, MOI.TimeLimitSec(), time_limit)

    # load constraints from miplib instance
    file_name = string(example, ".mps")
    src = MOI.FileFormats.Model(filename=file_name)
    MOI.read_from_file(src, joinpath(@__DIR__, string("mps-examples/mps-files/", file_name)))

    MOI.copy_to(o, src)
    MOI.set(o, MOI.Silent(), true)
    n = MOI.get(o, MOI.NumberOfVariables())

    x = JuMP.all_variables(o)
    expr1 = @expression(o, dot(b_mps, x))
    expr = @expression(o, expr1 + 0.5 * 1/max_norm * sum(dot((x - vs[i]), (x - vs[i]) ) for i in 1:length(vs)))
    @objective(o, Min, expr)

    return o, x, n
end

function miplib_pavito(seed, num_v; example, time_limit=1800)
    @show example, num_v, seed

    o = SCIP.Optimizer()
    f, grad!, max_norm, vs, b_mps = build_function(o, example, num_v, seed)
    m, x, n = build_pavito_model(example, max_norm, vs, b_mps; time_limit=time_limit)

    # println(m)
    optimize!(m)
    termination_pavito = String(string(MOI.get(m, MOI.TerminationStatus())))

    if termination_pavito != "TIME_LIMIT" && termination_pavito != "OPTIMIZE_NOT_CALLED"
        time_pavito = MOI.get(m, MOI.SolveTimeSec())
        vars_pavito = value.(x)
        
        o_check = SCIP.Optimizer()
        lmo = build_optimizer(o_check, example)
        @assert Boscia.is_linear_feasible(lmo.o, vars_pavito)

        solution_pavito = f(vars_pavito)
    else 
        solution_pavito = NaN
        time_pavito = time_limit
    end
    @show solution_pavito, time_pavito

    df = DataFrame(seed=seed, num_v=num_v, time=time_pavito, solution=solution_pavito, termination=termination_pavito)
    file_name = joinpath(@__DIR__,"csv/pavito_miplib_" * example * "_" * string(num_v) * "_" * string(seed) * ".csv")
    CSV.write(file_name, df, append=false, writeheader=true)
end

function build_shot_model(example, max_norm, vs, b_mps; time_limit=1800)
    o = Model(() -> AmplNLWriter.Optimizer(SHOT_jll.amplexe))
    # set_silent(m)
    set_optimizer_attribute(o, "Termination.TimeLimit", time_limit)
    set_optimizer_attribute(o, "Output.Console.LogLevel", 3)
    set_optimizer_attribute(o, "Output.File.LogLevel", 6)
    set_optimizer_attribute(o, "Termination.ObjectiveGap.Absolute", 1e-6)
    set_optimizer_attribute(o, "Termination.ObjectiveGap.Relative", 1e-6)

    # load constraints from miplib instance
    file_name = string(example, ".mps")
    src = MOI.FileFormats.Model(filename=file_name)
    MOI.read_from_file(src, joinpath(@__DIR__, string("mps-examples/mps-files/", file_name)))

    MOI.copy_to(o, src)
    # MOI.set(o, MOI.Silent(), true)
    n = MOI.get(o, MOI.NumberOfVariables())

    x = JuMP.all_variables(o)
    expr1 = @expression(o, dot(b_mps, x))
    expr = @expression(o, expr1 + 0.5 * 1/max_norm * sum(dot((x - vs[i]), (x - vs[i]) ) for i in 1:length(vs)))
    @objective(o, Min, expr)

    return o, x, n
end

function miplib_shot(seed, num_v; example, time_limit=1800)
    @show example, num_v, seed

    o = SCIP.Optimizer()
    f, grad!, max_norm, vs, b_mps = build_function(o, example, num_v, seed)
    m, x, n = build_shot_model(example, max_norm, vs, b_mps; time_limit=time_limit)

    # println(m)
    optimize!(m)
    termination_shot = String(string(MOI.get(m, MOI.TerminationStatus())))

    if termination_shot != "TIME_LIMIT" && termination_shot != "OPTIMIZE_NOT_CALLED"
        time_shot = MOI.get(m, MOI.SolveTimeSec())
        vars_shot = value.(x)
        
        o_check = SCIP.Optimizer()
        lmo = build_optimizer(o_check, example)
        # println(lmo.o)
        @assert Boscia.is_linear_feasible(lmo.o, vars_shot)

        solution_shot = f(vars_shot)
    else 
        solution_shot = NaN
        time_shot = time_limit
    end
    @show solution_shot, time_shot


    df = DataFrame(seed=seed, num_v=num_v, time=time_shot, solution=solution_shot, termination=termination_shot)
    file_name = joinpath(@__DIR__,"csv/shot_miplib_" * example * "_" * string(num_v) * "_" * string(seed) * ".csv")
    CSV.write(file_name, df, append=false, writeheader=true)
end

function build_example_scip(example, num_v, seed; limit)
    f, grad!, max_norm, vs, b_mps = build_function(SCIP.Optimizer(), example, num_v, seed)
    o = SCIP.Optimizer()
    MOI.set(o, MOI.TimeLimitSec(), limit)
    MOI.set(o, MOI.Silent(), true)
    lmo = build_optimizer(o, example)
    x = MOI.get(lmo.o, MOI.ListOfVariableIndices())
    z_i = MOI.add_variable(lmo.o)
    epigraph_ch = GradientCutHandler(lmo.o, f, grad!, zeros(length(x)), z_i, x, 0)
    SCIP.include_conshdlr(lmo.o, epigraph_ch; needs_constraints=false, name="handler_gradient_cuts")
    MOI.set(lmo.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    
    o_check = SCIP.Optimizer()
    lmo_check = build_optimizer(o_check, example)
    z_i = MOI.add_variable(lmo_check.o)
    MOI.set(lmo_check.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    
    return lmo, epigraph_ch, x, f, lmo_check
end

function miplib_scip(seed=1, num_v=5; example, time_limit=1800)
    lmo, epigraph_ch, x, f, lmo_check = build_example_scip(example, num_v, seed; limit=time_limit)

    MOI.optimize!(lmo.o)
    time_scip = MOI.get(lmo.o, MOI.SolveTimeSec())
    vars_scip = MOI.get(lmo.o, MOI.VariablePrimal(), x)
    @assert Boscia.is_linear_feasible(lmo_check.o, vars_scip)
    
    integer_variables = Vector{Int}()
    num_int = 0
    num_bin = 0
    for cidx in MOI.get(lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}())
        push!(integer_variables, cidx.value)
        num_int += 1
    end
    for cidx in MOI.get(lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.ZeroOne}())
        push!(integer_variables, cidx.value)
        num_bin += 1
    end

    @assert Boscia.is_integer_feasible(integer_variables, vars_scip)
    
    solution_scip = f(vars_scip)
    termination_scip = String(string(MOI.get(lmo.o, MOI.TerminationStatus())))
    ncalls_scip = epigraph_ch.ncalls
    
    df = DataFrame(seed=seed, num_v=num_v, time=time_scip, solution=solution_scip, termination=termination_scip, calls=ncalls_scip)
    file_name = joinpath(@__DIR__,"csv/scip_oa_miplib_" * example * "_" * string(num_v) * "_" * string(seed) * ".csv")
    CSV.write(file_name, df, append=false, writeheader=true)

    return f(vars_scip), vars_scip
end

# build tree 
function build_bnb_ipopt_model(example, vs, b_mps, max_norm; time_limit=1800)
    m = Model(Ipopt.Optimizer)
    set_silent(m)

    file_name = example * ".mps"
    src = MOI.FileFormats.Model(filename=file_name)
    MOI.read_from_file(src, joinpath(@__DIR__, "mps-examples/mps-files/" ,file_name))
    int_vars = []

    o = SCIP.Optimizer()
    MOI.copy_to(o, src)
    MOI.set(o, MOI.Silent(), true)
    n = MOI.get(o, MOI.NumberOfVariables())
    lbs = fill(-Inf, n)
    ubs = fill(Inf, n)

    lmo = FrankWolfe.MathOptLMO(o)

    for cidx in MOI.get(lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}())
        push!(int_vars, cidx.value)
    end
    for cidx in MOI.get(lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.ZeroOne}())
        push!(int_vars, cidx.value)
    end

    consVal_list = MOI.get(o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Interval{Float64}}())
    for idx in consVal_list
        cons_val = MOI.get(o, MOI.ConstraintSet(), idx)
        lbs[idx.value] = cons_val.lower
        ubs[idx.value] = cons_val.upper
    end
    MOI.delete.(o, consVal_list)
    for idx in 1:n
        MOI.add_constraint(o, MOI.VariableIndex(idx), MOI.GreaterThan(lbs[idx]))
        MOI.add_constraint(o, MOI.VariableIndex(idx), MOI.LessThan(ubs[idx]))
    end

    # Relaxed version
    filtered_src = MOI.Utilities.ModelFilter(o) do item
    if item isa Tuple
        (_, S) = item
        if S <: Union{MOI.Indicator, MOI.Integer, MOI.ZeroOne}
            return false
        end
    end
    return !(item isa MOI.ConstraintIndex{<:Any, <:Union{MOI.ZeroOne, MOI.Integer, MOI.Indicator}})
    end

    JuMP.Model(() -> Ipopt.Optimizer()) #MOI.Bridges.full_bridge_optimizer(Ipopt.Optimizer(), Float64)) () -> Ipopt.Optimizer()
    index_map = MOI.copy_to(m, filtered_src)
    # sanity check, otherwise the functions need permuted indices
    for (v1, v2) in index_map
        if v1 isa MOI.VariableIndex
            @assert v1 == v2
        end
    end

    x = JuMP.all_variables(m)
    expr1 = @expression(m, dot(b_mps, x))
    expr = @expression(m, expr1 + 0.5 * 1/max_norm * sum(dot((x - vs[i]), (x - vs[i]) ) for i in 1:length(vs)))
    @objective(m, Min, expr)

    model = IpoptOptimizationProblem(int_vars, m, Boscia.SOLVING, time_limit, lbs, ubs)
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

# BnB tree with Ipopt
function miplib_ipopt(seed=1, num_v=5, full_callback = false; example, time_limit=1800)
    o = SCIP.Optimizer()
    f, grad!, max_norm, vs, b_mps = build_function(o, example, num_v, seed)

    # build tree
    bnb_model, expr = build_bnb_ipopt_model(example, vs, b_mps, max_norm, time_limit=time_limit)
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
    if full_callback
        df = DataFrame(seed=seed, num_v=num_v,number_nodes = bnb_model.num_nodes, time=list_time, lowerBound = list_lb, upperBound = list_ub, termination=status,)
        file_name =joinpath(@__DIR__,"csv/ipopt_" * example * "_" * string(num_v) * "_" * string(seed) *  ".csv")
        CSV.write(file_name, df, append=false)
    else
        df = DataFrame(seed=seed, num_v=num_v, time=total_time_in_sec, number_nodes = bnb_model.num_nodes, solution=bnb_model.incumbent, termination=status)
        file_name = joinpath(@__DIR__,"csv/ipopt_miplib_" * example * "_" * string(num_v) * "_" * string(seed) * ".csv")
        CSV.write(file_name, df, append=false, writeheader=true)
    end
end


