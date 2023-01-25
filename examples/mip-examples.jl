using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using JuMP
using Ipopt
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
using DataFrames
using CSV
include("scip_oa.jl")
include("BnB_Ipopt.jl")

# MIPLIB instances
# Objective function: Minimize the distance to randomely picked vertices

# Possible files
# 22433               https://miplib.zib.de/instance_details_22433.html
# n5-3                https://miplib.zib.de/instance_details_n5-3.html
# neos5               https://miplib.zib.de/instance_details_neos5.html
# pg                  https://miplib.zib.de/instance_details_pg.html !
# pg5_34              https://miplib.zib.de/instance_details_pg5_34.html
# ran14x18-disj-8     https://miplib.zib.de/instance_details_ran14x18-disj-8.html !
# timtab1             https://miplib.zib.de/instance_details_timtab1.html   (Takes LONG!)


function mip_lib(seed=1, num_v=5; example, bo_mode)
    limit = 1800

    o = SCIP.Optimizer()
    lmo, f, grad! = build_example(o, example, num_v, seed)
    Boscia.solve(f, grad!, lmo; verbose=false, time_limit=10, afw=true)

    o = SCIP.Optimizer()
    lmo, f, grad! = build_example(o, example, num_v, seed)

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
    end             

    total_time_in_sec=result[:total_time_in_sec]
    status = result[:status]
    number_nodes = result[:number_nodes]
    # if occursin("Optimal", result[:status])
    #     status = "OPTIMAL"
    # end
    df = DataFrame(seed=seed, num_v=num_v, time=total_time_in_sec, solution=result[:primal_objective], dual_gap=result[:dual_gap], rel_dual_gap=result[:rel_dual_gap], number_nodes=number_nodes, termination=status, ncalls=result[:lmo_calls])
    if bo_mode ==  "afw"
        file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_mip_lib_" * example * ".csv")
    elseif bo_mode == "boscia"
        file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_mip_lib_" * example * ".csv")
    else 
        file_name = joinpath(@__DIR__,"csv/no_warm_start_" * bo_mode * "_mip_lib_" * example * ".csv")
    end
    if !isfile(file_name)
        CSV.write(file_name, df, append=true, writeheader=true)
    else 
        CSV.write(file_name, df, append=true)
    end

    return f(x), x
end

function mip_lib_scip(seed=1, num_v=5; example)
    limit = 1800
    lmo, epigraph_ch, x, f, lmo_check = build_example_scip(example, num_v, seed, limit)

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
    file_name = joinpath(@__DIR__,"csv/scip_oa_mip_lib_" * example * ".csv")
    if !isfile(file_name)
        CSV.write(file_name, df, append=true, writeheader=true)
    else 
        CSV.write(file_name, df, append=true)
    end

    return f(vars_scip), vars_scip
end

function build_example(o, example, num_v, seed)
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
    # done to avoid one vertex being systematically selected
    unique!(vs)

    @assert !isempty(vs)
    # for 22433 and neos5
    # b_mps = randn(n) and max_norm = maximum(norm.(vs)) -> scip will be better than us in some cases but still within tolerance
    b_mps = randn(n)
    max_norm = maximum(norm.(vs))
    #max_norm = norm(b_mps) * 2

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

    return lmo, f, grad!
end

function build_example_scip(example, num_v, seed, limit)
    o = SCIP.Optimizer()
    MOI.set(o, MOI.TimeLimitSec(), limit)
    MOI.set(o, MOI.Silent(), true)
    lmo, f, grad! = build_example(o, example, num_v, seed)
    x = MOI.get(lmo.o, MOI.ListOfVariableIndices())
    z_i = MOI.add_variable(lmo.o)
    epigraph_ch = GradientCutHandler(lmo.o, f, grad!, zeros(length(x)), z_i, x, 0)
    SCIP.include_conshdlr(lmo.o, epigraph_ch; needs_constraints=false, name="handler_gradient_cuts")
    MOI.set(lmo.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    
    o_check = SCIP.Optimizer()
    lmo_check, _, _ = build_example(o_check, example, num_v, seed)
    z_i = MOI.add_variable(lmo_check.o)
    MOI.set(lmo_check.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    
    return lmo, epigraph_ch, x, f, lmo_check
end

# BnB tree with Ipopt
function mip_lib_ipopt(seed=1, num_v=5; example)
      # build tree
      bnb_model, expr = build_bnb_ipopt_model(seed, num_v, example)
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
  
      df = DataFrame(seed=seed, num_v=num_v, time=total_time_in_sec, number_nodes = bnb_model.num_nodes, solution=bnb_model.incumbent, termination=status)
      file_name = joinpath(@__DIR__,"csv/ipopt_" * example * ".csv")
      if !isfile(file_name)
          CSV.write(file_name, df, append=true, writeheader=true)
      else 
          CSV.write(file_name, df, append=true)
      end
end

# build tree 
function build_bnb_ipopt_model(seed, num_v, example)
    Random.seed!(seed)
    time_limit = 1800

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

    #trick to push the optimum towards the interior
    vs = [FrankWolfe.compute_extreme_point(lmo, randn(n)) for _ in 1:num_v]
    unique!(vs)
    @assert !isempty(vs)
    b_mps = randn(n)
    max_norm = maximum(norm.(vs))

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
    return bnb_model, expr

end

# num_v = 0
# if example == "neos5"
#     num_v = 5
# elseif example == "pg"
#     num_v = 5
# elseif example == "22433"
#     num_v = 20
# elseif example == "pg5_34"
#     num_v = 5
# elseif example == "ran14x18-disj-8"
#     num_v = 5
# elseif example == "n5-3"
#     num_v = 100
# elseif example == "timtab1"
#     num_v = 3
# end

# test_instance = string("MPS ", example, " instance")
# @testset "$test_instance" begin
#     println("Example $(example)")
#     lmo, f, grad! = build_example(example, num_v)
#     x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, print_iter = 10, fw_epsilon = 1e-1, min_node_fw_epsilon = 1e-3, time_limit=time_limit)
#     @test f(x) <= f(result[:raw_solution])
#     # TODO: save dual gap
# end

