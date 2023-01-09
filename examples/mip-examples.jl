using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
using DataFrames
using CSV

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
    lmo, f, grad! = build_problem(o, example, num_v, seed)

    # println("BOSCIA MODEL")
    # print(o)

    # x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=10, afw=true)

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

    total_time_in_sec=result[:total_time_in_sec]
    status = result[:status]
    if occursin("Optimal", result[:status])
        status = "OPTIMAL"
    end
    df = DataFrame(seed=seed, num_v=num_v, time=total_time_in_sec, solution=result[:primal_objective], dual_gap=result[:dual_gap], rel_dual_gap=result[:rel_dual_gap], termination=status, ncalls=result[:lmo_calls])
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


function build_example(example, num_v, seed)
    Random.seed!(seed)
    file_name = string(example, ".mps")
    src = MOI.FileFormats.Model(filename=file_name)
    MOI.read_from_file(src, joinpath(@__DIR__, string("mps-files/", file_name)))

    o = SCIP.Optimizer()
    MOI.copy_to(o, src)
    MOI.set(o, MOI.Silent(), true)
    n = MOI.get(o, MOI.NumberOfVariables())
    lmo = FrankWolfe.MathOptLMO(o)

    #trick to push the optimum towards the interior
    vs = [FrankWolfe.compute_extreme_point(lmo, randn(n)) for _ in 1:num_v]   
    # done to avoid one vertex being systematically selected
    unique!(vs)

    @assert !isempty(vs)
    b_mps = randn(n)
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

    return lmo, f, grad!
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

