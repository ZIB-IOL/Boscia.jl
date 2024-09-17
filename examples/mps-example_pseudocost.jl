using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using HiGHS
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface


# A MIPLIB instance: 22433
# https://miplib.zib.de/instance_details_22433.html
# Objective function: Minimize the distance to randomely picked vertices
# Number of variables   429
# Number of integers      0
# Number of binaries    231
# Number of constraints 198

# seed = rand(UInt64)
# @show seed
# Random.seed!(seed)

function build_example(o, seed)
    Random.seed!(seed)
    MOI.empty!(o)
    src = MOI.FileFormats.Model(filename="22433.mps")
    MOI.read_from_file(src, joinpath(@__DIR__, "mps-examples/mps-files/22433.mps"))
   
    MOI.copy_to(o, src)
    MOI.set(o, MOI.Silent(), true)
    n = MOI.get(o, MOI.NumberOfVariables())
    lmo = Boscia.MathOptBLMO(o)
    vs = [FrankWolfe.compute_extreme_point(lmo, randn(n)) for _ in 1:20]

    unique!(vs)
    filter!(vs) do v
        return v[end] != 21477.0
    end

    @assert !isempty(vs)
    b_mps = randn(n)
    function f(x)
        r = dot(b_mps, x)
        for v in vs
            r += 1 / 2 * norm(x - v)^2
        end
        return r
    end
    
    function grad!(storage, x)
        mul!(storage, length(vs) * I, x)
        storage .+= b_mps
        for v in vs
            @. storage -= v
        end
    end
    return  lmo, f, grad!    
end




############ Decide which strategies to run #####################
strategies = Any[
    "MOST_INFEASIBLE", "Strong_Branching"
]

for iterations_stable in Int64[5,10]
    for decision_function in [
        "product", 
        "weighted_sum"
        ]
        if decision_function == "product"
            μ = 1e-6
            push!(strategies, Dict(:iterations_stable => iterations_stable, :μ => μ, :decision_function => decision_function))
        else
            for μ in [0.7]
                push!(strategies, Dict(:iterations_stable => iterations_stable, :μ => μ, :decision_function => decision_function))
            end
        end
    end
end


############## Example sizes ######################

seeds = rand(UInt64, 3)

############## Set Parameters for all runs ######################
verbose = true
print_iter = 100
time_limit = 60
rel_dual_gap=1e-2
# Set parameters for saving results
file_name = "MPS_22433_instance_a_c"
#example_name = string("int_sparse_reg_n_", n, "_m_", m, "_l_",l, "_k_", k)
for i in [1]
    o = SCIP.Optimizer()
    lmo, f, grad! = build_example(o, 1)
    # println(o)
    println("precompile")
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
    # print(lmo.o)
end
println("actual run")#################################################################

#################################################################

for seed in seeds
    for branching_strategy in strategies
        example_name = "MPS_22433_instance"
        o = SCIP.Optimizer()
        f, grad!, lmo = build_example(o,  seed)
        if branching_strategy == "Strong_Branching"
            blmo = Boscia.MathOptBLMO(HiGHS.Optimizer())
            branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
            MOI.set(branching_strategy.bounded_lmo.o, MOI.Silent(), true)
            x, _, result =
                Boscia.solve(
                    f, 
                    grad!, 
                    lmo,  
                    branching_strategy=branching_strategy,verbose=verbose,
                    print_iter=print_iter, 
                    time_limit=time_limit)
            settings = "Strong_Branching"
            Boscia.save_results(result, settings, example_name, seed, file_name, false) 
        
        elseif branching_strategy == "MOST_INFEASIBLE"
            x, _, result = Boscia.solve(
                f, 
                grad!, 
                lmo, 
                verbose=verbose, 
                print_iter=print_iter, 
                time_limit=time_limit)
            settings = "MOST_INFEASIBLE"
            Boscia.save_results(result, settings, example_name, seed, file_name, false) 
        else
            iterations_stable = branching_strategy[:iterations_stable]
            decision_function = branching_strategy[:decision_function]
            μ = branching_strategy[:μ]
            x, _, result = Boscia.solve(
                f, 
                grad!, 
                lmo,
                branching_strategy=Boscia.PSEUDO_COST(iterations_stable,false, lmo, μ, decision_function),
                verbose=verbose, 
                print_iter=print_iter, 
                time_limit=time_limit)
                settings = "PSEUDO_COST_" * decision_function * "_" * string(iterations_stable) * "_μ=" * string(μ)
            Boscia.save_results(result, settings, example_name, seed, file_name, false)
        end
    end
end




# src = MOI.FileFormats.Model(filename="22433.mps")
# MOI.read_from_file(src, joinpath(@__DIR__, "mps-examples/mps-files/22433.mps"))

# o = SCIP.Optimizer()
# MOI.copy_to(o, src)
# MOI.set(o, MOI.Silent(), true)
# n = MOI.get(o, MOI.NumberOfVariables())
# lmo = Boscia.MathOptBLMO(o)

# #trick to push the optimum towards the interior
# const vs = [FrankWolfe.compute_extreme_point(lmo, randn(n)) for _ in 1:20]
# # done to avoid one vertex being systematically selected
# unique!(vs)
# filter!(vs) do v
#     return v[end] != 21477.0
# end

# @assert !isempty(vs)
# const b_mps = randn(n)

# function f(x)
#     r = dot(b_mps, x)
#     for v in vs
#         r += 1 / 2 * norm(x - v)^2
#     end
#     return r
# end

# function grad!(storage, x)
#     mul!(storage, length(vs) * I, x)
#     storage .+= b_mps
#     for v in vs
#         @. storage -= v
#     end
# end

# @testset "MPS 22433 instance" begin
#     iterations_stable = 1::Int
#     μ = 0.7 # μ used in the computation of the branching score
#     x, _, result = Boscia.solve(f, grad!, lmo, branching_strategy=Boscia.PSEUDO_COST(iterations_stable,false, lmo, μ), verbose=true)
#     @test f(x) <= f(result[:raw_solution])
# end
