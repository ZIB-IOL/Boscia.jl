using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
import Ipopt


# MIPLIB instances
# Objective function: Minimize the distance to randomely picked vertices

# Possible files
# 22433               https://miplib.zib.de/instance_details_22433.html
# n5-3                https://miplib.zib.de/instance_details_n5-3.html
# neos5               https://miplib.zib.de/instance_details_neos5.html
# pg                  https://miplib.zib.de/instance_details_pg.html
# pg5_34              https://miplib.zib.de/instance_details_pg5_34.html
# ran14x18-disj-8     https://miplib.zib.de/instance_details_ran14x18-disj-8.html
# timtab1             https://miplib.zib.de/instance_details_timtab1.html   (Takes LONG!)

seed = rand(UInt64)
#seed =  0x1d52d0c243ef0c61
seed = 7924543777773248845
@show seed

#Random.seed!(seed)
file_name = "mip_examples_long" # for saving results in "./results/"
# To see debug statements
#ENV["JULIA_DEBUG"] = "Boscia"



function build_example(example, num_v, seed)
    file_name = string(example, ".mps")
    src = MOI.FileFormats.Model(filename=file_name)
    MOI.read_from_file(src, joinpath(@__DIR__, string("mps-files/", file_name)))

    o = SCIP.Optimizer()
    MOI.copy_to(o, src)
    MOI.set(o, MOI.Silent(), true)
    n = MOI.get(o, MOI.NumberOfVariables())
    lmo = Boscia.MathOptBLMO(o)

    # Disable Presolving
    MOI.set(o, MOI.RawOptimizerAttribute("presolving/maxrounds"), 0)
    Random.seed!(seed)
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
        mul!(storage, length(vs) / max_norm * I, x)
        storage .+= b_mps
        for v in vs
            @. storage -= 1 / max_norm * v
        end
    end

    return lmo, f, grad!
end
example = "neos5"



############ Decide which strategies to run #####################
strategies = Any[
    "MOST_INFEASIBLE", "Strong_Branching"
]

for iterations_stable in Int64[5,10,20]
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


############## Set Parameters for all runs ######################
verbose = true
print_iter=10
fw_epsilon=1e-1
min_node_fw_epsilon=1e-3
time_limit=600
rel_dual_gap=1e-2
# Set parameters for saving results
file_name = "mip-examples_a_c"


#example_name = string("int_sparse_reg_n_", n, "_m_", m, "_l_",l, "_k_", k)
#################################################################




for example in ["neos5", 
    "pg", "22433", "pg5_34", "ran14x18-disj-8", "n5-3",
    "timtab1" ]
    num_v = 0
    if example == "neos5"
        num_v = 5
    elseif example == "pg"
        num_v = 5
    elseif example == "22433"
        num_v = 20
    elseif example == "pg5_34"
        num_v = 5
    elseif example == "ran14x18-disj-8"
        num_v = 5
    elseif example == "n5-3"
        num_v = 100
    elseif example == "timtab1"
        num_v = 3
    end

    test_instance = string("MPS ", example, " instance")

    #println("Example $(example)")
    example_name = test_instance
    for branching_strategy in strategies
        lmo, f, grad! = build_example(example, num_v, seed)
        if branching_strategy == "Strong_Branching"
            blmo = Boscia.MathOptBLMO(SCIP.Optimizer())
            branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
            MOI.set(branching_strategy.bounded_lmo.o, MOI.Silent(), true)
            x, _, result =
                Boscia.solve(
                    f, 
                    grad!, 
                    lmo,  
                    branching_strategy=branching_strategy,
                    verbose=verbose,
                    print_iter=print_iter, 
                    time_limit=time_limit,
                    rel_dual_gap=rel_dual_gap,
                    fw_epsilon=fw_epsilon,
                    min_node_fw_epsilon=min_node_fw_epsilon
                )
            settings = "Strong_Branching"
            Boscia.save_results(result, settings, example_name, seed, file_name, false) 
        
        elseif branching_strategy == "MOST_INFEASIBLE"
            x, _, result = Boscia.solve(
                f, 
                grad!, 
                lmo, 
                verbose=verbose, 
                print_iter=print_iter, 
                time_limit=time_limit,
                rel_dual_gap=rel_dual_gap,
                fw_epsilon=fw_epsilon,
                min_node_fw_epsilon=min_node_fw_epsilon
                )
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
                time_limit=time_limit,
                rel_dual_gap=rel_dual_gap,
                fw_epsilon=fw_epsilon,
                min_node_fw_epsilon=min_node_fw_epsilon
                )
                settings = "PSEUDO_COST_" * decision_function * "_" * string(iterations_stable) * "_μ=" * string(μ)
            Boscia.save_results(result, settings, example_name, seed, file_name, false)
        end
    end
end



#     ############## Product PSEUDO_COST ###############################
    
#     iterations_stable = 10
#     decision_function = "product"
#     if decision_function == "product"
#         μ = 1e-6
#     else
#         μ = 0.7 # μ used in the computation of the branching score
#     end

#     x, _, result = Boscia.solve(
#         f,
#         grad!,
#         lmo,
#         branching_strategy=Boscia.PSEUDO_COST(iterations_stable,false, lmo, μ, decision_function),
#         verbose=true,
#         print_iter=10,
#         fw_epsilon=1e-1,
#         min_node_fw_epsilon=1e-3,
#         time_limit=3600,
#     )
#     settings = "PSEUDO_COST_" * decision_function * "_" * string(iterations_stable) * "_μ=" * string(μ)
#     Boscia.save_results(result, settings, example, seed, file_name, false) 

#     ############## MOST_INFEASIBLE #####################
#     lmo, f, grad! = build_example(example, num_v, seed)
#     settings = "MOST_INFEASIBLE"
#     x, _, result = Boscia.solve(
#         f,
#         grad!,
#         lmo,
#         verbose=true,
#         print_iter=10,
#         fw_epsilon=1e-1,
#         min_node_fw_epsilon=1e-3,
#         time_limit=3600,
#     )

#     settings = "MOST_INFEASIBLE"
#     Boscia.save_results(result, settings, example, seed, file_name, false) 

#     ############## Strong_Branching ##############################
#     lmo, f, grad! = build_example(example, num_v, seed)
#     blmo = Boscia.MathOptBLMO(SCIP.Optimizer())
#     branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
#     MOI.set(branching_strategy.bounded_lmo.o, MOI.Silent(), true)

#     x, _, result =
#         Boscia.solve(
#         f, 
#         grad!, 
#         lmo,
#         branching_strategy=branching_strategy, 
#         verbose=true,
#         print_iter=10,
#         fw_epsilon=1e-1,
#         min_node_fw_epsilon=1e-3,
#         time_limit=3600,
#     )
#     settings = "Strong_Branching"
#     Boscia.save_results(result, settings, example, seed, file_name, false) 
# end