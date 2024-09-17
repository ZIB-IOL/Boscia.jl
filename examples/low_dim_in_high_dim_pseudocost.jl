using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using HiGHS
# using Statistics
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface

# The example from  "Optimizing a low-dimensional convex function over a high-dimensional cube"
# by Christoph Hunkenschröder, Sebastian Pokutta, Robert Weismantel
# https://arxiv.org/abs/2204.05266. 

m = 500 # larger dimension
n = 12 # small dimension

function low_dim_high_dim(o, m, n, seed; alpha=0.00)
    Random.seed!(seed)
    refpoint = 0.5 * ones(n) + Random.rand(n) * alpha * 1 / n
    W = rand(m, n)
    Ws = transpose(W) * W
    function f(x)
        return 0.5 * (dot(x, Ws, x) - dot(refpoint, Ws, x) - dot(x, Ws, refpoint))
    end
    
    function grad!(storage, x)
        return mul!(storage, Ws, (x - refpoint))
    end

    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
        MOI.add_constraint(o, xi, MOI.ZeroOne())
    end
    lmo = Boscia.MathOptBLMO(o)
    return lmo, f, grad!
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

n_choices = Int[10,
#40
]
m_choices = [500]
seeds = rand(UInt64, 3)

############## Set Parameters for all runs ######################
verbose = true
print_iter = 100
time_limit = 60
rel_dual_gap=1e-2
# Set parameters for saving results
file_name = "low_dim_high_dim_a_c"
#example_name = string("int_sparse_reg_n_", n, "_m_", m, "_l_",l, "_k_", k)
for i in [1]
    o = SCIP.Optimizer()
    lmo, f, grad! = low_dim_high_dim(o, 500, 10, 1; alpha=0.00)
    # println(o)
    println("precompile")
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
    # print(lmo.o)
end
println("actual run")#################################################################


for seed in seeds
    for dim in n_choices
        n0 = dim       
        for branching_strategy in strategies
            o = SCIP.Optimizer()
            lmo, f, grad! = low_dim_high_dim(o, m, n, seed; alpha= 0.00)
            example_name = string("low_dim_high_dim_m_", m, "n_", n)
            if branching_strategy == "Strong_Branching"
                #blmo = Boscia.MathOptBLMO(SCIP.Optimizer())
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
                        time_limit=time_limit,
                        rel_dual_gap=rel_dual_gap
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
                    rel_dual_gap=rel_dual_gap
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
                    rel_dual_gap=rel_dual_gap
                    )
                    settings = "PSEUDO_COST_" * decision_function * "_" * string(iterations_stable) * "_μ=" * string(μ)
                Boscia.save_results(result, settings, example_name, seed, file_name, false)
            end
        end
    end
end



# alpha = 0.00
# const refpoint = 0.5 * ones(n) + Random.rand(n) * alpha * 1 / n
# W = rand(m, n)
# const Ws = transpose(W) * W

# function f(x)
#     return 0.5 * (dot(x, Ws, x) - dot(refpoint, Ws, x) - dot(x, Ws, refpoint))
# end

# function grad!(storage, x)
#     return mul!(storage, Ws, (x - refpoint))
# end

# @testset "Low-dimensional function (SCIP)" begin
#     o = SCIP.Optimizer()
#     MOI.set(o, MOI.Silent(), true)
#     MOI.empty!(o)
#     x = MOI.add_variables(o, n)
#     for xi in x
#         MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
#         MOI.add_constraint(o, xi, MOI.LessThan(1.0))
#         MOI.add_constraint(o, xi, MOI.ZeroOne())
#     end
#     lmo = FrankWolfe.MathOptLMO(o)

#     x, _, result = Boscia.solve(f, grad!, lmo, verbose=true)

#     if n < 15  # only do for small n 
#         valopt, xopt = Boscia.min_via_enum(f, n)
#         @test (f(x) - f(xopt)) / abs(f(xopt)) <= 1e-3
#     end

#     @test f(x) <= f(result[:raw_solution]) + 1e-6
# end

# @testset "Low-dimensional function (CubeSimpleBLMO)" begin

#     int_vars = collect(1:n)

#     lbs = zeros(n)
#     ubs = ones(n)
    
#     sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
    
#     # modified solve call from managed_blmo.jl automatically wraps sblmo into a managed_blmo
#     x, _, result = Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, verbose=true)

#     if n < 15  # only do for small n 
#         valopt, xopt = Boscia.min_via_enum(f, n)
#         @test (f(x) - f(xopt)) / abs(f(xopt)) <= 1e-3
#     end

#     @test f(x) <= f(result[:raw_solution]) + 1e-6
# end
