using FrankWolfe
using LinearAlgebra
import MathOptInterface
using Random
using Boscia
using Bonobo
import Bonobo
using Printf
using Dates
using HiGHS
using SCIP
const MOI = MathOptInterface

# n = 50
# seed = rand(UInt64)

# Random.seed!(seed)
# @show(seed)

# ## generate constants ###
# const A = let
#     A = randn(n, n)
#     A' * A
# end

# @assert isposdef(A) == true

# const y = Random.rand(Bool, n) * 0.6 .+ 0.3

################################################################
# alternative implementation of LMO using MOI and SCIP
################################################################

function build_examples(o, n,  seed)
    Random.seed!(seed)
    A = let
        A = randn(n, n)
        A' * A
    end
    
    @assert isposdef(A) == true
    
    y = Random.rand(Bool, n) * 0.6 .+ 0.3
    
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.ZeroOne())
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
    end
    lmo = Boscia.MathOptBLMO(o)

    function f(x)
        d = x - y
        return dot(d, A, d)
    end

    function grad!(storage, x)
        # storage = Ax
        mul!(storage, A, x)
        # storage = 2Ax - 2Ay
        return mul!(storage, A, y, -2, 2)
    end
    return f, grad!, lmo
end
   

################################################################
# LMO via CubeSimpleBLMO
################################################################
# function build_examples(n, seed)
#     Random.seed!(seed)
#     int_vars = collect(1:n)

#     lbs = zeros(n)
#     ubs = ones(n)

#     sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
#     # wrap the sblmo into a bound manager
#     lmo = Boscia.ManagedBoundedLMO(sblmo, lbs[int_vars], ubs[int_vars], int_vars, n)

#     const A = let
#         A = randn(n, n)
#         A' * A
#     end

#     @assert isposdef(A) == true

#     const y = Random.rand(Bool, n) * 0.6 .+ 0.3

#     function f(x)
#         d = x - y
#         return dot(d, A, d)
#     end

#     function grad!(storage, x)
#         # storage = Ax
#         mul!(storage, A, x)
#         # storage = 2Ax - 2Ay
#         return mul!(storage, A, y, -2, 2)
#     end
#     return f, grad!, lmo
# end
   


#################


# are these lmo calls counted as well?

# #####
# # follow the gradient for a fixed number of steps and collect solutions on the way
# #####

# function follow_gradient_heuristic(tree::Bonobo.BnBTree, blmo::Boscia.BoundedLinearMinimizationOracle, x, k)
#     nabla = similar(x)
#     x_new = copy(x)
#     sols = []
#     for i in 1:k
#         tree.root.problem.g(nabla,x_new)
#         x_new = Boscia.compute_extreme_point(blmo, nabla)
#         push!(sols, x_new)
#     end
#     return sols, false
# end

# #####
# # rounding respecting the hidden feasible region structure
# #####

# function rounding_lmo_01_heuristic(tree::Bonobo.BnBTree, blmo::Boscia.BoundedLinearMinimizationOracle, x)
#     nabla = zeros(length(x))
#     for idx in tree.branching_indices
#         nabla[idx] = 1 - 2*round(x[idx]) # (0.7, 0.3) -> (1, 0) -> (-1, 1) -> min -> (1,0)
#     end
#     x_rounded = Boscia.compute_extreme_point(blmo, nabla)
#     return [x_rounded], false
# end

#####
# geometric scaling like for a couple of steps
#####


# depth = 5
# heu  = Boscia.Heuristic((tree, blmo, x) -> Boscia.follow_gradient_heuristic(tree,blmo,x, depth), 0.8, :follow_gradient)
# heu2 = Boscia.Heuristic(Boscia.rounding_lmo_01_heuristic, 0.8, :lmo_rounding)

# heuristics = [heu, heu2]
# # heuristics = []

# x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, branching_strategy = branching_strategy, print_iter=500, custom_heuristics=heuristics)

# benchmarking Oracles
# f, grad!, lmo = build_examples(n, A, y)
# FrankWolfe.benchmark_oracles(f, grad!, () -> rand(n), lmo; k=100)



############## Example sizes ######################

n_choices = Int[10,
#40
]
seeds = rand(UInt64, 3)


############ Decide which strategies to run #####################
strategies = Any[
    "MOST_INFEASIBLE", "Strong_Branching"
]

for iterations_stable in Int64[10,20]
    for decision_function in [
        "product", 
        "weighted_sum"
        ]
        if decision_function == "product"
            μ = 1e-6
            push!(strategies, Dict(:iterations_stable => iterations_stable, :μ => μ, :decision_function => decision_function))
        else
            for μ in [0.7, 0.1,0.5,0.9]
                push!(strategies, Dict(:iterations_stable => iterations_stable, :μ => μ, :decision_function => decision_function))
            end
        end
    end
end

############## Set Parameters for all runs ######################
verbose = true
print_iter = 500
time_limit = 1800

# Set parameters for saving results
file_name = "nonlinear_examples_a_c"






for i in [1]
    o = SCIP.Optimizer()
    f, grad!, lmo = build_examples(o,10, 1)
    # println(o)
    println("precompile")
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
    # print(lmo.o)
end
println("actual run")#################################################################


#################################################################
for n in n_choices
    for seed in seeds
        for branching_strategy in strategies
            example_name = "nonlinear_n_" * string(n)
            o = SCIP.Optimizer()
            f, grad!, lmo = build_examples(o,n,  seed)
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
end

