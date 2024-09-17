using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using HiGHS
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
import HiGHS

# Example on the Birkhoff polytope but using permutation matrices directly
# https://arxiv.org/pdf/2011.02752.pdf
# https://www.sciencedirect.com/science/article/pii/S0024379516001257

# For bug hunting:
# seed = rand(UInt64)

# #seed = 0x3eb09305cecf69f0
# @show seed
# Random.seed!(seed)


# min_{X, θ} 1/2 * || ∑_{i in [k]} θ_i X_i - Xhat ||^2
# θ ∈ Δ_k (simplex)
# X_i ∈ P_n (permutation matrix)

# we linearize the bilinear terms in the objective
# min_{X, Y, θ} 1/2 ||∑_{i in [k]} Y - Xhat ||^2
# θ ∈ Δ_k (simplex)
# X_i ∈ P_n (permutation matrix)
# 0 ≤ Y_i ≤ X_i
# 0 ≤ θ_i - Y_i ≤ 1 - X_i

# The variables are ordered (Y, X, theta) in the MOI model
# the objective only uses the last n^2 variables
# Small dimensions since the size of the problem grows quickly (2 k n^2 + k variables)
# n = 5
# k = 3
# file_name = "birkhoff_examples"
# example_name = "birkhoff_n_" * string(n) * "_k_" * string(k)
# # generate random doubly stochastic matrix
# const Xstar = rand(n, n)
# while norm(sum(Xstar, dims=1) .- 1) > 1e-6 || norm(sum(Xstar, dims=2) .- 1) > 1e-6
#     Xstar ./= sum(Xstar, dims=1)
#     Xstar ./= sum(Xstar, dims=2)
# end

function build_function(n, k, seed)

    Random.seed!(seed)
    Xstar = rand(n, n)
    while norm(sum(Xstar, dims=1) .- 1) > 1e-6 || norm(sum(Xstar, dims=2) .- 1) > 1e-6
        Xstar ./= sum(Xstar, dims=1)
        Xstar ./= sum(Xstar, dims=2)
    end

    function f(x)
        s = zero(eltype(x))
        for i in eachindex(Xstar)
            s += 0.5 * (sum(x[(j-1)*n^2+i] for j in 1:k) - Xstar[i])^2
        end
        return s
    end
    
    # note: reshape gives a reference to the same data, so this is updating storage in-place
    function grad!(storage, x)
        storage .= 0
        for j in 1:k
            Sk = reshape(@view(storage[(j-1)*n^2+1:j*n^2]), n, n)
            @. Sk = -Xstar
            for m in 1:k
                Yk = reshape(@view(x[(m-1)*n^2+1:m*n^2]), n, n)
                @. Sk += Yk
            end
        end
        return storage
    end
    return f, grad!
end

function build_birkhoff_lmo(o,n,k)
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    Y = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
    X = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
    theta = MOI.add_variables(o, k)

    for i in 1:k
        MOI.add_constraint.(o, Y[i], MOI.GreaterThan(0.0))
        MOI.add_constraint.(o, Y[i], MOI.LessThan(1.0))
        MOI.add_constraint.(o, X[i], MOI.ZeroOne())
        MOI.add_constraint(o, theta[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, theta[i], MOI.LessThan(1.0))
        # doubly stochastic constraints
        MOI.add_constraint.(
            o,
            vec(sum(X[i], dims=1, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
            MOI.EqualTo(1.0),
        )
        MOI.add_constraint.(
            o,
            vec(sum(X[i], dims=2, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
            MOI.EqualTo(1.0),
        )
        # 0 ≤ Y_i ≤ X_i
        MOI.add_constraint.(o, 1.0 * Y[i] - X[i], MOI.LessThan(0.0))
        # 0 ≤ θ_i - Y_i ≤ 1 - X_i
        MOI.add_constraint.(o, 1.0 * theta[i] .- Y[i] .+ X[i], MOI.LessThan(1.0))
    end
    MOI.add_constraint(o, sum(theta, init=0.0), MOI.EqualTo(1.0))
    return Boscia.MathOptBLMO(o)
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

n_choices = Int[3,
#40
]
k_choices = [2]

seeds = rand(UInt64, 3)

############## Set Parameters for all runs ######################
verbose = true
print_iter = 100
time_limit = 60
rel_dual_gap=1e-2
# Set parameters for saving results
file_name = "birkhoff_a_c"
#example_name = string("int_sparse_reg_n_", n, "_m_", m, "_l_",l, "_k_", k)
for _ in [1]
    f, grad! = build_function(3,2, 1)
    o = SCIP.Optimizer()
    lmo = build_birkhoff_lmo(o, 3,2)
    println("precompile")
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
end
# print(lmo.o)
println("actual run")#################################################################


for seed in seeds
    for dim in n_choices
        for k in k_choices
            n = dim       
            for branching_strategy in strategies
                f, grad! = build_function(n, k, seed)
                o = SCIP.Optimizer()
                lmo = build_birkhoff_lmo(o, n,k)
                example_name = string("birkhoff_", n, "_k_",k)
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
end









# ############## weighted_sum PSEUDO_COST #############################
# function f(x)
#     s = zero(eltype(x))
#     for i in eachindex(Xstar)
#         s += 0.5 * (sum(x[(j-1)*n^2+i] for j in 1:k) - Xstar[i])^2
#     end
#     return s
# end

# # note: reshape gives a reference to the same data, so this is updating storage in-place
# function grad!(storage, x)
#     storage .= 0
#     for j in 1:k
#         Sk = reshape(@view(storage[(j-1)*n^2+1:j*n^2]), n, n)
#         @. Sk = -Xstar
#         for m in 1:k
#             Yk = reshape(@view(x[(m-1)*n^2+1:m*n^2]), n, n)
#             @. Sk += Yk
#         end
#     end
#     return storage
# end

# lmo = build_birkhoff_lmo()
# iterations_stable = 10::Int 
# decision_function = "weighted_sum"
# if decision_function == "product"
#     μ = 1e-6
# else
#     μ = 0.7 # μ used in the computation of the branching score
# end

# x, _, result = Boscia.solve(f, grad!, lmo, branching_strategy=Boscia.PSEUDO_COST(iterations_stable,false, lmo, μ, decision_function), verbose=true, fw_epsilon=1e-3, print_iter=100, time_limit=2400)
# settings = "PSEUDO_COST_" * decision_function * "_" * string(iterations_stable) * "_μ=" * string(μ)
# Boscia.save_results(result, settings, example_name, seed, file_name, false) 


# # ############## product PSEUDO_COST #############################
# function f(x)
#     s = zero(eltype(x))
#     for i in eachindex(Xstar)
#         s += 0.5 * (sum(x[(j-1)*n^2+i] for j in 1:k) - Xstar[i])^2
#     end
#     return s
# end

# # note: reshape gives a reference to the same data, so this is updating storage in-place
# function grad!(storage, x)
#     storage .= 0
#     for j in 1:k
#         Sk = reshape(@view(storage[(j-1)*n^2+1:j*n^2]), n, n)
#         @. Sk = -Xstar
#         for m in 1:k
#             Yk = reshape(@view(x[(m-1)*n^2+1:m*n^2]), n, n)
#             @. Sk += Yk
#         end
#     end
#     return storage
# end

# lmo = build_birkhoff_lmo()
# iterations_stable = 10::Int
# decision_function = "product"
# if decision_function == "product"
#     μ = 1e-6
# else
#     μ = 0.7 # μ used in the computation of the branching score
# end
# x, _, result = Boscia.solve(f, grad!, lmo, branching_strategy=Boscia.PSEUDO_COST(iterations_stable,false, lmo, μ, decision_function),verbose=true, fw_epsilon=1e-3, print_iter=100, time_limit=2400)
# settings = "PSEUDO_COST_" * decision_function * "_" * string(iterations_stable)
# Boscia.save_results(result, settings, example_name, seed, file_name, false)


# ############## MOST_INFEASIBLE #############################
# function f(x)
#     s = zero(eltype(x))
#     for i in eachindex(Xstar)
#         s += 0.5 * (sum(x[(j-1)*n^2+i] for j in 1:k) - Xstar[i])^2
#     end
#     return s
# end

# # note: reshape gives a reference to the same data, so this is updating storage in-place
# function grad!(storage, x)
#     storage .= 0
#     for j in 1:k
#         Sk = reshape(@view(storage[(j-1)*n^2+1:j*n^2]), n, n)
#         @. Sk = -Xstar
#         for m in 1:k
#             Yk = reshape(@view(x[(m-1)*n^2+1:m*n^2]), n, n)
#             @. Sk += Yk
#         end
#     end
#     return storage
# end
# lmo = build_birkhoff_lmo()
# x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, fw_epsilon=1e-3, print_iter=100, time_limit=2400)
# settings = "MOST_INFEASIBLE"
# Boscia.save_results(result, settings, example_name, seed, file_name, false) 



# ############## Strong_Branching #############################
# function f(x)
#     s = zero(eltype(x))
#     for i in eachindex(Xstar)
#         s += 0.5 * (sum(x[(j-1)*n^2+i] for j in 1:k) - Xstar[i])^2
#     end
#     return s
# end

# # note: reshape gives a reference to the same data, so this is updating storage in-place
# function grad!(storage, x)
#     storage .= 0
#     for j in 1:k
#         Sk = reshape(@view(storage[(j-1)*n^2+1:j*n^2]), n, n)
#         @. Sk = -Xstar
#         for m in 1:k
#             Yk = reshape(@view(x[(m-1)*n^2+1:m*n^2]), n, n)
#             @. Sk += Yk
#         end
#     end
#     return storage
# end
# lmo = build_birkhoff_lmo()
# blmo = Boscia.MathOptBLMO(SCIP.Optimizer())
# branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
# MOI.set(branching_strategy.bounded_lmo.o, MOI.Silent(), true)

# x, _, result_strong_branching =
#     Boscia.solve(f, grad!, lmo, verbose=true, branching_strategy=branching_strategy, fw_epsilon=1e-3, time_limit=2400)
# settings = "Strong_Branching"
# Boscia.save_results(result_strong_branching, settings, example_name, seed, file_name, false) 











# # TODO the below needs to be fixed
# # TODO can use the min_via_enum function if not too many solutions
# # build optimal solution
# # xopt = zeros(n)
# # for i in 1:n
# #     if diffi[i] > 0.5
# #         xopt[i] = 1
# #     end
# # end

# # @testset "Birkhoff" begin
# #     lmo = build_birkhoff_lmo()
# #     x, _, result_baseline = Boscia.solve(f, grad!, lmo, verbose=true)
# #     @test f(x) <= f(result_baseline[:raw_solution]) + 1e-6
# #     lmo = build_birkhoff_lmo()
# #     blmo = Boscia.MathOptBLMO(HiGHS.Optimizer())
# #     branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
# #     MOI.set(branching_strategy.bounded_lmo.o, MOI.Silent(), true)
# #     x_strong, _, result_strong =
# #         Boscia.solve(f, grad!, lmo, verbose=true, branching_strategy=branching_strategy)
# #     @test f(x) ≈ f(x_strong)
# #     @test f(x) <= f(result_strong[:raw_solution]) + 1e-6
# # end
