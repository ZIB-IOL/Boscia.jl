using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using HiGHS	
# using Statistics
using LinearAlgebra
#using Distributions
import MathOptInterface
const MOI = MathOptInterface

include("cube_blmo.jl")

n = 20
diffi = Random.rand(Bool, n) * 0.6 .+ 0.3


function approx_planted_point_integer(n, seed)
    Random.seed!(seed)
    diffi = Random.rand(Bool, n) * 0.6 .+ 0.3
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end
    # using SCIP
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
        MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
    end
    lmo = Boscia.MathOptBLMO(o)
    return lmo, f, grad!
end

function approx_planted_point_mixed(n, seed)
    Random.seed!(seed)
    diffi = Random.rand(Bool, n) * 0.6 .+ 0.3
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end
    int_vars = unique!(rand(collect(1:n), Int(floor(n / 2))))
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
        if xi.value in int_vars
            MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
        end
    end
    lmo = Boscia.MathOptBLMO(o)

    return lmo, f, grad!
end 





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


############## Example sizes ######################

example_dimensions = [20, 25]
seeds = rand(UInt64, 3)

############## Set Parameters for all runs ######################
verbose = true
print_iter = 100
time_limit = 600
rel_dual_gap=1e-2
# Set parameters for saving results
file_name = "approx_planted_point_a_c"
#example_name = string("int_sparse_reg_n_", n, "_m_", m, "_l_",l, "_k_", k)
#################################################################

for seed in seeds
    for dim in example_dimensions
        example_name = string("approx_planted_point_integer_n", dim)
        for branching_strategy in strategies
            lmo, f, grad! = approx_planted_point_integer(dim, seed)
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

for seed in seeds
    for dim in example_dimensions
        example_name = string("approx_planted_point_mixed_n", dim)
        for branching_strategy in strategies
            lmo, f, grad! = approx_planted_point_mixed(dim, seed)
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





























# @testset "Approximate planted point - Integer" begin

#     function f(x)
#         return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
#     end
#     function grad!(storage, x)
#         @. storage = x - diffi
#     end

#     @testset "Using SCIP" begin
#         o = SCIP.Optimizer()
#         MOI.set(o, MOI.Silent(), true)
#         MOI.empty!(o)
#         x = MOI.add_variables(o, n)
#         for xi in x
#             MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
#             MOI.add_constraint(o, xi, MOI.LessThan(1.0))
#             MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
#         end
#         lmo = Boscia.MathOptBLMO(o)
        
#         x, _, result = Boscia.solve(f, grad!, lmo, branching_strategy = Boscia.PSEUDO_COST(iterations_stable,false, lmo, μ), verbose=true)

#         @test x == round.(diffi)
#         @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
#     end

#     @testset "Using Cube LMO" begin
#         int_vars = collect(1:n)

#         bounds = Boscia.IntegerBounds()
#         for i in 1:n
#             push!(bounds, (i, 0.0), :greaterthan)
#             push!(bounds, (i, 1.0), :lessthan)
#         end
#         blmo = CubeBLMO(n, int_vars, bounds)

#         x, _, result = Boscia.solve(f, grad!, blmo, branching_strategy = Boscia.PSEUDO_COST(iterations_stable,false, blmo, μ), verbose=true)


#         @test x == round.(diffi)
#         @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
#     end

#     @testset "Using Cube Simple LMO" begin
#         int_vars = collect(1:n)
#         lbs = zeros(n)
#         ubs = ones(n)

#         sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)

#         x, _, result =
#             Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, branching_strategy = Boscia.PSEUDO_COST(iterations_stable,false, sblmo, μ), verbose=true)

#         @test x == round.(diffi)
#         @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
#     end
# end


# @testset "Approximate planted point - Mixed" begin

#     function f(x)
#         return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
#     end
#     function grad!(storage, x)
#         @. storage = x - diffi
#     end

#     int_vars = unique!(rand(collect(1:n), Int(floor(n / 2))))

#     @testset "Using SCIP" begin
#         o = SCIP.Optimizer()
#         MOI.set(o, MOI.Silent(), true)
#         MOI.empty!(o)
#         x = MOI.add_variables(o, n)
#         for xi in x
#             MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
#             MOI.add_constraint(o, xi, MOI.LessThan(1.0))
#             if xi.value in int_vars
#                 MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
#             end
#         end
#         lmo = Boscia.MathOptBLMO(o)

#         x, _, result = Boscia.solve(f, grad!, blmo, branching_strategy = Boscia.PSEUDO_COST(iterations_stable,false, lmo, μ), verbose=true)

#         sol = diffi
#         sol[int_vars] = round.(sol[int_vars])
#         @test sum(isapprox.(x, sol, atol=1e-6, rtol=1e-2)) == n
#         @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
#     end

#     @testset "Using Cube LMO" begin
#         bounds = Boscia.IntegerBounds()
#         for i in 1:n
#             push!(bounds, (i, 0.0), :greaterthan)
#             push!(bounds, (i, 1.0), :lessthan)
#         end
#         blmo = CubeBLMO(n, int_vars, bounds)

#         x, _, result = Boscia.solve(f, grad!, blmo, branching_strategy = Boscia.PSEUDO_COST(iterations_stable,false, blmo, μ), verbose=true)

#         sol = diffi
#         sol[int_vars] = round.(sol[int_vars])
#         @test sum(isapprox.(x, sol, atol=1e-6, rtol=1e-2)) == n
#         @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
#     end

#     @testset "Using Cube Simple LMO" begin
#         lbs = zeros(n)
#         ubs = ones(n)

#         sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)

#         x, _, result =
#             Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, branching_strategy = Boscia.PSEUDO_COST(iterations_stable,false, sblmo, μ), verbose=true)

#         sol = diffi
#         sol[int_vars] = round.(sol[int_vars])
#         @test sum(isapprox.(x, sol, atol=1e-6, rtol=1e-2)) == n
#         @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
#     end
# end
