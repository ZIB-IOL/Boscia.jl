using Boscia
using Random
using Distributions
using LinearAlgebra
using FrankWolfe
using Statistics
using Test 

# The function building the problem data and other structures is in a separate file.
include("oed_utils.jl")

"""
    The Optimal Experiment Design Problem consists of choosing a subset of experiments
    maximising the information gain. 
    We generate the Experiment matrix A ∈ R^{mxn} randomly. The information matrix is a linear
    map X(x) = A' * diag(x) * A. There exists different information measures Φ. 
    We concentrate on the A-criterion and D-criterion, i.e. 

        Trace(X^{-1})       (A-Criterion)
    and
        -logdet(X)          (D-Criterion).

    Consequently, the optimization problems we want to solve are

    min_x  Trace( (A' * diag(x) * A)^{-1} )
    s.t.   ∑ x_i = N                            (A-Optimal Design Problem)
           0 ≤ x ≤ u

           min_x  -logdet(A' * diag(x) * A)
    s.t.   ∑ x_i = N                            (D-Optimal Design Problem)
           0 ≤ x ≤ u

    where N is our bugdet for the experiments, i.e. this is the amount of experiments
    we can perform. We set N = 3/2 * n. The upperbounds u are randomly generated. 
    
    Also, check this paper: https://arxiv.org/abs/2312.11200 and the corresponding 
    respository https://github.com/ZIB-IOL/OptimalDesignWithBoscia.

    A continuous version of the problem can be found in the examples in FrankWolfe.jl:
    https://github.com/ZIB-IOL/FrankWolfe.jl/blob/master/examples/optimal_experiment_design.jl
"""

seed = 1234
m = 40
verbose = true

## A-Optimal Design Problem

@testset "A-Optimal Design" begin

    Ex_mat, n, N, ub = build_data(seed, m)

    # sharpness constants
    σ = minimum(Ex_mat' * Ex_mat)
    λ_max = maximum(ub) * maximum([norm(Ex_mat[i,:])^2 for i=1:size(Ex_mat,1)])
    θ = 1/2
    M = n * σ^4 / λ_max^2


    f, grad! = build_a_criterion(Ex_mat, build_safe=true)
    blmo = build_blmo(m, N, ub)
    x0, active_set = build_start_point(Ex_mat, N, ub)
    z = greedy_incumbent(Ex_mat, N, ub)
    domain_oracle = build_domain_oracle(Ex_mat, n)
    heu = Boscia.Heuristic(Boscia.probability_rounding, 0.8, :probability_rounding)

    x, _, result = Boscia.solve(f, grad!, blmo, active_set=active_set, start_solution=z, verbose=verbose, sharpness_exponent=θ, sharpness_constant=M, domain_oracle=domain_oracle, custom_heuristics=[heu]) 
    sol_a = f(x)

    f, grad! = build_a_criterion(Ex_mat, build_safe=false)
    blmo = build_blmo(m, N, ub)
    x0, active_set = build_start_point(Ex_mat, N, ub)
    z = greedy_incumbent(Ex_mat, N, ub)
    domain_oracle = build_domain_oracle(Ex_mat, n)
    heu = Boscia.Heuristic(Boscia.probability_rounding, 0.6, :probability_rounding)

    x_s, _, result = Boscia.solve(f, grad!, blmo, active_set=active_set, start_solution=z, verbose=verbose, line_search=FrankWolfe.Secant(40, 1e-8, domain_oracle), sharpness_exponent=θ, sharpness_constant=M, domain_oracle=domain_oracle, custom_heuristics=[heu])

    @test isapprox(sol_a, f(x_s), atol=1e-4, rtol=1e-2)
end

## D-Optimal Design Problem

@testset "D-optimal Design" begin
    Ex_mat, n, N, ub = build_data(seed, m)

    # sharpness constants
    σ = minimum(Ex_mat' * Ex_mat)
    λ_max = maximum(ub) * maximum([norm(Ex_mat[i,:])^2 for i=1:size(Ex_mat,1)])
    θ = 1/2
    M = n * σ^4 / (2 * λ_max^2)


    f, grad! = build_d_criterion(Ex_mat, build_safe=true)
    blmo = build_blmo(m, N, ub)
    x0, active_set = build_start_point(Ex_mat, N, ub)
    z = greedy_incumbent(Ex_mat, N, ub)
    domain_oracle = build_domain_oracle(Ex_mat, n)
    heu = Boscia.Heuristic(Boscia.probability_rounding, 0.6, :probability_rounding)

    x, _, result = Boscia.solve(f, grad!, blmo, active_set=active_set, start_solution=z, verbose=verbose,  domain_oracle=domain_oracle, custom_heuristics=[heu])
    sol_a = f(x) #sharpness_exponent=θ, sharpness_constant=M,
@show x 
    Ex_mat, n, N, ub = build_data(seed, m)

    f, grad! = build_d_criterion(Ex_mat, build_safe=false)
    blmo = build_blmo(m, N, ub)
    x0, active_set = build_start_point(Ex_mat, N, ub)
    z = greedy_incumbent(Ex_mat, N, ub)
    domain_oracle = build_domain_oracle(Ex_mat, n)
    heu = Boscia.Heuristic(Boscia.probability_rounding, 0.6, :probability_rounding)

    x_s, _, result = Boscia.solve(f, grad!, blmo, active_set=active_set, start_solution=z, verbose=verbose, line_search=FrankWolfe.Secant(40, 1e-8, domain_oracle), domain_oracle=domain_oracle, custom_heuristics=[heu]) #sharpness_exponent=θ, sharpness_constant=M, 
@show x_s
@show sum(x_s-x), findall(x-> x != 0, x-x_s)
@show sum(x_s), N, findall(x-> x > 0, x_s-ub)
    @test isapprox(sol_a, f(x_s), atol=1e-4, rtol=1e-2)
end

# in interface.jl at the polishing solution
@show x_polished
        @show findall(x -> x != round.(x), x_polished[tree.root.problem.integer_variables])
