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


m = 50
n = Int(floor(m/10))
verbose = true
seed = 0xf87101b7b85e0fcc
Random.seed!(seed)

## A-Optimal Design Problem
@testset "A-Optimal Design" begin

    Ex_mat, N, ub = build_data(m, n)

    g, grad! = build_a_criterion(Ex_mat, build_safe=false)
    blmo = build_blmo(m, N, ub)
    heu = Boscia.Heuristic(Boscia.rounding_hyperplane_heuristic, 0.7, :hyperplane_aware_rounding)
    domain_oracle = build_domain_oracle(Ex_mat, n)

    # precompile
    line_search = FrankWolfe.MonotonicGenericStepsize(FrankWolfe.Adaptive(), domain_oracle)
    x0, active_set = build_start_point(Ex_mat, N, ub)
    z = greedy_incumbent(Ex_mat, N, ub)
    x, _, _ = Boscia.solve(g, grad!, blmo, active_set=active_set, start_solution=z, time_limit=10, verbose=false, domain_oracle=domain_oracle, custom_heuristics=[heu], line_search=line_search)

    # proper run with MGLS and Adaptive
    line_search = FrankWolfe.MonotonicGenericStepsize(FrankWolfe.Adaptive(), domain_oracle)
    x0, active_set = build_start_point(Ex_mat, N, ub)
    @show domain_oracle(x0), N
    @show x0
    z = greedy_incumbent(Ex_mat, N, ub)
   x, _, result = Boscia.solve(
        g, 
        grad!, 
        blmo, 
        active_set=active_set, 
        start_solution=z, 
        verbose=verbose, 
        domain_oracle=domain_oracle, 
        custom_heuristics=[heu], 
        line_search=line_search,
    ) 
    
    # Run with Secant    
    x0, active_set = build_start_point(Ex_mat, N, ub)
    z = greedy_incumbent(Ex_mat, N, ub)
    line_search = FrankWolfe.Secant(domain_oracle=domain_oracle)

    x_s, _, result_s = Boscia.solve(
        g, 
        grad!, 
        blmo, 
        active_set=active_set, 
        start_solution=z, 
        verbose=verbose, 
        domain_oracle=domain_oracle, 
        custom_heuristics=[heu], 
        line_search=line_search,
    )
@show X
@show x_s
    @test result_s[:dual_bound] <= g(x) + 1e-4
    @test result[:dual_bound] <= g(x_s) + 1e-4
    @test isapprox(g(x), g(x_s), atol=1e-3) 
end 

## D-Optimal Design Problem
@testset "D-optimal Design" begin
    Ex_mat, N, ub = build_data(m, n)

    g, grad! = build_d_criterion(Ex_mat, build_safe=false)
    blmo = build_blmo(m, N, ub)
    heu = Boscia.Heuristic(Boscia.rounding_hyperplane_heuristic, 0.7, :hyperplane_aware_rounding)
    domain_oracle = build_domain_oracle(Ex_mat, n)

    # precompile
    line_search = FrankWolfe.MonotonicGenericStepsize(FrankWolfe.Adaptive(), domain_oracle)
    x0, active_set = build_start_point(Ex_mat, N, ub)
    z = greedy_incumbent(Ex_mat, N, ub)
    x, _, _ = Boscia.solve(g, grad!, blmo, active_set=active_set, start_solution=z, time_limit=10, verbose=false, domain_oracle=domain_oracle, custom_heuristics=[heu], line_search=line_search)

    # proper run with MGLS and Adaptive
    line_search = FrankWolfe.MonotonicGenericStepsize(FrankWolfe.Adaptive(), domain_oracle)
    x0, active_set = build_start_point(Ex_mat, N, ub)
    z = greedy_incumbent(Ex_mat, N, ub)
    x, _, result = Boscia.solve(
        g, 
        grad!, 
        blmo, 
        active_set=active_set, 
        start_solution=z, 
        verbose=verbose, 
        domain_oracle=domain_oracle, 
        custom_heuristics=[heu], 
        line_search=line_search,
    )
    
    # Run with Secant    
    x0, active_set = build_start_point(Ex_mat, N, ub)
    z = greedy_incumbent(Ex_mat, N, ub)
    line_search = FrankWolfe.Secant(domain_oracle=domain_oracle)

    x_s, _, result_s = Boscia.solve(
        g, 
        grad!, 
        blmo, 
        active_set=active_set, 
        start_solution=z, 
        verbose=verbose, 
        domain_oracle=domain_oracle, 
        custom_heuristics=[heu], 
        line_search=line_search,
    )

    @test result_s[:dual_bound] <= g(x)
    @test result[:dual_bound] <= g(x_s)
    @test isapprox(g(x), g(x_s), rtol=1e-2) 
end

