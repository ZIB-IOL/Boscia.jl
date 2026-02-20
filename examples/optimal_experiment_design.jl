using Boscia
using Random
using Distributions
using LinearAlgebra
using FrankWolfe
using Statistics
using Test
using StableRNGs

println("\nOptimal Experiment Design Example")

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

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
n = Int(floor(m / 10))
verbose = true

## A-Optimal Design Problem
@testset "A-Optimal Design" begin

    Ex_mat, N, ub = build_data(rng, m, n)

    g, grad! = build_a_criterion(Ex_mat, build_safe=false)
    blmo = build_blmo(m, N, ub)
    heu = Boscia.Heuristic(Boscia.rounding_hyperplane_heuristic, 0.7, :hyperplane_aware_rounding)
    domain_oracle = build_domain_oracle(Ex_mat, n)
    domain_point =
        build_domain_point_function(domain_oracle, Ex_mat, N, collect(1:m), fill(0.0, m), ub)

    # precompile
    line_search = FrankWolfe.Adaptive(domain_oracle=domain_oracle)
    x0, active_set = build_start_point(Ex_mat, N, ub)
    z = greedy_incumbent(Ex_mat, N, ub)
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:time_limit] = 10
    settings.branch_and_bound[:verbose] = false
    settings.branch_and_bound[:start_solution] = z
    settings.heuristic[:custom_heuristics] = [heu]
    settings.frank_wolfe[:line_search] = line_search
    settings.domain[:active_set] = active_set
    settings.domain[:domain_oracle] = domain_oracle
    settings.domain[:find_domain_point] = domain_point
    x, _, _ = Boscia.solve(g, grad!, blmo, settings=settings)

    # proper run with MGLS and Adaptive
    line_search = FrankWolfe.Adaptive(domain_oracle=domain_oracle)
    x0, active_set = build_start_point(Ex_mat, N, ub)
    z = greedy_incumbent(Ex_mat, N, ub)
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = verbose
    settings.branch_and_bound[:start_solution] = z
    settings.heuristic[:custom_heuristics] = [heu]
    settings.frank_wolfe[:line_search] = line_search
    settings.domain[:active_set] = active_set
    settings.domain[:domain_oracle] = domain_oracle
    settings.domain[:find_domain_point] = domain_point
    x, _, result = Boscia.solve(g, grad!, blmo, settings=settings)

    # Run with Secant    
    x0, active_set = build_start_point(Ex_mat, N, ub)
    z = greedy_incumbent(Ex_mat, N, ub)
    line_search = FrankWolfe.Secant(domain_oracle=domain_oracle)
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = verbose
    settings.branch_and_bound[:start_solution] = z
    settings.heuristic[:custom_heuristics] = [heu]
    settings.frank_wolfe[:line_search] = line_search
    settings.domain[:active_set] = active_set
    settings.domain[:domain_oracle] = domain_oracle
    settings.domain[:find_domain_point] = domain_point
    x_s, _, result_s = Boscia.solve(g, grad!, blmo, settings=settings)

    @test result_s[:dual_bound] <= g(x) + 1e-3
    @test result[:dual_bound] <= g(x_s) + 1e-3
    @test isapprox(g(x), g(x_s), atol=1e-3)
end

## D-Optimal Design Problem
@testset "D-optimal Design" begin
    Ex_mat, N, ub = build_data(rng, m, n)

    g, grad! = build_d_criterion(Ex_mat, build_safe=false)
    blmo = build_blmo(m, N, ub)
    heu = Boscia.Heuristic(Boscia.rounding_hyperplane_heuristic, 0.7, :hyperplane_aware_rounding)
    domain_oracle = build_domain_oracle2(Ex_mat, n)
    domain_point =
        build_domain_point_function(domain_oracle, Ex_mat, N, collect(1:m), fill(0.0, m), ub)

    # precompile
    line_search = FrankWolfe.Adaptive(domain_oracle=domain_oracle)
    x0, active_set = build_start_point(Ex_mat, N, ub)
    z = greedy_incumbent(Ex_mat, N, ub)
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:time_limit] = 10
    settings.branch_and_bound[:verbose] = false
    settings.branch_and_bound[:start_solution] = z
    settings.heuristic[:custom_heuristics] = [heu]
    settings.frank_wolfe[:line_search] = line_search
    settings.domain[:active_set] = active_set
    settings.domain[:domain_oracle] = domain_oracle
    settings.domain[:find_domain_point] = domain_point
    x, _, _ = Boscia.solve(g, grad!, blmo, settings=settings)

    # proper run with MGLS and Adaptive
    line_search = FrankWolfe.Adaptive(domain_oracle=domain_oracle)
    x0, active_set = build_start_point(Ex_mat, N, ub)
    z = greedy_incumbent(Ex_mat, N, ub)
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = verbose
    settings.branch_and_bound[:start_solution] = z
    settings.heuristic[:custom_heuristics] = [heu]
    settings.frank_wolfe[:line_search] = line_search
    settings.domain[:active_set] = active_set
    settings.domain[:domain_oracle] = domain_oracle
    settings.domain[:find_domain_point] = domain_point
    x, _, result = Boscia.solve(g, grad!, blmo, settings=settings)

    # Run with Secant    
    x0, active_set = build_start_point(Ex_mat, N, ub)
    z = greedy_incumbent(Ex_mat, N, ub)
    line_search = FrankWolfe.Secant(domain_oracle=domain_oracle)
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = verbose
    settings.branch_and_bound[:start_solution] = z
    settings.heuristic[:custom_heuristics] = [heu]
    settings.frank_wolfe[:line_search] = line_search
    settings.domain[:active_set] = active_set
    settings.domain[:domain_oracle] = domain_oracle
    settings.domain[:find_domain_point] = domain_point
    x_s, _, result_s = Boscia.solve(g, grad!, blmo, settings=settings)

    @test result_s[:dual_bound] <= g(x)
    @test result[:dual_bound] <= g(x_s)
    @test isapprox(g(x), g(x_s), rtol=1e-2)
end
