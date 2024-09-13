using Boscia
using Random
using Distributions
using LinearAlgebra
using FrankWolfe
using Statistics

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
m = 100
verbose = true

## A-Optimal Design Problem

A, n, N, ub = build_data(seed, m)

# sharpness constants
σ = minimum(A' * A)
a = maximum(ub) * maximum([norm(A[i,:])^2 for i=1:size(A,1)])
θ = 1/2
M = n * σ^4 / a^2


f, grad! = build_a_criterion(A, build_safe=true)
blmo = build_blmo(m, N, ub)
x0, active_set = build_start_point(A, N, ub)
z = greedy_incumbent(A, N, ub)
domain_oracle = build_domain_oracle(A, n)

x, _, result = Boscia.solve(f, grad!, blmo, active_set=active_set, start_solution=z, verbose=verbose, sharpness_exponent=θ, sharpness_constant=M, domain_oracle=domain_oracle) 

f, grad! = build_a_criterion(A, build_safe=false)
blmo = build_blmo(m, N, ub)
x0, active_set = build_start_point(A, N, ub)
z = greedy_incumbent(A, N, ub)
domain_oracle = build_domain_oracle(A, n)

x, _, result = Boscia.solve(f, grad!, blmo, active_set=active_set, start_solution=z, verbose=verbose, line_search=FrankWolfe.Secant(40, 1e-8, domain_oracle), sharpness_exponent=θ, sharpness_constant=M, domain_oracle=domain_oracle)


## D-Optimal Design Problem

A, n, N, ub = build_data(seed, m)

# sharpness constants
σ = minimum(A' * A)
a = maximum(ub) * maximum([norm(A[i,:])^2 for i=1:size(A,1)])
θ = 1/2
M = n * σ^4 / (2 * a^2)


f, grad! = build_d_criterion(A, build_safe=true)
blmo = build_blmo(m, N, ub)
x0, active_set = build_start_point(A, N, ub)
z = greedy_incumbent(A, N, ub)
domain_oracle = build_domain_oracle(A, n)

x, _, result = Boscia.solve(f, grad!, blmo, active_set=active_set, start_solution=z, verbose=verbose, sharpness_exponent=θ, sharpness_constant=M, domain_oracle=domain_oracle)

A, n, N, ub = build_data(seed, m)

f, grad! = build_d_criterion(A, build_safe=false)
blmo = build_blmo(m, N, ub)
x0, active_set = build_start_point(A, N, ub)
z = greedy_incumbent(A, N, ub)
domain_oracle = build_domain_oracle(A, n)

x, _, result = Boscia.solve(f, grad!, blmo, active_set=active_set, start_solution=z, verbose=verbose, line_search=FrankWolfe.Secant(40, 1e-8, domain_oracle), sharpness_exponent=θ, sharpness_constant=M, domain_oracle=domain_oracle)
