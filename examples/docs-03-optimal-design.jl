# Optimal Design of Experiments
#
# This example shows the A-Optimal and D-Optimal Design of Experiments problems.
# To quantify information, we use the Fisher information matrix: 
# ```math
# X(x) = A' * \text{diag}(x) * A
# ```
# where each row of $A$ corresponds to an experiment.
# For the D-criterion, the objective is the negative log determinant of the Fisher information matrix.
# The objective associated with the A-criterion is the trace of the inverse of the Fisher information matrix.

using Boscia
using Random
using Distributions
using LinearAlgebra
using FrankWolfe
using Statistics
using Test
using StableRNGs

println("\nDocumentation Example 03: Optimal Design of Experiments")

seed = rand(UInt64)
@show seed  #seed = 0x7be8a16f815cd122
rng = StableRNG(seed)

# ## Experiment matrix and objectives

# We generate the experiment matrix $A$ randomly.
m = 50
n = Int(floor(m / 10))
N = round(Int, 1.5 * n)

B = rand(rng, m, n)
B = B' * B
@assert isposdef(B)
D = MvNormal(randn(rng, n), B)

const A = rand(D, m)'
@assert rank(A) == n

# Next, we define the two criteria and their gradients.
# The A-criterion is::
# ```math
# f_a(x) = \text{Tr}|left(X(x)^{-1}/right)
# ```
μ = 1e-4
function f_a(x)
    X = transpose(A) * diagm(x) * A + Matrix(μ * I, n, n)
    X = Symmetric(X)
    U = cholesky(X)
    X_inv = U \ I
    return LinearAlgebra.tr(X_inv) 
end

function grad_a!(storage, x)
    X = transpose(A) * diagm(x) * A + Matrix(μ * I, n, n)
    X = Symmetric(X * X)
    F = cholesky(X)
    for i in 1:length(x)
        storage[i] = LinearAlgebra.tr(-(F \ A[i, :]) * transpose(A[i, :])) 
    end
    return storage
end
# The D-criterion is:
# ```math
# f_d(x) = -\log(\det(X(x)))
# ```
function f_d(x)
    X = transpose(A) * diagm(x) * A
    X = Symmetric(X)
    return float(-log(det(X)))
end

function grad_d!(storage, x)
    X = transpose(A) * diagm(x) * A
    X = Symmetric(X)
    F = cholesky(X)
    for i in 1:length(x)
        storage[i] = LinearAlgebra.tr(-(F \ A[i, :]) * transpose(A[i, :]))
    end
    return storage
end

# ## Domain Issues
# The feasible region is a scaled and truncated probability simplex.
# ```math
# S = \{x \in \mathbb{R}^n, 0 \leq x \leq u, \sum_{i=1}^n x_i = N\}
# ```
# where $N$ is the budget for the experiments and $u$ are upper bounds.
ub = floor(N/3)
u = rand(rng, 1.0:ub, m)
simplex_lmo = Boscia.ProbabilitySimplexLMO(N)
lmo = Boscia.ManagedLMO(simplex_lmo, fill(0.0, m), u, collect(1:m), m)


# An issue arising from this is that the feasible region and the domain of the objectives don't completely overlap.
# Thus, we cannot start Boscia and by extension Frank-Wolfe at a random start point. 
# Also, during the line search, we have to be careful to pick a step size that does not lead to the iterate leaving the domain.
# To address this problem, we first need to define a domain oracle tht given a point $x$ returns true if $x$ is feasible.
# There are different ways to check domain feasibility, here we chose to test if the activated rows of $A$ are linearly independent and span the $\mathbb{R}^n$.
function domain_oracle(x)
    S = findall(x -> !iszero(x), x)
    return length(S) >= n && rank(A[S, :]) == n
end

# Even if we start Boscia with a domain feasible point, we might end up with domain infeasible points later in the tree.
# Observe that the vertices in the active set are not necessarily domain feasible.
# Therefore, while branching, we can have initial points that are not domain feasible.
# To address this, we need to define a domain point function that given the current node bounds returns a domain feasible point respecting the bounds, if possible. 
function domain_point(local_bounds)
    # Find n linearly independent rows of A to build the starting point.
    function linearly_independent_rows(A; u=fill(1, size(A, 1)))
        S = []
        m, n = size(A)
        for i in 1:m
            if iszero(u[i])
                continue
            end
            S_i = vcat(S, i)
            if rank(A[S_i, :]) == length(S_i)
                S = S_i
            end
            if length(S) == n # we only n linearly independent points
                return S
            end
        end
        return S # then x= zeros(m) and x[S] = 1
    end

    # Add to the smallest value of x while respecting the upper bounds u.
    function add_to_min(x, u)
        perm = sortperm(x)
        for i in perm
            if x[i] < u[i]
                x[i] += 1
                break
            end
        end
        return x
    end

    lb = fill(0.0, m)
    ub = copy(u)
    x = zeros(m)
    for idx in 1:m
        if haskey(local_bounds.lower_bounds, idx)
            lb[idx] = max(0.0, local_bounds.lower_bounds[idx])
        end
        if haskey(local_bounds.upper_bounds, idx)
            ub[idx] = min(u[idx], local_bounds.upper_bounds[idx])
        end
    end
    # Node itself infeasible
    if sum(lb) > N
        return nothing
    end
    # No intersection between node and domain
    if !domain_oracle(ub)
        return nothing
    end
    x = lb

    # build domain feasible point
    S = linearly_independent_rows(A, u=(.!(iszero.(ub))))
        while sum(x) <= N
            if sum(x) == N
                if domain_oracle(x)
                    return x
                else
                    @warn "Domain feasible point not found."
                    return nothing
                end
            end
            if !iszero(x[S] - ub[S])
                y = add_to_min(x[S], ub[S])
                x[S] = y
            else
                x = add_to_min(x, ub)
            end
        end
    return x
end
# Note that the domain point function does not necessarily have to return an integer point. 
# The generated point is used to solve a min distance problem over the feasible region to move the current iterate closer to the domain.
# To that end, the domain point should not be at the boundary of the domain as this can lead to numerical issues later in the node solve.

# ## Build initial start point
# We can use the same principal to generate an initial start point for Boscia.
# Note that Boscia expects the initial point to be given via an active set.
intial_bounds = Boscia.IntegerBounds(fill(0.0, m), u, collect(1:m))
x0 = domain_point(intial_bounds)
f_help(x) = 1 / 2 * LinearAlgebra.norm(x - x0)^2
grad_help!(storage, x) = storage .= x - x0
v0 = compute_extreme_point(lmo, collect(1.0:m))

# We do not need to solve this problem to optimality.
# However, we do not want to stop as soon as we reach the domain because this can lead to numerical issues later in the node solve.
# Therefore, we count the iteration after entering the domain and stop if we have not found a feasible point after 5 iterations.
function build_inner_callback()
    domain_counter = 0
    return function inner_callback(state, active_set, kwargs...)
        # Once we find a domain feasible point, we count the iteration
        # and stop if we have not found a feasible point after 5 iterations..
        if domain_oracle(state.x)
            if domain_counter > 10
                return false
            end
            domain_counter += 1
        end
    end
end

inner_callback = build_inner_callback()

x, _, _, _, _, _, active_set = FrankWolfe.blended_pairwise_conditional_gradient(
    f_help,
    grad_help!,
    lmo,
    v0,
    callback=inner_callback,
    lazy=true,
)

@show N, u

# Now we can use Boscia to solve the problem.
# As line search, we use the Secant method which receives the domain oracle as input.
# We also set some heuristics to be used during the node solve by specifying a probability for each heuristic.
settings = Boscia.create_default_settings()
settings.branch_and_bound[:verbose] = false
settings.branch_and_bound[:time_limit] = 10.0
settings.domain[:active_set] = copy(active_set) # this will be overwritten by Boscia during the solve
settings.domain[:domain_oracle] = domain_oracle
settings.domain[:find_domain_point] = domain_point
settings.domain[:depth_domain] = 10
settings.heuristic[:hyperplane_aware_rounding_prob] = 0.7
settings.heuristic[:rounding_lmo_01_prob] = 0.5
settings.frank_wolfe[:line_search] = FrankWolfe.Secant(domain_oracle=domain_oracle)
settings.frank_wolfe[:lazy] = true

# First, we are calling the algorithm for a few seconds for precompilation.
x_a, _, _ = Boscia.solve(f_a, grad_a!, lmo, settings=settings)

settings.branch_and_bound[:verbose] = true
settings.branch_and_bound[:time_limit] = Inf
settings.domain[:active_set] = copy(active_set) # this will be overwritten by Boscia during the solve

x_a, _, result_a = Boscia.solve(f_a, grad_a!, lmo, settings=settings)

settings = Boscia.create_default_settings()
settings.branch_and_bound[:verbose] = false
settings.branch_and_bound[:time_limit] = 10.0
settings.domain[:active_set] = copy(active_set) # this will be overwritten by Boscia during the solve
settings.domain[:domain_oracle] = domain_oracle
settings.domain[:find_domain_point] = domain_point
settings.domain[:depth_domain] = 10
settings.heuristic[:hyperplane_aware_rounding_prob] = 0.7
settings.heuristic[:rounding_lmo_01_prob] = 0.5
settings.frank_wolfe[:line_search] = FrankWolfe.Secant(domain_oracle=domain_oracle)
settings.frank_wolfe[:lazy] = true

x_d, _, _ = Boscia.solve(f_d, grad_d!, lmo, settings=settings)

settings.branch_and_bound[:verbose] = true
settings.branch_and_bound[:time_limit] = Inf
settings.domain[:active_set] = copy(active_set) # this will be overwritten by Boscia during the solve

x_d, _, result_d = Boscia.solve(f_d, grad_d!, lmo, settings=settings)

# ## Plotting the progress
#=
using PyPlot
# Load plotting utilities
include("plot_utilities.jl")

# Create plots for A-criterion (if solved)
if @isdefined(result_a)
    filename_a = "oed_A_criterion_m$(m)_seed_$(seed).pdf"
    fig_a = plot_bounds_progress(
        result_a,
        filename_a,
        title_prefix="A-Criterion",
        use_latex=true,
        font_size=11,
        linewidth=2,
    )
    display(fig_a)
end

# Create plots for D-criterion
filename_d = "oed_D_criterion_m$(m)_seed_$(seed).pdf"
fig_d = plot_bounds_progress(
    result_d,
    filename_d,
    title_prefix="D-Criterion",
    use_latex=true,
    font_size=11,
    linewidth=2,
)
display(fig_d)
=#
