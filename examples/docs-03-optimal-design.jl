# # Optimal Design of Experiments
#
# Given a large set of experiments, the *Optimal Design of Experiments (OEDP)* problem aims to 
# select a subset of experiments that maximizes the information gain.
# Formally, we are given a experiment matrix $A \in \mathbb{R}^{m \times n}$ encoding the
# experiments data where $m$ denotes the number of experiments, $n$ denotes the number of parameters
# and generally $m \gg n$.
# To quantify information, we utilize the Fisher information matrix defined as: 
# ```math
# X(x) = A' * \text{diag}(x) * A
# ```
# where $x \in \mathbb{Z}^n$ is the design vector.
# There exist multiple information measures, i.e. a function that maps the Fisher information matrix to a real number,
# for a comprehensive overview, see the book by Friedrich Pukelsheim titled "Optimal Design of Experiments".
# For this example, we consider the A-criterion and D-criterion.

# ## Imports and problem setup
# We start by generating the experiment matrix $A$ randomly.
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

m = 50
n = Int(floor(m / 10))
N = round(Int, 1.5 * n)

B = rand(rng, m, n)
B = B' * B
@assert isposdef(B)
const D = MvNormal(randn(rng, n), B)

const A = rand(D, m)'
@assert rank(A) == n

# Next, we define the two criteria and their gradients.
# The A-criterion is defined as:
# ```math
# f_a(x) = \text{Tr}\left(X(x)^{-1}\right)
# ```
# so the trace of the inverse of the Fisher information matrix.
function f_a(x)
    X = transpose(A) * diagm(x) * A
    X = Symmetric(X)
    U = cholesky(X)
    X_inv = U \ I
    return LinearAlgebra.tr(X_inv)
end

function grad_a!(storage, x)
    X = transpose(A) * diagm(x) * A
    X = Symmetric(X * X)
    F = cholesky(X)
    for i in 1:length(x)
        storage[i] = LinearAlgebra.tr(-(F \ A[i, :]) * transpose(A[i, :]))
    end
    return storage
end
# The D-criterion is defined as:
# ```math
# f_d(x) = -\log(\det(X(x))).
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

# ## Issue: Restricted function domain
# The feasible region is a scaled and truncated probability simplex.
# ```math
# \Delta = \left\{x \in \mathbb{R}^n, 0 \leq x \leq u, \sum_{i=1}^n x_i = N\right\}
# ```
# where $N$ is the budget and $u$ are upper bounds.

# An issue arising in OEDP is that the objective functions and their gradients are not well defined
# over the entire feasible region.
# Note that for both the A-criterion and D-criterion, the associated Fisher information matrix has to be positive definite.
# Thus, we cannot start Boscia, and by extension Frank-Wolfe, at an arbitrary start point. 
# Additionally, we have to be careful not to leave the domain during computation of the step size for Frank-Wolfe in the line search.
# To address this problem, we first need to define a domain oracle that given a point $x$ returns true if $x$ is feasible.
# There are different ways to check domain feasibility, we choose to test if the minimum 
# eigenvalue i strictly positive (up to numerical tolerance).
ub = floor(N/3)
u = rand(rng, 1.0:ub, m)
simplex_lmo = Boscia.ProbabilitySimplexLMO(N)
lmo = Boscia.ManagedLMO(simplex_lmo, fill(0.0, m), u, collect(1:m), m)

function domain_oracle(x)
    X = transpose(A) * diagm(x) * A
    X = Symmetric(X)
    #return LinearAlgebra.isposdef(X)
    return minimum(eigvals(X)) > sqrt(eps())
end

# Next, we have to ensure that the start points of the child nodes are also domain feasible.
# Observe that the vertices in the active set are not necessarily domain feasible.
# Therefore, while branching, we can have initial points that are not domain feasible.
# To address this, we need to define a domain point function that given the current node bounds returns 
# a domain feasible point respecting the bounds, if possible. 
# For OEDP, we start by setting $x$ equal to the current lower bounds and 
# finding n linearly independent rows of $A$.
# If $x$ does not yet satisfy the knapsack constraint, we increase the values of $X$, first by sampling 
# from the linearly independent rows and then by adding 1 to the smallest value of $x$ while respecting the upper bounds $u$.
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
function domain_point(local_bounds)
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
    if sum(lb) > N
        return nothing
    end
    if !domain_oracle(ub)
        return nothing
    end
    x = lb
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
# Note that the domain point function does not necessarily has to return an integral feasible point. 
# The generated point is used to solve a projection problem over the feasible region to move the current iterate into the domain.
# To that end, the generated point should not be at the boundary of the domain as this can lead to numerical issues later in the node solve.

# ## Generating the initial start point
# We can use the same principal Boscia uses to generate domain feasible starting points for the 
# child nodes to generate an initial start point.
# To this end, we use the `find_domain_point` function to generate a domain feasible point respecting the bounds.
# The projection problem can be solved using Frank-Wolfe.
# Note that Boscia expects the initial point to be given via an active set.
initial_bounds = Boscia.IntegerBounds(fill(0.0, m), u, collect(1:m))
x0 = domain_point(initial_bounds)
f_help(x) = 1 / 2 * LinearAlgebra.norm(x - x0)^2
grad_help!(storage, x) = storage .= x - x0

# We do not need to solve this problem to optimality.
# However, we do not want to stop as soon as we reach the domain because this can lead to numerical issues later in the node solve.
# Therefore, we count the iteration after entering the domain and stop if we have not found a feasible point after 5 iterations.
function build_inner_callback()
    domain_counter = 0
    return function inner_callback(state, active_set, kwargs...)
        if domain_oracle(state.x)
            if domain_counter > 10
                return false
            end
            domain_counter += 1
        end
    end
end

inner_callback = build_inner_callback()
v0 = compute_extreme_point(lmo, collect(1.0:m))

_, _, _, _, _, _, active_set = FrankWolfe.blended_pairwise_conditional_gradient(
    f_help,
    grad_help!,
    lmo,
    v0,
    callback=inner_callback,
    lazy=true,
)

# ## Calling Boscia

# Now we have everything set up and ready to use Boscia to solve the problem.
# As line search, we use the Secant method which receives the domain oracle as input.
# We also set some heuristics to be used during the node solve by specifying a probability for each heuristic.
settings = Boscia.create_default_settings()
settings.branch_and_bound[:verbose] = true
settings.domain[:active_set] = copy(active_set) # this will be overwritten by Boscia during the solve
settings.domain[:domain_oracle] = domain_oracle
settings.domain[:find_domain_point] = domain_point
settings.domain[:depth_domain] = 10
settings.heuristic[:hyperplane_aware_rounding_prob] = 0.7
settings.heuristic[:rounding_lmo_01_prob] = 0.5
settings.frank_wolfe[:line_search] = FrankWolfe.Secant(domain_oracle=domain_oracle)
settings.frank_wolfe[:lazy] = true

x_a, _, _ = Boscia.solve(f_a, grad_a!, lmo, settings=settings)


settings = Boscia.create_default_settings()
settings.branch_and_bound[:verbose] = true
settings.domain[:active_set] = copy(active_set)
settings.domain[:domain_oracle] = domain_oracle
settings.domain[:find_domain_point] = domain_point
settings.domain[:depth_domain] = 10
settings.heuristic[:hyperplane_aware_rounding_prob] = 0.7
settings.heuristic[:rounding_lmo_01_prob] = 0.5
settings.frank_wolfe[:line_search] = FrankWolfe.Secant(domain_oracle=domain_oracle)
settings.frank_wolfe[:lazy] = true

x_d, _, _ = Boscia.solve(f_d, grad_d!, lmo, settings=settings)
