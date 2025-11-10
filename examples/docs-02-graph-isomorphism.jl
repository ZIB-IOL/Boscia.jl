# # Graph Isomorphism Problem
#
# This example shows how to use Boscia to certify whether two graphs are isomorphic.
# Given adjacency matrices A and B, the graphs are isomorphic if and only if there exists a
# permutation matrix $X \in \mathcal{P}_n$ such that:
# ```math
#   X * A = B * X
# ```
# where $\mathcal{P}_n$ denotes the set of permutation matrices.
# Equivalently, we consider the optimization problem
# ```math
#   \min_{X\in \mathcal{P}_n} f(X) = \| X A - B X \|_F^2,
# ```
# whose optimum is exactly 0 if and only if the graphs are isomorphic. We solve over the Birkhoff
# polytope (convex hull of permutation matrices) with a branch-and-bound scheme
# plus Frank–Wolfe in the nodes; the lower bound allows pruning, and a zero
# incumbent certifies isomorphism.

# ## Imports and graph generation utilities
# We begin by importing the packages used in this example.  
using Boscia
using Random
using SparseArrays
using FrankWolfe
using Bonobo
using CSV
using StableRNGs
using CombinatorialLinearOracles
const CLO = CombinatorialLinearOracles

println("\nDocumentation Example 02: Graph Isomorphism Problem")

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

# To create test instances, we provide two helper functions that construct
# pairs of graphs with either matching or mismatching structure.
#
# **Isomorphic case:**  
# Given an adjacency matrix A, we sample a permutation matrix P and form  
# B = P A P′. This produces a graph that is isomorphic to A by construction.
function randomIsomorphic(A)
    n = size(A, 1)
    p = randperm(n)
    P = sparse(1:n, p, ones(Float64, n), n, n)
    B = P * A * P'
    return B, P
end

# **Non-isomorphic case:**  
# To obtain a simple counterexample, we toggle a single undirected edge of A
# to produce B. Such a small perturbation typically breaks isomorphism while
# preserving symmetry of the adjacency matrix.
function randomNonIsomorphic(A::AbstractMatrix)
    B = copy(A)
    n = size(B, 1)
    i = rand(1:(n-1))
    j = rand((i+1):n)
    B[i, j] = 1 - B[i, j]
    B[j, i] = B[i, j]
    return B
end

# For this example, we work with the Petersen graph, provided as a CSV file
# containing its adjacency matrix.  
# After loading A, we generate an isomorphic graph B using the routine above.
path = joinpath(@__DIR__, "Petersen.csv")
rows = [collect(Int, r) for r in CSV.File(path; header=false, types=Int)]
const A = sparse(reduce(vcat, (permutedims(r) for r in rows)))
n = size(A, 1)

B, P = randomIsomorphic(Matrix(A))
B = sparse(B)

# ## Objective and gradient
#
# To measure how well a matrix X satisfies the relation X A = B X, we minimize the
# Frobenius norm of the mismatch:
#
# ```math
# f(X) = \lVert X A - B X \rVert_F^{2}.
# ```
#
# The gradient has the form
#
# ```math
# \nabla f(X)
#   = 2\,(X A - B X)\,A^{\top}
#     - 2\,B^{\top}(X A - B X).
# ```
#
# In the implementation, X is stored in vectorized form for compatibility with the solver.
function f(x)
    X = reshape(x, n, n)
    R = X * A - B * X
    return sum(abs2, R)
end

function grad!(storage, x)
    X = reshape(x, n, n)
    grad_matrix = 2 * (X * A - B * X) * A' - 2 * B' * (X * A - B * X)
    return storage .= vec(grad_matrix)
end

# ## Linear Minimization Oracle (LMO)
#
# The feasible region of the optimization is the Birkhoff polytope, the convex hull of
# permutation matrices.  
# We use the Birkhoff LMO provided by 
# [CombinatorialLinearOracles](https://github.com/ZIB-IOL/CombinatorialLinearOracles.jl), which
# performs the required linear subproblem via the Hungarian algorithm.
blmo = CLO.BirkhoffLMO(n, collect(1:(n^2)))

# ## Branching & pruning callbacks
#
# We define callback routines to steer the branch-and-bound process.
#
# **Branch callback:**  
# At a given node, if the node’s lower bound is already strictly positive,
# no permutation can achieve an objective value of zero in that subtree.
# On such nodes, we do not need to branch further..
function build_branch_callback()
    return function (tree, node, vidx::Int)
        x = Bonobo.get_relaxed_values(tree, node)
        primal = tree.root.problem.f(x)
        lower_bound = primal - node.dual_gap
        if lower_bound > 0.0 + eps()
            println("No need to branch here. Node lower bound already positive.")
        end
        valid_lower = lower_bound > 0.0 + eps()
        return valid_lower, valid_lower
    end
end

# **Tree callback:**  
# The search can be terminated early under either of two conditions:
# 1. The current incumbent reaches objective value of 0.0, certifying isomorphism.
# 2. The lower bound of the B&B tree becomes strictly positive, implying that no
#    permutation satisfies X A = B X, and thus the graphs are not isomorphic.
function build_tree_callback()
    return function (tree, node; worse_than_incumbent=false, node_infeasible=false, lb_update=false)
        if isapprox(tree.incumbent, 0.0, atol=eps())
            tree.root.problem.solving_stage = Boscia.USER_STOP
            println("Optimal solution found.")
        end
        if Boscia.tree_lb(tree::Bonobo.BnBTree) > 0.0 + eps()
            tree.root.problem.solving_stage = Boscia.USER_STOP
            println("Tree lower bound already positive. No solution possible.")
        end
    end
end

# ## Neighborhood heuristic over the Birkhoff polytope
#
# We include a simple neighborhood heuristic that generates a few alternative
# permutations around the current incumbent.  
# This provides additional candidates that the solver may consider during the
# search.  
#
# We generate k = ⌊√n⌋ neighbor candidates during each invocation of the heuristic.
function random_k_neighbor_matrix(
    tree::Bonobo.BnBTree,
    blmo::Boscia.TimeTrackingLMO,
    x,
    k::Int,
    use_mip=false,
)
    P = tree.incumbent_solution.solution
    n0 = size(P, 1)
    n = Int(sqrt(n0))
    P = reshape(P, n, n)
    new_P = copy(P)
    Ps = []
    for _ in 1:k
        i, j = rand(1:n, 2)
        while i == j
            j = rand(1:n)
        end
        col_i = findfirst(x -> x == 1, new_P[i, :])
        col_j = findfirst(x -> x == 1, new_P[j, :])
        new_P[i, col_i] = 0
        new_P[i, col_j] = 1
        new_P[j, col_j] = 0
        new_P[j, col_i] = 1
        new_p = use_mip ? vec(new_P) : sparsevec(vec(new_P))
        push!(Ps, new_p)
    end
    return Ps, false
end

k = Int(round(sqrt(n)))
swap_heu = Boscia.Heuristic(
    (tree, blmo, x) -> random_k_neighbor_matrix(tree, blmo, x, k, false),
    1.0,
    :swap,
)


# ## Solver configuration and solve call
#
# We configure Boscia with the problem-specific callbacks, the swap-based
# neighborhood heuristic, and a decomposition-invariant Frank-Wolfe method
# using a secant line search.
settings = Boscia.create_default_settings()
settings.branch_and_bound[:verbose] = true
settings.branch_and_bound[:print_iter] = 10
settings.branch_and_bound[:bnb_callback] = build_tree_callback()
settings.branch_and_bound[:branch_callback] = build_branch_callback()
settings.heuristic[:custom_heuristics] = [swap_heu]
settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()
settings.frank_wolfe[:line_search] = FrankWolfe.Secant()
settings.frank_wolfe[:lazy] = true
settings.frank_wolfe[:max_fw_iter] = 1000

# We now call `Boscia.solve` with the objective, gradient, and Birkhoff LMO.
# If A and B are isomorphic, the solver should identify a permutation matrix
# X with objective value f(X) = 0.
x, _, result = Boscia.solve(f, grad!, blmo, settings=settings)

# A successful solve provides a permutation matrix X such that:
#
# ```math
# A \approx X^{\top} B X .
# ```
#
# This equality certifies that the two graphs are isomorphic.
X = reshape(x, n, n)
@assert A ≈ X' * B * X
println("Certificate verified: graphs are isomorphic (A ≈ X' * B * X)")


# ## Complement: Non-isomorphic case
#
# To certify non-isomorphism, we can replace B by a perturbed version (e.g., by toggling an edge).
# In that case, no permutation satisfies X A = B X, and the optimization yields a strictly positive
# lower bound:
#
# ```math
# \text{dual bound} \;>\; 0 .
# ```

B = randomNonIsomorphic(A)
x, _, result = Boscia.solve(f, grad!, blmo, settings = settings)
@assert result[:dual_bound] > 0.0
println("Graphs are not isomorphic (lower bound > 0)")