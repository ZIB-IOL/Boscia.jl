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
#
# We also add a simple neighborhood heuristic that (randomly) swaps two rows'
# 1-positions in a permutation matrix to explore nearby vertices.
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
# ## Generate a random isomorphic graph
# Given an adjacency matrix A, this function creates a random permutation
# matrix P and returns the permuted adjacency matrix B = P * A * P'.
# The resulting graphs A and B are isomorphic, and P encodes the vertex
# relabeling between them.
function randomIsomorphic(A)
    n = size(A, 1)
    p = randperm(n)
    P = sparse(1:n, p, ones(Float64, n), n, n)
    B = P * A * P'
    return B, P
end

# ## Generate a random non-isomorphic graph
# Given an adjacency matrix A, this function randomly toggles one edge 
# (adds or removes it) to create a new graph B. The resulting graph B 
# is symmetric and typically non-isomorphic to A.
function randomNonIsomorphic(A::AbstractMatrix)
    B = copy(A)
    n = size(B, 1)
    i = rand(1:(n-1))
    j = rand((i+1):n)
    B[i, j] = 1 - B[i, j]
    B[j, i] = B[i, j]
    return B
end

# ## Neighborhood heuristic over the Birkhoff polytope
# We implement a simple k-swap heuristic that takes the current incumbent permutation,
# selects two distinct rows i≠j, finds the 1-entry in each row, and swaps their column
# positions—producing a neighboring permutation matrix. Repeating k times yields k
# candidate neighbors.
function random_k_neighbor_matrix(
    tree::Bonobo.BnBTree,
    blmo::Boscia.TimeTrackingLMO,
    x,
    k::Int,
    use_mip = false,
)
    P = tree.incumbent_solution.solution
    n0 = size(P, 1)
    n = Int(sqrt(n0))
    P = reshape(P, n, n)
    new_P = copy(P)
    Ps = []
    for _ = 1:k
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

# ## Graph data
# For this self-contained example, we use the Petersen graph as A.
# We then create B either as an isomorphic graph via a random permutation,
# or as a non-isomorphic graph by randomly toggling one edge.
path = joinpath(@__DIR__, "Petersen.csv")
rows = [collect(Int, r) for r in CSV.File(path; header = false, types = Int)]
const A = sparse(reduce(vcat, (permutedims(r) for r in rows)))
n = size(A, 1)

# --- Isomorphic case ---
B, P = randomIsomorphic(Matrix(A))
B = sparse(B)

# # --- Non-isomorphic case ---
# B = randomNonIsomorphic(A)

# ## Objective and gradient
# We represent X as a vector x ∈ ℝ^{n^2}, reshaped into X ∈ ℝ^{n×n} in the routines.
# Objective:
#     f(x) = || X A - B X ||_F^2
# Gradient (by standard matrix calculus):
#     ∇f(X) = 2 (X A - B X) A' - 2 B' (X A - B X)
# We return/accept vectorized storage for Boscia/Frank–Wolfe.
function f(x)
    X = reshape(x, n, n)
    R = X * A - B * X
    return sum(abs2, R)
end

function grad!(storage, x)
    X = reshape(x, n, n)
    grad_matrix = 2 * (X * A - B * X) * A' - 2 * B' * (X * A - B * X)
    storage .= vec(grad_matrix)
end

# ## Linear Minimization Oracle (LMO)
# We use the Birkhoff LMO from CombinatorialLinearOracles, i.e., linear optimization over
# the Birkhoff polytope is solved by the Hungarian algorithm.
blmo = CLO.BirkhoffLMO(n, collect(1:(n^2)))

# ## Branching & pruning callbacks
# **Branch callback**: If the node’s lower bound is already strictly positive, no permutation
# can reach objective 0 in this subtree, so we prune.
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

# **Tree callback**: Early-stop logic.
#  - If the incumbent reaches 0 (within tolerance), we have found an isomorphism.
#  - If the global tree lower bound becomes strictly positive, we can certify non-isomorphism.
function build_tree_callback()
    return function (
        tree,
        node;
        worse_than_incumbent = false,
        node_infeasible = false,
        lb_update = false,
    )
        if isapprox(tree.incumbent, 0.0, atol = eps())
            tree.root.problem.solving_stage = Boscia.USER_STOP
            println("Optimal solution found.")
        end
        if Boscia.tree_lb(tree::Bonobo.BnBTree) > 0.0 + eps()
            tree.root.problem.solving_stage = Boscia.USER_STOP
            println("Tree lower bound already positive. No solution possible.")
        end
    end
end

# ## Heuristic configuration
# We pick k ~ sqrt(n) random neighbors per heuristic call.
k = Int(round(sqrt(n)))
swap_heu = Boscia.Heuristic(
    (tree, blmo, x) -> random_k_neighbor_matrix(tree, blmo, x, k, false),
    1.0,
    :swap,
)

# ## Solver settings
# Configure Boscia settings
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

# ## Solve the graph isomorphism problem
x, _, result = Boscia.solve(f, grad!, blmo, settings = settings)

# ## Certificate
# If the graphs are isomorphic, the optimizer should recover a permutation
# matrix X with objective value f(X) = 0, satisfying B = X * A * X'
# (equivalently, A ≈ X' * B * X).
X = reshape(x, n, n)
@assert A ≈ X' * B * X
println("Certificate verified: graphs are isomorphic (A ≈ X' * B * X)")

# If the graphs are not isomorphic, no perfect permutation exists,
# and the optimization problem yields a positive lower bound.
# @assert result[:dual_bound] > 0.0
# println("Graphs are not isomorphic (lower bound > 0)")
