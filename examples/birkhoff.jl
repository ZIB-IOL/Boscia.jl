using BranchWolfe
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface

# Example on the Birkhoff polytope but using permutation matrices directly
# https://arxiv.org/pdf/2011.02752.pdf
# https://www.sciencedirect.com/science/article/pii/S0024379516001257


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

n = 10
k = 4

# generate random doubly stochastic matrix
const Xstar = rand(n, n)
while norm(sum(Xstar, dims=1) .- 1) > 1e-6 || norm(sum(Xstar, dims=2) .- 1) > 1e-6
    Xstar ./= sum(Xstar, dims=1)
    Xstar ./= sum(Xstar, dims=2)
end


function f(x)
    s = zero(eltype(x))
    for i in eachindex(Xstar)
        s += 0.5 * (sum(x[(j-1) * n^2 + i] for j in 1:k) - Xstar[i])^2
    end
    return s
end

# note: reshape gives a reference to the same data, so this is updating storage in-place
function grad!(storage, x)
    storage .= 0
    for j in 1:k
        Sk = reshape(@view(storage[(j-1) * n^2 + 1 : j * n^2]), n, n)
        @. Sk = - Xstar
        for m in 1:k
            Yk = reshape(@view(x[(m-1) * n^2 + 1 : m * n^2]), n, n)
            @. Sk += Yk
        end
    end
    storage
end

@testset "Interface - norm hyperbox" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    Y = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
    X = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
    theta = MOI.add_variables(o, k)

    # TODO FINISH
    for i in 1:k
        MOI.add_constraint.(o, Y[i], MOI.ZeroOne())
        MOI.add_constraint.(o, X[i], MOI.ZeroOne())
        MOI.add_constraint(o, theta[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, theta[i], MOI.LessThan(1.0))
    end
    lmo = FrankWolfe.MathOptLMO(o)

    x, _ = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = true)

    # build optimal solution
    xopt = zeros(n)
    for i in 1:n
        if diffi[i] > 0.5
            xopt[i] = 1
        end
    end

    @test f(x) == f(xopt)
    println("\nNumber of processed nodes should be: ", 2^(n+1)-1)
    println()
end
