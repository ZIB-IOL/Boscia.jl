using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
import HiGHS

# Example on the Birkhoff polytope but using permutation matrices directly
# https://arxiv.org/pdf/2011.02752.pdf
# https://www.sciencedirect.com/science/article/pii/S0024379516001257

# For bug hunting:
seed = rand(UInt64)
@show seed
Random.seed!(seed)


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
# Small dimensions since the size of the problem grows quickly (2 k n^2 + k variables)
n = 3
k = 2

# generate random doubly stochastic matrix
const Xstar = rand(n, n)
while norm(sum(Xstar, dims=1) .- 1) > 1e-6 || norm(sum(Xstar, dims=2) .- 1) > 1e-6
    Xstar ./= sum(Xstar, dims=1)
    Xstar ./= sum(Xstar, dims=2)
end

function f(x)
    s = zero(eltype(x))
    for i in eachindex(Xstar)
        s += 0.5 * (sum(x[(j-1)*n^2+i] for j in 1:k) - Xstar[i])^2
    end
    return s
end

# note: reshape gives a reference to the same data, so this is updating storage in-place
function grad!(storage, x)
    storage .= 0
    for j in 1:k
        Sk = reshape(@view(storage[(j-1)*n^2+1:j*n^2]), n, n)
        @. Sk = -Xstar
        for m in 1:k
            Yk = reshape(@view(x[(m-1)*n^2+1:m*n^2]), n, n)
            @. Sk += Yk
        end
    end
    return storage
end

function build_birkhoff_lmo()
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    Y = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
    X = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
    theta = MOI.add_variables(o, k)

    for i in 1:k
        MOI.add_constraint.(o, Y[i], MOI.GreaterThan(0.0))
        MOI.add_constraint.(o, Y[i], MOI.LessThan(1.0))
        MOI.add_constraint.(o, X[i], MOI.ZeroOne())
        MOI.add_constraint(o, theta[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, theta[i], MOI.LessThan(1.0))
        # doubly stochastic constraints
        MOI.add_constraint.(
            o,
            vec(sum(X[i], dims=1, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
            MOI.EqualTo(1.0),
        )
        MOI.add_constraint.(
            o,
            vec(sum(X[i], dims=2, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
            MOI.EqualTo(1.0),
        )
        # 0 ≤ Y_i ≤ X_i
        MOI.add_constraint.(o, 1.0 * Y[i] - X[i], MOI.LessThan(0.0))
        # 0 ≤ θ_i - Y_i ≤ 1 - X_i
        MOI.add_constraint.(o, 1.0 * theta[i] .- Y[i] .+ X[i], MOI.LessThan(1.0))
    end
    MOI.add_constraint(o, sum(theta, init=0.0), MOI.EqualTo(1.0))
    return FrankWolfe.MathOptLMO(o)
end

lmo = build_birkhoff_lmo()
x, _, _ = Boscia.solve(f, grad!, lmo, verbose=true)


# TODO the below needs to be fixed
# TODO can use the min_via_enum function if not too many solutions
# build optimal solution
# xopt = zeros(n)
# for i in 1:n
#     if diffi[i] > 0.5
#         xopt[i] = 1
#     end
# end

@testset "Birkhoff Decomposition" begin
    lmo = build_birkhoff_lmo()
    x, _, result_baseline = Boscia.solve(f, grad!, lmo, verbose=true)
    @test f(x) <= f(result_baseline[:raw_solution]) + 1e-6
    lmo = build_birkhoff_lmo()
    blmo = Boscia.MathOptBLMO(HiGHS.Optimizer())
    branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
    MOI.set(branching_strategy.bounded_lmo.o, MOI.Silent(), true)
    x_strong, _, result_strong =
        Boscia.solve(f, grad!, lmo, verbose=true, branching_strategy=branching_strategy)
    @test isapprox(f(x), f(x_strong), atol=1e-4, rtol=1e-2)
    @test f(x) <= f(result_strong[:raw_solution]) + 1e-6
end
