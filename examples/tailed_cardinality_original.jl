using Statistics
using Boscia
using FrankWolfe
using Random
using LinearAlgebra
using SCIP
import MathOptInterface
const MOI = MathOptInterface
using Dates
using Printf
using Test

# Two-tailed cardinality-constrained regression

# Constant parameters for the sparse regression
# min_{x,s,z} ∑_i f_i(x) - λ ∑_j z_j + μ ||x||²
# s.t. z_j = 1 => s ≤ 0
#      s ≥ x_j  - τ_j
#      s ≥ -x_j - τ_j
#      x, s ∈ X, 
#      z_j ∈ {0,1}^n

# f_i - contributions to the loss function.
# x - predictors.
# λ - cardinality penalty
# μ - ℓ₂ penalty

# modified from
# Tractable Continuous Approximations for Constraint Selection via Cardinality Minimization, Ahn, Gangammanavar†1, Troxell
Random.seed!(42)

const m0 = 1000;
const n0 = 100
const λ = rand()
const μ = 10.0 * rand()
const A = rand(m0, n0)
const y = rand(m0)
const τ = 6 * rand(n0)
const M = 20.0

function build_twotailed_optimizer(τ, M)
    n = length(τ)
    o = SCIP.Optimizer()
    x = MOI.add_variables(o, n)
    z = MOI.add_variables(o, n)
    s = MOI.add_variables(o, n)
    MOI.add_constraint.(o, z, MOI.ZeroOne())
    MOI.add_constraint.(o, s, MOI.GreaterThan(0.0))
    MOI.add_constraint.(o, x, MOI.GreaterThan(-M))
    MOI.add_constraint.(o, x, MOI.LessThan(M))
    MOI.add_constraint.(o, s, MOI.LessThan(M))
    for j in 1:n
        MOI.add_constraint(o, 1.0 * s[j] - x[j], MOI.GreaterThan(-τ[j]))
        MOI.add_constraint(o, 1.0 * s[j] + x[j], MOI.GreaterThan(-τ[j]))
        MOI.add_constraint(o,
            MOI.VectorAffineFunction(
                [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[j])), MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, s[j]))],
                [0.0, 0.0],
            ),
            MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(0.0)),
        )
    end
    return (o, x, z, s)
end

function build_objective_gradient(λ, μ, n, A, y)
    function f(x)
        xv = @view(x[1:n])
        zv = @view(x[n+1:2n])
        return norm(y - A * xv)^2 - λ * sum(zv) + μ * norm(xv)^2
    end

    function grad!(storage, x)
        xv = @view(x[1:n])
        storage .= 0
        @view(storage[1:n]) .= 2 * (transpose(A) * A * xv - transpose(A) * y + μ * xv)
        @view(storage[n+1:2n]) .= -λ
        return storage
    end
    return (f, grad!)
end

# "Sparse Regression" 
const f, grad! = build_objective_gradient(λ, μ, n0, A, y)
const (o, x, z, s) = build_twotailed_optimizer(τ, M)
const lmo = FrankWolfe.MathOptLMO(o)

gradient = randn(3n0)
x0 = FrankWolfe.compute_extreme_point(lmo, gradient)
MOI.set(o, MOI.Silent(), true)

Boscia.solve(f, grad!, lmo, verbose=true, fw_epsilon=1e-2, print_iter=10, time_limit=300)
