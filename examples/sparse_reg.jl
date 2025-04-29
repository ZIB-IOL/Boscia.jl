using Boscia
using FrankWolfe
using Random
using LinearAlgebra
using SCIP
import Bonobo
import MathOptInterface
const MOI = MathOptInterface
using Dates
using Printf
using Test
using StableRNGs

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

# Sparse regression

# Constant parameters for the sparse regression
# min norm(y-A β)² + λ_0 ∑ z_i + λ_2 ||β||²
# s.t. -Mz_i <= β_i <= Mz_i
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p 

# A - matrix of observations.
# y - vector of results.
# We want to match Aβ as closely as possible to y 
# while having relative few non zero entries in β.
# Each continuous variable β_i is assigned a binary z_i,
# z_i = 0 => β_i = 0


n0 = 20;
p = 5 * n0;
k = ceil(n0 / 5);
const lambda_0 = rand(rng, Float64);
const lambda_2 = 10.0 * rand(rng, Float64);
const A = rand(rng, Float64, n0, p)
const y = rand(rng, Float64, n0)
const M = 2 * var(A)

# "Sparse Regression" 
@testset "Sparse regression" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, 2p)
    for i in p+1:2p
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, x[i], MOI.ZeroOne()) # or MOI.Integer()
    end
    for i in 1:p
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, M], [x[i], x[i+p]]), 0.0),
            MOI.GreaterThan(0.0),
        )
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -M], [x[i], x[i+p]]), 0.0),
            MOI.LessThan(0.0),
        )
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p), x[p+1:2p]), 0.0),
        MOI.LessThan(k),
    )
    lmo = FrankWolfe.MathOptLMO(o)

    function f(x)
        xv = @view(x[1:p])
        return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
    end

    function grad!(storage, x)
        storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
        storage[p+1:2p] .= lambda_0
        return storage
    end

    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, fw_epsilon=1e-3, print_iter=10)

    # @show result // too large to be output
    @test f(x) <= f(result[:raw_solution]) + 1e-6
end
