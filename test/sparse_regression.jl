using Statistics
using LinearAlgebra
using Random
using SCIP
using Boscia
import MathOptInterface
const MOI = MathOptInterface
using Test
using Bonobo
using FrankWolfe
using StableRNGs

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

# Constant parameters for the sparse regression
# min norm(y-A β)² + λ_0 ∑ z_i + λ_2 ||β||²
# s.t. -Mz_i<=β_i <= Mz_i
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p 

n0 = 10;
p = 5 * n0;
k = ceil(n0 / 5);
const lambda_0 = rand(rng, Float64);
const lambda_2 = 10.0 * rand(rng, Float64);
const A = rand(rng, Float64, n0, p)
const y = rand(rng, Float64, n0)
const M = 2 * var(A)

@testset "Sparse Regression" begin
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
        return sum((y - A * x[1:p]) .^ 2) + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(x[1:p])^2
    end
    function grad!(storage, x)
        storage .= vcat(
            2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p]),
            lambda_0 * ones(p),
        )
        return storage
    end

    x, _, result = Boscia.solve(f, grad!, lmo, settings_bnb=Boscia.settings_bnb(verbose=true, time_limit=100))
    # println("Solution: $(x[1:p])")
    @test sum(x[1+p:2p]) <= k
    @test f(x) <= f(result[:raw_solution]) + 1e-6
end


n0 = 10;
p = 5 * n0;
k = ceil(n0 / 5);
group_size = convert(Int64, floor(p / k));
const lambda_0_g = rand(rng, Float64);
const lambda_2_g = 10.0 * rand(rng, Float64);
const A_g = rand(rng, Float64, n0, p)
const y_g = rand(rng, Float64, n0)
const M_g = 2 * var(A_g)
const groups = []

k_int = convert(Int64, k)
for i in 1:(k_int-1)
    push!(groups, ((i-1)*group_size+p+1):(i*group_size+p))
end
push!(groups, ((k_int-1)*group_size+p+1):2p)

function build_sparse_lmo_grouped()
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
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, M_g], [x[i], x[i+p]]), 0.0),
            MOI.GreaterThan(0.0),
        )
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -M_g], [x[i], x[i+p]]), 0.0),
            MOI.LessThan(0.0),
        )
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p), x[p+1:2p]), 0.0),
        MOI.LessThan(k),
    )
    for i in 1:k_int
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(group_size), x[groups[i]]), 0.0),
            MOI.GreaterThan(1.0),
        )
    end
    lmo = FrankWolfe.MathOptLMO(o)
    return lmo
end

@testset "Sparse Regression Group" begin
    function f(x)
        return sum(abs2, y_g - A_g * x[1:p]) +
               lambda_0_g * norm(x[p+1:2p])^2 +
               lambda_2_g * norm(x[1:p])^2
    end
    function grad!(storage, x)
        storage .= vcat(
            2 * (transpose(A_g) * A_g * x[1:p] - transpose(A_g) * y_g + lambda_2_g * x[1:p]),
            2 * lambda_0_g * x[p+1:2p],
        )
        return storage
    end

    lmo = build_sparse_lmo_grouped()
    x, _, result = Boscia.solve(f, grad!, lmo, 
        settings_bnb=Boscia.settings_bnb(verbose=true),
        settings_tolerances=Boscia.settings_tolerances(fw_epsilon=1e-3))

    @test sum(x[p+1:2p]) <= k
    for i in 1:k_int
        @test sum(x[groups[i]]) >= 1
    end
    println("Non zero entries:")
    for i in 1:p
        if x[i+p] == 1
            println("$(i)th entry: $(x[i])")
        end
    end

    # strong convexity
    μ = 2lambda_0_g

    lmo = build_sparse_lmo_grouped()
    x2, _, result2 = Boscia.solve(f, grad!, lmo, 
        settings_bnb=Boscia.settings_bnb(verbose=true),
        settings_tolerances=Boscia.settings_tolerances(fw_epsilon=1e-3),
        settings_tightening=Boscia.settings_tightening(strong_convexity=μ))
    @test sum(x2[p+1:2p]) <= k
    for i in 1:k_int
        @test sum(x2[groups[i]]) >= 1
    end
    @test f(x) ≈ f(x2)
end
