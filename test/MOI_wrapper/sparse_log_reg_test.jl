using Boscia
using JuMP
using LinearAlgebra
using SCIP
using StableRNGs
using Test

# Sparse logistic regression
#   min ∑_j log(1 + exp(-y_j X_j^T β)) + λ ||β||^2
#   s.t. -M z_i <= β_i <= M z_i
#        ∑ z_i <= k
#        z_i ∈ {0,1}

@testset "Sparse logistic regression" begin
    rng = StableRNG(111)

    m = 20
    n = 10
    M = 5.0
    k = 5
    λ = 0.1

    X = randn(rng, m, n)
    β_true = zeros(n)
    β_true[1:k] = randn(rng, k)
    y = sign.(X * β_true)

    f(x) = sum(log(1 + exp(-y[j] * dot(X[j, :], x[1:n]))) for j in 1:m) + λ * sum(abs2, x[1:n])
    function grad!(storage, x)
        β = x[1:n]
        storage .= 0
        for j in 1:m
            a = dot(X[j, :], β)
            storage[1:n] .+= -y[j] / (1 + exp(y[j] * a)) .* X[j, :]
        end
        storage[1:n] .+= 2λ .* β
        return storage
    end

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false

    # Boscia
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    βs = MOI.add_variables(o, n)
    zs = MOI.add_variables(o, n)
    for i in 1:n
        MOI.add_constraint(o, zs[i], MOI.ZeroOne())
        MOI.add_constraint(o, 1.0 * βs[i] + M * zs[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, 1.0 * βs[i] - M * zs[i], MOI.LessThan(0.0))
    end
    MOI.add_constraint(o, sum(zs, init=0.0), MOI.LessThan(1.0 * k))
    blmo = Boscia.MathOptBLMO(o)
    x_ref, _, _ = Boscia.solve(f, grad!, blmo; settings=settings)

    # JuMP
    model = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model)

    @variable(model, β[1:n])
    @variable(model, z[1:n], Bin)

    @constraint(model, [i = 1:n], β[i] + M * z[i] >= 0)
    @constraint(model, [i = 1:n], β[i] - M * z[i] <= 0)
    @constraint(model, sum(z) <= k)

    @expression(model, a[j=1:m], sum(X[j, i] * β[i] for i in 1:n))
    @objective(
        model,
        Min,
        sum(log(1 + exp(-y[j] * a[j])) for j in 1:m) + λ * sum(β[i]^2 for i in 1:n),
    )
    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test sum(value.(z)) <= k + 1e-6
    @test objective_value(model) ≈ f(x_ref) rtol = 1e-2 atol = 1e-6
end
