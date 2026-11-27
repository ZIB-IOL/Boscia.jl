using Boscia
using JuMP
using SCIP
using StableRNGs
using Test
using LinearAlgebra
using Distributions

# Poisson sparse regression

# min_{w, b, z} ∑_i exp(w x_i + b) - y_i (w x_i + b) + α norm(w)^2
# s.t. -N z_i <= w_i <= N z_i
# b ∈ [-N, N]
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p

# y_i    - data points, poisson distributed 
# X_i, b - coefficient for the linear estimation of the expected value of y_i
# w_i    - continuous variables
# z_i    - binary variables s.t. z_i = 0 => w_i = 0
# k      - max number of non zero entries in w

# In a poisson regression, we want to model count data.
# It is assumed that y_i is poisson distributed and that the log 
# of its expected value can be computed linearly.

@testset "Sparse poisson regression" begin
    rng = StableRNG(111)

    n = 20
    p = n

    ws = rand(rng, Float64, p)
    for _ in 1:20
        ws[rand(rng, 1:p)] = 0
    end
    bs = rand(rng, Float64)
    Xs = randn(rng, Float64, n, p)
    ys = map(1:n) do idx
        a = dot(Xs[idx, :], ws) + bs
        return rand(rng, Distributions.Poisson(exp(a)))
    end
    Ns = 0.10

    # original Boscia
    k = 10
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    w = MOI.add_variables(o, p)
    z = MOI.add_variables(o, p)
    b = MOI.add_variable(o)
    for i in 1:p
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())
    end
    for i in 1:p
        MOI.add_constraint(o, Ns * z[i] + w[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, -Ns * z[i] + w[i], MOI.LessThan(0.0))
    end
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(o, sum(z, init=0.0), MOI.GreaterThan(1.0))
    MOI.add_constraint(o, b, MOI.LessThan(Ns))
    MOI.add_constraint(o, b, MOI.GreaterThan(-Ns))
    blmo = Boscia.MathOptBLMO(o)

    α = 1.3
    function f(θ)
        w = @view(θ[1:p])
        b = θ[end]
        s = sum(1:n) do i
            a = dot(w, Xs[:, i]) + b
            return 1 / n * (exp(a) - ys[i] * a)
        end
        return s + α * norm(w)^2
    end
    function grad!(storage, θ)
        w = @view(θ[1:p])
        b = θ[end]
        storage[1:p] .= 2α .* w
        storage[(p+1):2p] .= 0
        storage[end] = 0
        for i in 1:n
            xi = @view(Xs[:, i])
            a = dot(w, xi) + b
            storage[1:p] .+= 1 / n * xi * exp(a)
            storage[1:p] .-= 1 / n * ys[i] * xi
            storage[end] += 1 / n * (exp(a) - ys[i])
        end
        return storage
    end

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    x_ref, _, result = Boscia.solve(f, grad!, blmo, settings=settings)



    # JuMP 
    model = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model)

    @variable(model, w[1:p])
    @variable(model, z[1:p], Bin)
    @variable(model, b)

    @constraint(model, [i=1:p], -Ns * z[i] <= w[i])
    @constraint(model, [i=1:p], w[i] <= Ns * z[i])
    @constraint(model, 1 <= sum(z))
    @constraint(model, sum(z) <= k)
    @constraint(model, -Ns <= b)
    @constraint(model, b <= Ns)

    @expression(model, a[i=1:n], sum(w[r] * Xs[r, i] for r in 1:p) + b)
    @objective(
        model,
        Min,
        1 / n * sum(exp(a[i]) - ys[i] * a[i] for i in 1:n) + α * sum(w[j]^2 for j in 1:p),
    )

    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL

    z_val = value.(z)
    @test sum(z_val) <= k + 1e-6
    @test sum(z_val) >= 1 - 1e-6
    @test objective_value(model) ≈ f(x_ref) rtol = 1e-2 atol = 1e-6


end
