using Boscia
using JuMP
using Test
using SCIP
using StableRNGs

# Integer sparse regression
#
# original formulation in example:  min 1/2 || y - Dx|| ^ 2, s.t. 0 <= x_i <= l * z_i, ∑z_i <= k, x int, z Bin
# UnitSimplex formulation:  min 1/2 || y - Dx|| ^ 2, s.t. 0 <= x_i , ∑x_i <= k, x int

@testset "Integer sparse regression" begin
    rng = StableRNG(111)
    n = 10
    m = 30
    l = 5
    k = 4

    sol_x = rand(rng, 1:l, n)
    for _ in 1:(n-k)
        sol_x[rand(rng, 1:n)] = 0
    end

    D = rand(rng, m, n)
    y_d = D * sol_x

    # Original Boscia 
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    xs = MOI.add_variables(o, n)
    zs = MOI.add_variables(o, n)
    for i in 1:n
        MOI.add_constraint(o, xs[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xs[i], MOI.LessThan(1.0 * l))
        MOI.add_constraint(o, xs[i], MOI.Integer())
        MOI.add_constraint(o, zs[i], MOI.ZeroOne())
        MOI.add_constraint(o, 1.0 * xs[i] - 1.0 * l * zs[i], MOI.LessThan(0.0))
    end
    MOI.add_constraint(o, sum(zs, init=0.0), MOI.LessThan(1.0 * k))
    blmo = Boscia.MathOptBLMO(o)

    f(v) = 0.5 * sum(abs2, y_d - D * v[1:n])
    function grad!(storage, v)
        storage .= 0
        storage[1:n] .= D' * (D * v[1:n] - y_d)
        return storage
    end
    x_ref, _, _ = Boscia.solve(f, grad!, blmo; settings=Boscia.create_default_settings())

    # JuMP formulation of the example
    model = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model)

    @variable(model, 0 <= x[1:n] <= l, Int)
    @variable(model, z[i=1:n], Bin)

    @constraint(model, [i=1:n], x[i] <= l * z[i])
    @constraint(model, sum(z) <= k)

    @objective(model, Min, 0.5 * sum((y_d[j] - sum(D[j, i] * x[i] for i in 1:n))^2 for j in 1:m))

    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test value.(x) ≈ x_ref[1:n] atol = 1e-6
    @test objective_value(model) ≈ f(x_ref) atol = 1e-6

    # Original Boscia, UnitSimplex
    f2(v) = 0.5 * sum(abs2, y_d - D * v)
    function grad2!(storage, v)
        storage .= D' * (D * v - y_d)
        return storage
    end
    x_ref2, _, _ = Boscia.solve(
        f2,
        grad2!,
        Boscia.UnitSimplexLMO(1.0 * k),
        fill(0.0, n),
        fill(1.0 * k, n),
        collect(1:n),
        n,
    )

    # JuMP, UnitSimplex
    model2 = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model2)

    @variable(model2, x2[i=1:n], Int)

    @constraint(model2, x2 in Boscia.UnitSimplex(n, k))

    @objective(model2, Min, 0.5 * sum((y_d[j] - sum(D[j, i] * x2[i] for i in 1:n))^2 for j in 1:m))

    optimize!(model2)

    @test termination_status(model2) == MOI.OPTIMAL
    @test value.(x2) ≈ x_ref2 atol = 1e-6
    @test objective_value(model2) ≈ f2(x_ref2) atol = 1e-6

end
