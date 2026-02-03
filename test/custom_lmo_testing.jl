using MathOptInterface
using Test
using HiGHS
using JuMP
using Boscia
using FrankWolfe


@testset "LMO vs MOI on the same feasible region - K" begin
    n = 10
    K = 5
    rhs = 3.0

    direction = randn(n)
    lb = fill(-7.0, n)
    ub = fill(7.0, n)
    int_vars = [2, 5, 7]

    # --- LMO ---
    lmo = FrankWolfe.KSparseLMO(K, rhs)

    v_lmo = Boscia.bounded_compute_extreme_point(lmo, direction, lb, ub, int_vars)

    # --- MOI ---
    model = Model(HiGHS.Optimizer)

    # variables v with bounds
    @variable(model, lb[i] <= v[i=1:n] <= ub[i])

    # integer variables
    for i in int_vars
        set_integer(v[i])
    end

    @variable(model, abs_v[i=1:n] >= 0)

    for i in 1:n
        @constraint(model, abs_v[i] <= rhs)
        @constraint(model, abs_v[i] >= v[i])
        @constraint(model, abs_v[i] >= -v[i])
    end

    @constraint(model, sum(abs_v[i] for i in 1:n) <= K * rhs)

    # objective
    @objective(model, Min, sum(direction[i] * v[i] for i in 1:n))

    optimize!(model)

    v_moi = value.(v)

    # compare objective values (not pointwise equality!)
    @test isapprox(v_lmo, v_moi; atol=1e-6)
end
