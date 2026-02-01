using MathOptInterface
const MOI = MathOptInterface
using Test
using HiGHS
using Boscia
using FrankWolfe
using LinearAlgebra


@testset "LMO vs MOI on the same feasible region - Knorm" begin
    n = 10
    K = 5
    rhs = 3.0

    direction = randn(n)
    lb = fill(-7.0, n)
    ub = fill(7.0, n)
    int_vars = [2, 5, 7]

    # --- LMO ---
    lmo = FrankWolfe.KNormBallLMO(K, rhs)

    v_lmo = Boscia.bounded_compute_extreme_point(lmo, direction, lb, ub, int_vars)

    # --- MOI ---
    model_l1 = MOI.instantiate(HiGHS.Optimizer)
    model_inf = MOI.instantiate(HiGHS.Optimizer)

    v = [MOI.add_variable(model_inf) for _ in 1:n]
    v = [MOI.add_variable(model_l1) for _ in 1:n]
    for i in 1:n
        MOI.add_constraint(model_inf, v[i], MOI.GreaterThan(lb[i]))
        MOI.add_constraint(model_l1, v[i], MOI.GreaterThan(lb[i]))
        MOI.add_constraint(model_inf, v[i], MOI.LessThan(ub[i]))
        MOI.add_constraint(model_l1, v[i], MOI.LessThan(ub[i]))
        if i in int_vars
            MOI.add_constraint(model_inf, v[i], MOI.Integer())
            MOI.add_constraint(model_l1, v[i], MOI.Integer())
        end
    end

    # |v_i| linearization
    abs_v = [MOI.add_variable(model_inf) for _ in 1:n]
    abs_v = [MOI.add_variable(model_l1) for _ in 1:n]
    for i in 1:n
        MOI.add_constraint(model_inf, abs_v[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(model_l1, abs_v[i], MOI.GreaterThan(0.0))

        MOI.add_constraint(model_inf, abs_v[i], MOI.LessThan(rhs / K))

        MOI.add_constraint(
            model_inf,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -1.0], [abs_v[i], v[i]]), 0.0),
            MOI.GreaterThan(0.0),
        )
        MOI.add_constraint(
            model_l1,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -1.0], [abs_v[i], v[i]]), 0.0),
            MOI.GreaterThan(0.0),
        )

        MOI.add_constraint(
            model_inf,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], [abs_v[i], v[i]]), 0.0),
            MOI.GreaterThan(0.0),
        )
        MOI.add_constraint(
            model_l1,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], [abs_v[i], v[i]]), 0.0),
            MOI.GreaterThan(0.0),
        )
    end

    # ℓ1 constraint
    MOI.add_constraint(
        model_l1,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), abs_v), 0.0),
        MOI.LessThan(rhs),
    )

    # objective
    MOI.set(
        model_inf,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(direction, v), 0.0),
    )

    MOI.set(
        model_l1,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(direction, v), 0.0),
    )

    MOI.set(model_inf, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model_l1, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(model_l1)
    MOI.optimize!(model_inf)

    v_moi = [MOI.get(model_l1, MOI.VariablePrimal(), v[i]) for i in 1:n]
    v_inf = [MOI.get(model_inf, MOI.VariablePrimal(), v[i]) for i in 1:n]

    if dot(v_inf, direction) < dot(v_moi, direction)
        v_moi = v_inf
    end

    # compare objective values (not pointwise equality!)
    @test isapprox(v_lmo, v_moi; atol=1e-6)
end
