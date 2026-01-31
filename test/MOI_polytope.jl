using MathOptInterface
const MOI = MathOptInterface
using Test
using HiGHS
using Boscia
using FrankWolfe


@testset "LMO vs MOI on the same feasible region" begin
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
    model = MOI.instantiate(HiGHS.Optimizer)

    v = [MOI.add_variable(model) for _ in 1:n]
    for i in 1:n
        MOI.add_constraint(model, v[i], MOI.GreaterThan(lb[i]))
        MOI.add_constraint(model, v[i], MOI.LessThan(ub[i]))
        if i in int_vars
            MOI.add_constraint(model, v[i], MOI.Integer())
        end
    end

    # |v_i| linearization
    abs_v = [MOI.add_variable(model) for _ in 1:n]
    for i in 1:n
        MOI.add_constraint(model, abs_v[i], MOI.GreaterThan(0.0))

        MOI.add_constraint(model, abs_v[i], MOI.LessThan(rhs))

        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -1.0], [abs_v[i], v[i]]), 0.0),
            MOI.GreaterThan(0.0),
        )

        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], [abs_v[i], v[i]]), 0.0),
            MOI.GreaterThan(0.0),
        )
    end

    # ℓ1 constraint
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), abs_v), 0.0),
        MOI.LessThan(K * rhs),
    )

    # objective
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(direction, v), 0.0),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(model)

    v_moi = [MOI.get(model, MOI.VariablePrimal(), v[i]) for i in 1:n]

    # compare objective values (not pointwise equality!)
    @test isapprox(v_lmo, v_moi; atol=1e-6)
end
