using Boscia
using JuMP
using SCIP
using StableRNGs
using Test

# min 1/2 ∑ (x_i - d_i)^2  s.t. x ∈ {0,1}^n

rng = StableRNG(111)
n = 20
diffi = rand(rng, Bool, n) * 0.6 .+ 0.3

@testset "Approximate planted point - Integer" begin

    f(x) = 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    function grad!(storage, x)
        @. storage = x - diffi
    end

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false

    # Original Boscia, SCIP
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    xs = MOI.add_variables(o, n)
    for xi in xs
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
        MOI.add_constraint(o, xi, MOI.ZeroOne())
    end
    blmo = Boscia.MathOptBLMO(o)
    x_ref, _, _ = Boscia.solve(f, grad!, blmo; settings=settings)

    # JuMP, Bin
    model = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model)

    @variable(model, x[1:n], Bin)

    @objective(model, Min, 0.5 * sum((x[i] - diffi[i])^2 for i in 1:n))
    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test value.(x) ≈ x_ref atol = 1e-6
    @test objective_value(model) ≈ f(x_ref) atol = 1e-6

    # Original Boscia, ZeroOneHypercubeLMO
    x_ref2, _, _ = Boscia.solve(
        f,
        grad!,
        Boscia.ZeroOneHypercubeLMO(),
        zeros(n),
        ones(n),
        collect(1:n),
        n;
        settings=settings,
    )

    # JuMP, ZeroOneHypercube
    model2 = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model2)

    @variable(model2, x2[1:n], Int)

    @constraint(model2, x2 in Boscia.ZeroOneHypercube(n))

    @objective(model2, Min, 0.5 * sum((x2[i] - diffi[i])^2 for i in 1:n))
    optimize!(model2)

    @test termination_status(model2) == MOI.OPTIMAL
    @test value.(x2) ≈ x_ref2 atol = 1e-6
    @test objective_value(model2) ≈ f(x_ref2) atol = 1e-6

    # Both should be the planted point
    @test x_ref ≈ round.(diffi) atol = 1e-6
    @test x_ref2 ≈ round.(diffi) atol = 1e-6
end

# min 1/2 ∑ (x_i - d_i)^2
# s.t. x ∈ [0,1]^n, x_i ∈ {0,1} for i ∈ int_vars

@testset "Approximate planted point - Mixed" begin
    f(x) = 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    function grad!(storage, x)
        @. storage = x - diffi
    end

    int_vars = sort!(unique!(rand(rng, 1:n, n ÷ 2)))
    sol = copy(diffi)
    sol[int_vars] = round.(sol[int_vars])

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false

    # Original Boscia, SCIP
    o3 = SCIP.Optimizer()
    MOI.set(o3, MOI.Silent(), true)
    xs3 = MOI.add_variables(o3, n)
    for xi in xs3
        MOI.add_constraint(o3, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o3, xi, MOI.LessThan(1.0))
        if xi.value in int_vars
            MOI.add_constraint(o3, xi, MOI.ZeroOne())
        end
    end
    blmo3 = Boscia.MathOptBLMO(o3)
    x_ref3, _, _ = Boscia.solve(f, grad!, blmo3; settings=settings)

    # JuMP, bounds + integrality
    model3 = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model3)

    @variable(model3, 0 <= x3[1:n] <= 1)
    for i in int_vars
        set_integer(x3[i])
    end

    @objective(model3, Min, 0.5 * sum((x3[i] - diffi[i])^2 for i in 1:n))
    optimize!(model3)

    @test termination_status(model3) == MOI.OPTIMAL
    @test value.(x3) ≈ x_ref3 atol = 1e-6
    @test objective_value(model3) ≈ f(x_ref3) atol = 1e-6

    # Original Boscia, BoxLMO
    x_ref4, _, _ = Boscia.solve(
        f,
        grad!,
        Boscia.BoxLMO(zeros(n), ones(n)),
        zeros(length(int_vars)),
        ones(length(int_vars)),
        int_vars,
        n;
        settings=settings,
    )

    # JuMP, ZeroOneHypercube
    model4 = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model4)

    @variable(model4, x4[1:n])
    for i in int_vars
        set_integer(x4[i])
    end

    @constraint(model4, x4 in Boscia.ZeroOneHypercube(n))

    @objective(model4, Min, 0.5 * sum((x4[i] - diffi[i])^2 for i in 1:n))
    optimize!(model4)

    @test termination_status(model4) == MOI.OPTIMAL
    @test value.(x4) ≈ x_ref4 atol = 1e-6
    @test objective_value(model4) ≈ f(x_ref4) atol = 1e-6

    # Both should be the planted point
    @test x_ref3 ≈ sol atol = 1e-6
    @test x_ref4 ≈ sol atol = 1e-6
end
