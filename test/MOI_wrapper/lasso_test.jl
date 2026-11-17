using Boscia
using JuMP
using SCIP
using StableRNGs
using Test

# Sparse regression with group constraints (examples/lasso.jl)
#   min ||y - Aβ||² + λ_0 ∑ z_i + λ_2 ||β||²
#   s.t. -M z_i <= β_i <= M z_i
#        z_i = 1 => -M <= β_i <= M
#        ∑ z_i <= k
#        ∑_{i ∈ G_j} z_i >= 1 for every group G_j
#        z_i ∈ {0,1}

@testset "Sparse regression with group constraints" begin
    rng = StableRNG(111)

    n = 8
    p = 2 * n
    k_int = 4                     # number of groups
    group_size = p ÷ k_int
    M_g = 5.0

    lambda_0_g = 0.0
    lambda_2_g = 0.0
    A_g = rand(rng, Float64, n, p)
    β_sol = rand(rng, p) .* (2 * M_g) .- M_g

    for i in 1:k_int
        for _ in 1:(group_size-1)
            β_sol[rand(rng, ((i-1)*group_size+1):(i*group_size))] = 0
        end
    end
    y_g = A_g * β_sol
    k = count(!iszero, β_sol)

    groups = UnitRange{Int}[]
    for i in 1:(k_int-1)
        push!(groups, ((i-1)*group_size+1):(i*group_size))
    end
    push!(groups, ((k_int-1)*group_size+1):p)

    f(x) =
        sum((y_g - A_g * x[1:p]) .^ 2) +
        lambda_0_g * sum(x[(p+1):2p]) +
        lambda_2_g * sum(abs2, x[1:p])
    function grad!(storage, x)
        storage .=
            vcat(2 * (A_g' * A_g * x[1:p] - A_g' * y_g + lambda_2_g * x[1:p]), lambda_0_g * ones(p))
        return storage
    end

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.tolerances[:rel_dual_gap] = 1e-2
    settings.tolerances[:dual_gap] = 1e-5

    # Original Boscia
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, p)
    z = MOI.add_variables(o, p)
    for i in 1:p
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne()) # or MOI.Integer()
    end
    for i in 1:p
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, M_g], [x[i], z[i]]), 0.0),
            MOI.GreaterThan(0.0),
        )
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -M_g], [x[i], z[i]]), 0.0),
            MOI.LessThan(0.0),
        )
        # Indicator: x[i+p] = 1 => -M_g <= x[i] <= M_g
        gl = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[i])),
            ],
            [0.0, 0.0],
        )
        gg = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, x[i])),
            ],
            [0.0, 0.0],
        )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(M_g)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(M_g)))
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p), z), 0.0),
        MOI.LessThan(1.0 * k),
    )
    for i in 1:k_int
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(group_size), z[groups[i]]), 0.0),
            MOI.GreaterThan(1.0),
        )
    end
    blmo = Boscia.MathOptBLMO(o)

    x_ref, _, result = Boscia.solve(f, grad!, blmo, settings=settings)

    # JuMP
    model = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model)

    @variable(model, β[1:p])
    @variable(model, z[1:p], Bin)

    @constraint(model, [i = 1:p], β[i] + M_g * z[i] >= 0)
    @constraint(model, [i = 1:p], β[i] - M_g * z[i] <= 0)
    @constraint(model, [i = 1:p], z[i] --> {β[i] <= M_g})
    @constraint(model, [i = 1:p], z[i] --> {β[i] >= -M_g})
    @constraint(model, sum(z) <= k)
    @constraint(model, [j = 1:length(groups)], sum(z[groups[j]]) >= 1)

    @objective(model, Min, sum((y_g[r] - sum(A_g[r, i] * β[i] for i in 1:p))^2 for r in 1:n))
    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL

    z_val = value.(z)
    @test sum(z_val) <= k + 1e-6
    for j in 1:length(groups)
        @test sum(z_val[groups[j]]) >= 1 - 1e-6
    end
    @test objective_value(model) ≈ f(x_ref) rtol = 1e-2 atol = 1e-6
end
