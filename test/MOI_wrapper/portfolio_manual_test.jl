using Boscia
using JuMP
using SCIP
using Test
using Random
using LinearAlgebra
using StableRNGs

println("\nPortfolio MOI Wrapper Example")

# --------------------------------------------------
# Generate the same kind of portfolio data
# --------------------------------------------------

seed = 128

rng = StableRNG(seed)

n = 30

ri = rand(rng, n)
ai = rand(rng, n)
Ωi = rand(rng, Float64)

bi = sum(ai)

Ai = randn(rng, n, n)
Ai = Ai' * Ai

Mi = (Ai + Ai') / 2

@assert isposdef(Mi)

# --------------------------------------------------
# Build model through JuMP -> Boscia.Optimizer
# --------------------------------------------------

model = Model(
    () -> Boscia.Optimizer(SCIP.Optimizer)
)

# All variables are nonnegative integers,
# matching the original example.
@variable(
    model,
    x[1:n] >= 0,
    Int,
)

# --------------------------------------------------
# Constraints
#
# sum(ai[i] * x[i]) <= bi
# sum(x[i]) >= 1
# --------------------------------------------------

@constraint(
    model,
    sum(ai[i] * x[i] for i in 1:n) <= bi,
)

@constraint(
    model,
    sum(x[i] for i in 1:n) >= 1.0,
)

# --------------------------------------------------
# Quadratic objective
#
# f(x) =
#   1/2 * Ωi * x' * Mi * x
#   - ri' * x
# --------------------------------------------------

@objective(
    model,
    Min,
    0.5 * Ωi *
    sum(
        Mi[i, j] * x[i] * x[j]
        for i in 1:n, j in 1:n
    )
    -
    sum(
        ri[i] * x[i]
        for i in 1:n
    ),
)

# --------------------------------------------------
# MOI attributes
# --------------------------------------------------

set_time_limit_sec(
    model,
    120.0,
)

println("\n----- Calling JuMP optimize! -----\n")

optimize!(model)

# --------------------------------------------------
# Results
# --------------------------------------------------

status = termination_status(model)
pstatus = primal_status(model)

println("\n----- Results -----")
println("Termination status: ", status)
println("Primal status:      ", pstatus)

if has_values(model)
    x_value = value.(x)
    obj_value = objective_value(model)

    println("Objective value:    ", obj_value)

    println(
        "ai' * x =            ",
        dot(ai, x_value),
    )

    println(
        "sum(x) =             ",
        sum(x_value),
    )

    println(
        "All integer:         ",
        all(
            isapprox(v, round(v); atol = 1e-6)
            for v in x_value
        ),
    )

    # --------------------------------------------------
    # Check feasibility
    # --------------------------------------------------

    @test dot(ai, x_value) <= bi + 1e-6
    @test sum(x_value) >= 1.0 - 1e-6

    @test all(
        x_value .>= -1e-6
    )

    @test all(
        isapprox(v, round(v); atol = 1e-6)
        for v in x_value
    )

    # --------------------------------------------------
    # Independently recompute objective
    # --------------------------------------------------

    expected_objective =
        0.5 * Ωi * dot(x_value, Mi, x_value) -
        dot(ri, x_value)

    println(
        "Recomputed objective: ",
        expected_objective,
    )

    @test isapprox(
        obj_value,
        expected_objective;
        atol = 1e-5,
        rtol = 1e-5,
    )
end