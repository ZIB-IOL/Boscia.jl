using JuMP
using Boscia
using SCIP

# --------------------------------------------------
# Problem:
#
#   min (x - 2)^2
#
#   s.t.
#       0 <= x <= 5
#       x integer
#
# Expected:
#   x = 2
#   objective = 0
# --------------------------------------------------

model = Model(
    () -> Boscia.Optimizer(SCIP.Optimizer)
)

@variable(
    model,
    0 <= x <= 5,
    Int,
)

@objective(
    model,
    Min,
    (x - 2)^2,
)

println("----- Calling JuMP optimize! -----")

optimize!(model)

println("\n----- Results -----")
println("Termination status: ", termination_status(model))
println("Primal status:      ", primal_status(model))
println("x =                  ", value(x))
println("Objective value =    ", objective_value(model))

println("\nExpected:")
println("x ≈ 2.0")
println("objective ≈ 0.0")