using Test
using Boscia
using SCIP
import MathOptInterface as MOI

println("=== Boscia objective oracle smoke test ===")

# ------------------------------------------------------------
# Quadratic objective:
# f(x) = (x - 2)^2
#      = x^2 - 4x + 4
#
# gradient:
# f'(x) = 2x - 4
# ------------------------------------------------------------

opt = Boscia.Optimizer(SCIP.Optimizer)

x = MOI.add_variable(opt)

quadratic_obj = MOI.ScalarQuadraticFunction(
    [
        MOI.ScalarQuadraticTerm(2.0, x, x),
    ],
    [
        MOI.ScalarAffineTerm(-4.0, x),
    ],
    4.0,
)

MOI.set(
    opt,
    MOI.ObjectiveFunction{
        MOI.ScalarQuadraticFunction{Float64}
    }(),
    quadratic_obj,
)

MOI.set(
    opt,
    MOI.ObjectiveSense(),
    MOI.MIN_SENSE,
)

# ------------------------------------------------------------
# Build Boscia objective oracles
# ------------------------------------------------------------

f, grad! = Boscia.build_objective_oracles(opt)

# ------------------------------------------------------------
# Test at x = 3
#
# f(3) = (3 - 2)^2 = 1
# f'(3) = 2*3 - 4 = 2
# ------------------------------------------------------------

point = [3.0]

@test f(point) ≈ 1.0

storage = zeros(1)
grad!(storage, point)

@test storage[1] ≈ 2.0

println("f(3)     = ", f(point))
println("grad f(3) = ", storage[1])

println("\n=== All objective oracle tests passed ===")