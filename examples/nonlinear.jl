using FrankWolfe
using LinearAlgebra
import MathOptInterface
using Random
using Boscia
using Bonobo
import Bonobo
using Printf
using Dates
using StableRNGs

println("\nNonlinear Example")

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

n = 30

# using SCIP
# const MOI = MathOptInterface


################################################################
# alternative implementation of LMO using MOI and SCIP
################################################################
# o = SCIP.Optimizer()
# MOI.set(o, MOI.Silent(), true)
# MOI.empty!(o)
# x = MOI.add_variables(o, n)
# for xi in x
#     MOI.add_constraint(o, xi, MOI.ZeroOne())
#     MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
#     MOI.add_constraint(o, xi, MOI.LessThan(1.0))
# end
# lmo = FrankWolfe.MathOptLMO(o)


################################################################
# LMO via CubeSimpleBLMO
################################################################
int_vars = collect(1:n)

lbs = zeros(n)
ubs = ones(n)

sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
# wrap the sblmo into a bound manager
lmo = Boscia.ManagedBoundedLMO(sblmo, lbs[int_vars], ubs[int_vars], int_vars, n)

const A = let
    A = randn(rng, n, n)
    A' * A
end

@assert isposdef(A) == true

const y = rand(rng, Bool, n) * 0.6 .+ 0.3

function f(x)
    d = x - y
    return dot(d, A, d)
end

function grad!(storage, x)
    # storage = Ax
    mul!(storage, A, x)
    # storage = 2Ax - 2Ay
    return mul!(storage, A, y, -2, 2)
end

# benchmarking Oracles
FrankWolfe.benchmark_oracles(f, grad!, () -> rand(n), lmo; k=100)

#################


# are these lmo calls counted as well?

# #####
# # follow the gradient for a fixed number of steps and collect solutions on the way
# #####

# function follow_gradient_heuristic(tree::Bonobo.BnBTree, blmo::Boscia.BoundedLinearMinimizationOracle, x, k)
#     nabla = similar(x)
#     x_new = copy(x)
#     sols = []
#     for i in 1:k
#         tree.root.problem.g(nabla,x_new)
#         x_new = Boscia.compute_extreme_point(blmo, nabla)
#         push!(sols, x_new)
#     end
#     return sols, false
# end

# #####
# # rounding respecting the hidden feasible region structure
# #####

# function rounding_lmo_01_heuristic(tree::Bonobo.BnBTree, blmo::Boscia.BoundedLinearMinimizationOracle, x)
#     nabla = zeros(length(x))
#     for idx in tree.branching_indices
#         nabla[idx] = 1 - 2*round(x[idx]) # (0.7, 0.3) -> (1, 0) -> (-1, 1) -> min -> (1,0)
#     end
#     x_rounded = Boscia.compute_extreme_point(blmo, nabla)
#     return [x_rounded], false
# end

#####
# geometric scaling like for a couple of steps
#####


depth = 5
heu = Boscia.Heuristic(
    (tree, blmo, x) -> Boscia.follow_gradient_heuristic(tree, blmo, x, depth),
    0.8,
    :follow_gradient,
)
heu2 = Boscia.Heuristic(Boscia.rounding_lmo_01_heuristic, 0.8, :lmo_rounding)

heuristics = [heu, heu2]
# heuristics = []

x, _, _ = Boscia.solve(
    f,
    grad!,
    lmo,
    settings_bnb=Boscia.settings_bnb(verbose=true, print_iter=500, time_limit=300),
    settings_heuristic=Boscia.settings_heuristic(custom_heuristics=heuristics),
)

@show x
