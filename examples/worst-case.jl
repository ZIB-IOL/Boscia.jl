using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface

# The example from  "Optimizing a low-dimensional convex function over a high-dimensional cube"
# by Christoph Hunkenschröder, Sebastian Pokutta, Robert Weismantel
# https://arxiv.org/abs/2204.05266 after Lemma 2.2
#
# This example is a good example, basically
#
#            min_{x in [0,1]^n} ||x - (1/2, ..., 1/2) ||^2
#
# as all the gradient at each 0/1 point "look" the same, i.e., 
# (+/- 1/2, ...., +/- 1/2) as gradients.
#
# Any 0/1 solution is optimal (if alpha = 0) in this case 
# but you need to *prove* it. Hence, you are forced to clear out 
# the full tree. For a given n the tree thus has 2^(n+1) - 1 nodes
# which is exactly what we see when solving.
#
#
# We can also set alpha > 0, which is essentially the same the same problem but the 
# (1/2, ..., 1/2) point is slightly perturbed so that the problem has (usually)
# a unique solution. See below for the exact realization

n = 10

@testset "Interface - 2-norm over hypercube -- α = $alpha" for alpha in (0.0, 0.05)
    diff_point = 0.5 * ones(n) + Random.rand(n) * alpha * 1 / n
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
        MOI.add_constraint(o, xi, MOI.ZeroOne())
    end
    lmo = FrankWolfe.MathOptLMO(o)

    function f(x)
        return 0.5 * sum((x .- diff_point) .^ 2)
    end

    function grad!(storage, x)
        @. storage = x - diff_point
    end

    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true)

    # build optimal solution
    xopt = zeros(n)
    for i in 1:n
        if diff_point[i] > 0.5
            xopt[i] = 1
        end
    end

    @test f(x) == f(xopt)
    if alpha == 0 # next test is only valid if we do not have any perturbation on the continuous opt
        @test result[:number_nodes] == 2^(n + 1) - 1
    end

    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    println("\nNumber of processed nodes should be: ", 2^(n + 1) - 1)
    println()

    # test if number of nodes is still correct when stopping FW early
    x, _, result = Boscia.solve(f, grad!, lmo, verbose=false, min_number_lower=5)
    @test result[:number_nodes] == 2^(n + 1) - 1

    x_strong, _, result_strong = Boscia.solve(f, grad!, lmo, verbose=true, strong_convexity=1.0)

    @test f(x_strong) == f(x)
end
