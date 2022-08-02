using BranchWolfe
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
alpha = 0.00

const diffw = 0.5 * ones(n) + Random.rand(n)* alpha * 1/n

@testset "Interface - 2-norm over hypercube" begin
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
        return sum(0.5*(x.-diffw).^2)
    end

    function grad!(storage, x)
        @. storage = x-diffw
    end

    x, _, result = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = true)

    # build optimal solution
    xopt = zeros(n)
    for i in 1:n
        if diffw[i] > 0.5
            xopt[i] = 1
        end
    end

    @test f(x) == f(xopt)
    if alpha == 0 # next test is only valid if we do not have any perturbation on the continuous opt
        @test result[:number_nodes] ==  2^(n+1)-1
    end

    println("\nNumber of processed nodes should be: ", 2^(n+1)-1)
    println()
end
