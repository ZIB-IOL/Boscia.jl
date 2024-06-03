using FrankWolfe
using FiniteDifferences
using LinearAlgebra
using Random
using Boscia
using Test

include("poisson_reg.jl")

"""
Check if the gradient using finite differences matches the grad! provided.
Copied from FrankWolfe package: https://github.com/ZIB-IOL/FrankWolfe.jl/blob/master/examples/plot_utils.jl
"""
function check_gradients(grad!, f, gradient, num_tests=10, tolerance=1.0e-5)
    for i in 1:num_tests
        random_point = rand(length(gradient))
        grad!(gradient, random_point)
        if norm(grad(central_fdm(5, 1), f, random_point)[1] - gradient) > tolerance
            @warn "There is a noticeable difference between the gradient provided and
            the gradient computed using finite differences.:\n$(norm(grad(central_fdm(5, 1), f, random_point)[1] - gradient))"
            return false
        end
    end
    return true
end

@testset "Poisson gradient" begin
    for dim in [20,50,80]
        for seed in 1:10
            @show dim, seed
            f, grad!, p, α, bs, Xs, ys, ws = build_function(seed, dim)
            gradient = rand(2*dim + 1)

            @test check_gradients(grad!, f, gradient)
        end
    end
end

#poisson_reg_shot(1, 50, 1.0)
#poisson_reg_boscia(1, 50, 1.0)

poisson_reg_ipopt(1, 50, 10.0, time_limit=1800)
poisson_reg_boscia(1, 50, 10.0, limit = 1800, bo_mode="default")
poisson_reg_shot(1, 50, 10.0, time_limit = 1800)