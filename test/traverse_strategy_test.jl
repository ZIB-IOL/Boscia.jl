using Boscia
using Bonobo
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra

import MathOptInterface
const MOI = MathOptInterface

function build_examples(o, n, seed)
    Random.seed!(seed)
    A = let
        A = randn(n, n)
        A' * A
    end

    @assert isposdef(A)

    y = Random.rand(Bool, n) * 0.6 .+ 0.3

    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.ZeroOne())
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
    end
    lmo = Boscia.MathOptBLMO(o)

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
    return f, grad!, lmo
end


@testset "DepthFirstSearch Traverse Strategy" begin
    dimension = 20
    seed = 1
    o = SCIP.Optimizer()
    f, grad!, lmo = build_examples(o, dimension, seed)
    time_limit = 60

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:time_limit] = time_limit
    x_mi, _, result_mi = Boscia.solve(f, grad!, lmo, settings=settings)

    @testset "DepthFirstSearch favoring right" begin
        o = SCIP.Optimizer()
        f, grad!, lmo = build_examples(o, dimension, seed)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = false
        settings.branch_and_bound[:time_limit] = time_limit
        settings.branch_and_bound[:traverse_strategy] = Boscia.DepthFirstSearch(true)
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test isapprox(f(x_mi), f(x), atol=1e-6, rtol=1e-3)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "DepthFirstSearch favoring left" begin
        o = SCIP.Optimizer()
        f, grad!, lmo = build_examples(o, dimension, seed)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:time_limit] = time_limit
        settings.branch_and_bound[:traverse_strategy] = Boscia.DepthFirstSearch(false)
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test isapprox(f(x_mi), f(x), atol=1e-6, rtol=1e-3)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end
end
