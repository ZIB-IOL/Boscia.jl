using Boscia
using FrankWolfe
using Test
using Random
using SCIP
# using Statistics
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface

include("cube_blmo.jl")

n = 20
diffi = Random.rand(Bool, n) * 0.6 .+ 0.3

@testset "Approximate planted point - Integer" begin

    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    @testset "Using SCIP" begin
        o = SCIP.Optimizer()
        MOI.set(o, MOI.Silent(), true)
        MOI.empty!(o)
        x = MOI.add_variables(o, n)
        for xi in x
            MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
            MOI.add_constraint(o, xi, MOI.LessThan(1.0))
            MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
        end
        lmo = FrankWolfe.MathOptLMO(o)

        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true)

        @test x == round.(diffi)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Using Cube LMO" begin
        int_vars = collect(1:n)

        bounds = Boscia.IntegerBounds()
        for i in 1:n
            push!(bounds, (i, 0.0), :greaterthan)
            push!(bounds, (i, 1.0), :lessthan)
        end
        blmo = Boscia.CubeBLMO(n, int_vars, bounds)

        x, _, result = Boscia.solve(f, grad!, blmo, verbose=true)

        @test x == round.(diffi)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Using Cube Simple LMO" begin
        int_vars = collect(1:n)
        lbs = zeros(n)
        ubs = ones(n)

        sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)

        x, _, result =
            Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, verbose=true)

        @test x == round.(diffi)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end
end


@testset "Approximate planted point - Mixed" begin

    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    int_vars = unique!(rand(collect(1:n), Int(floor(n / 2))))

    @testset "Using SCIP" begin
        o = SCIP.Optimizer()
        MOI.set(o, MOI.Silent(), true)
        MOI.empty!(o)
        x = MOI.add_variables(o, n)
        for xi in x
            MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
            MOI.add_constraint(o, xi, MOI.LessThan(1.0))
            if xi.value in int_vars
                MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
            end
        end
        lmo = FrankWolfe.MathOptLMO(o)

        x, _, result = Boscia.solve(f, grad!, lmo, verbose=true)

        sol = diffi
        sol[int_vars] = round.(sol[int_vars])
        @test sum(isapprox.(x, sol, atol=1e-6, rtol=1e-2)) == n
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Using Cube LMO" begin
        bounds = Boscia.IntegerBounds()
        for i in 1:n
            push!(bounds, (i, 0.0), :greaterthan)
            push!(bounds, (i, 1.0), :lessthan)
        end
        blmo = Boscia.CubeBLMO(n, int_vars, bounds)

        x, _, result = Boscia.solve(f, grad!, blmo, verbose=true)

        sol = diffi
        sol[int_vars] = round.(sol[int_vars])
        @test sum(isapprox.(x, sol, atol=1e-6, rtol=1e-2)) == n
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Using Cube Simple LMO" begin
        lbs = zeros(n)
        ubs = ones(n)

        sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)

        x, _, result =
            Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, verbose=true)

        sol = diffi
        sol[int_vars] = round.(sol[int_vars])
        @test sum(isapprox.(x, sol, atol=1e-6, rtol=1e-2)) == n
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end
end
