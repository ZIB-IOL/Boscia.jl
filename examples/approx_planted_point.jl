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
using StableRNGs

println("\nApproximate Planted Point Example")

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

include("cube_blmo.jl")

n = 20
diffi = rand(rng, Bool, n) * 0.6 .+ 0.3

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
        blmo = Boscia.MathOptBLMO(o)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = true
        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = true
        x, _, result = Boscia.solve(f, grad!, blmo, settings=settings)

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
        blmo = CubeBLMO(n, int_vars, bounds)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = true
        x, _, result = Boscia.solve(f, grad!, blmo, settings=settings)

        @test x == round.(diffi)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Using Cube Simple LMO" begin
        int_vars = collect(1:n)
        lbs = zeros(n)
        ubs = ones(n)

        sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
        blmo = Boscia.ManagedBoundedLMO(sblmo, lbs[int_vars], ubs[int_vars], int_vars, n)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = true
        x, _, result = Boscia.solve(f, grad!, blmo, settings=settings)

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
        blmo = Boscia.MathOptBLMO(o)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = true
        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = true
        x, _, result = Boscia.solve(f, grad!, blmo, settings=settings)

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
        blmo = CubeBLMO(n, int_vars, bounds)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = true
        x, _, result = Boscia.solve(f, grad!, blmo, settings=settings)

        sol = diffi
        sol[int_vars] = round.(sol[int_vars])
        @test sum(isapprox.(x, sol, atol=1e-6, rtol=1e-2)) == n
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Using Cube Simple LMO" begin
        lbs = zeros(n)
        ubs = ones(n)

        sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
        blmo = Boscia.ManagedBoundedLMO(sblmo, lbs[int_vars], ubs[int_vars], int_vars, n)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = true
        x, _, result = Boscia.solve(f, grad!, blmo, settings=settings)

        sol = diffi
        sol[int_vars] = round.(sol[int_vars])
        @test sum(isapprox.(x, sol, atol=1e-6, rtol=1e-2)) == n
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end
end
