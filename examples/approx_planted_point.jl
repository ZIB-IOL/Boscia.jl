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

        x_scip = x
    end

    @testset "Using Cube LMO" begin
        int_vars = []
        bin_vars = collect(1:n)
        
        bounds = Boscia.IntegerBounds()
        for i in 1:n 
            push!(bounds, (i, MOI.GreaterThan(0.0)))
            push!(bounds, (i, MOI.LessThan(0.0)))
        end
        blmo = CubeBLMO(n, int_vars, bin_vars, bounds)

        x, _, result = Boscia.solve(f, grad!, blmo, verbose =true)

        @test x == round.(diffi)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end
end

