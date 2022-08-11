using Test
using Boscia
using FrankWolfe
using Random
using SCIP
import MathOptInterface
import Bonobo
using HiGHS
using Printf
using Dates
const MOI = MathOptInterface
const MOIU = MOI.Utilities

import MathOptSetDistances
const MOD = MathOptSetDistances
@testset "Integer bounds" begin
    n=10
    integer_bound = Boscia.IntegerBounds()
    @test isempty(integer_bound)
    push!(integer_bound, (1, MOI.GreaterThan(5.0)))
    push!(integer_bound, (2, MOI.LessThan(0.0)))
    push!(integer_bound, (3, MOI.LessThan(4.0)))
    push!(integer_bound, (4, MOI.LessThan(0.0)))
    push!(integer_bound, (5, MOI.GreaterThan(5.0)))
    @test haskey(integer_bound.lower_bounds, 2) == false
    @test haskey(integer_bound.upper_bounds, 4)
    @test haskey(integer_bound.lower_bounds, 1)

    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        if xi.value != 3
            MOI.add_constraint(o, xi, MOI.LessThan(5.0))
        end
    end
    lmo = FrankWolfe.MathOptLMO(o)

    global_bounds = Boscia.IntegerBounds()
    @test isempty(global_bounds)
    for i = 1:n
        push!(global_bounds, (i, MOI.GreaterThan(0.0)))
        if i != 3
            push!(global_bounds, (i, MOI.LessThan(5.0)))
        end
    end

    Boscia.build_LMO(lmo, global_bounds, integer_bound, collect(1:n))

    @test Boscia.is_linear_feasible(o, ones(n)) == false
    @test Boscia.is_linear_feasible(o, vcat([5.0, 0.0, 1.5, 0.0, 5.0], ones(n-5)))
    @test Boscia.is_linear_feasible(o, vcat([5.0, 2.0, 1.5, 0.0, 5.0], ones(n-5))) == false
    @test Boscia.is_linear_feasible(o, vcat([5.0, 0.0, 1.5, 0.0, 3.0], ones(n-5))) == false
    @test Boscia.is_linear_feasible(o, vcat([5.0, 0.0, 4.5, 0.0, 5.0], ones(n-5))) == false
end