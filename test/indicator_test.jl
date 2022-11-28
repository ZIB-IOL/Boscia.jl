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

@testset "Indicators" begin
    n = 5
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    z = MOI.add_variables(o, n)
    for i in 1:n
        MOI.add_constraint(o, x[i], MOI.GreaterThan(-1.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0))

        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne()) 
    end 
    lmo = FrankWolfe.MathOptLMO(o)

    @test Boscia.indicator_present(lmo.o) == false

    for i in 1:n
        gl = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[i])),],
            [0.0, 0.0], )
        gg = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, x[i])),],
            [0.0, 0.0], )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(0.0)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(0.0)))
    end
    @test Boscia.indicator_present(lmo.o) == true

    function ind_rounding(x)
        round.(x[n+1:2n])
        for i in 1:n
            if isapprox(x[n+i], 1.0)
                x[i] = 0.0
            end
        end 
    end

    x = [0.5, 1.0, 0.75, 0.0, 0.9, 1.0, 1.0, 1.0, 0.0, 0.0]
    y = [0.0, 0.0, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    @test Boscia.is_indicator_feasible(o, x) == false
    @test Boscia.is_indicator_feasible(o, y) == true
    ind_rounding(x)
    @test Boscia.is_indicator_feasible(o, x) == true 
end
