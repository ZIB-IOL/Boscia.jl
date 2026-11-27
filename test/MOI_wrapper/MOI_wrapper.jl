
using Test
using Boscia
import MathOptInterface as MOI
using SCIP

@testset "Boscia Optimizer skeleton" begin
    opt = Boscia.Optimizer(SCIP.Optimizer)

    @test opt isa Boscia.Optimizer
    @test MOI.is_empty(opt)

    MOI.empty!(opt)

    @test MOI.is_empty(opt)

    # attributes 
    @test !MOI.get(opt, MOI.Silent())
    MOI.set(opt, MOI.Silent(), true)
    @test MOI.get(opt, MOI.Silent())

end