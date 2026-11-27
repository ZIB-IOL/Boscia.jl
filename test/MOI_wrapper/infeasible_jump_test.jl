using Test
using JuMP
using Boscia
using SCIP

# ------------------------------------------------------------
# Test 
# f = x1 + x2  
# x1, x2 binary
# x1 + x2 >= 3
# ------------------------------------------------------------

@testset "infeasible model" begin
    model = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model)

    @variable(model, x[i=1:2], Bin)
    @constraint(model, x[1] + x[2] >= 3)
    @objective(model, Min, x[1] + x[2])

    optimize!(model)

    @test termination_status(model) == MOI.INFEASIBLE
    @test primal_status(model) == MOI.NO_SOLUTION
end
