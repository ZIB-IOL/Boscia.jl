using Boscia
using JuMP
using SCIP
using Test

# User-defined operators registered with @operator
#   min ∑_i (x_i - d_i)^2  s.t. x ∈ {0,1}^3
# The objective is written as a user-defined operator op_f.
# The optimal solution is round.(d).

@testset "User-defined operator" begin
    d = [0.3, 0.9, 0.3]
    f(x...) = sum((x[i] - d[i])^2 for i in 1:3)

    model = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model)
    @variable(model, 0 <= x[1:3] <= 1, Int)
    @operator(model, op_f, 3, f)
    @objective(model, Min, op_f(x...))
    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test value.(x) ≈ round.(d) atol = 1e-6
    @test objective_value(model) ≈ f(round.(d)...) atol = 1e-6
end

@testset "User-defined operator with gradient" begin
    d = [0.3, 0.9, 0.3]
    f(x...) = sum((x[i] - d[i])^2 for i in 1:3)
    function ∇f(g::AbstractVector{T}, x::T...) where {T}
        for i in 1:3
            g[i] = 2 * (x[i] - d[i])
        end
        return
    end

    model = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model)
    @variable(model, 0 <= x[1:3] <= 1, Int)
    @operator(model, op_f, 3, f, ∇f)
    @objective(model, Min, op_f(x...))
    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test value.(x) ≈ round.(d) atol = 1e-6
    @test objective_value(model) ≈ f(round.(d)...) atol = 1e-6
end
