using Test
using JuMP
using Boscia
using SCIP

# Solve the model built by `build!` and return the model and its variables.
function solve_model(build!)
    model = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model)
    x = build!(model)
    optimize!(model)
    return model, x
end

# Solve the same problem with a custom set and with ordinary constraints
# (inner MIP solver), and check that both give the same solution.
function compare_with_scip(build_custom!, build_mip!)
    model_custom, x_custom = solve_model(build_custom!)
    model_mip, x_mip = solve_model(build_mip!)
    @test termination_status(model_custom) == MOI.OPTIMAL
    @test value.(x_custom) ≈ value.(x_mip) atol = 1e-6
    @test objective_value(model_custom) ≈ objective_value(model_mip) atol = 1e-6
    return value.(x_custom)
end

# Data for the simplex sets and for the hypersimplex / reverse knapsack sets.
const a_simplex = [0.2, 1.8, 2.1, 0.9, 1.0]
const a_binary = [0.2, 0.8, 0.6, 0.1, 0.9]

@testset "ProbabilitySimplex" begin
    x = compare_with_scip(
        function (m)
            x = @variable(m, [1:5], Int)
            @constraint(m, x in Boscia.ProbabilitySimplex(5, 6.0))
            @objective(m, Min, sum((x[i] - a_simplex[i])^2 for i in 1:5))
            return x
        end,
        function (m)
            x = @variable(m, [1:5], Int, lower_bound = 0, upper_bound = 6)
            @constraint(m, sum(x) == 6)
            @objective(m, Min, sum((x[i] - a_simplex[i])^2 for i in 1:5))
            return x
        end,
    )
    @test x ≈ [0.0, 2.0, 2.0, 1.0, 1.0] atol = 1e-6
end

@testset "ProbabilitySimplex: infeasible" begin
    model = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model)
    @variable(model, x[1:5], Int)
    @constraint(model, x in Boscia.ProbabilitySimplex(5, -1.0))
    @objective(model, Min, sum(x[i]^2 for i in 1:5))
    optimize!(model)
    @test termination_status(model) == MOI.INFEASIBLE
end

@testset "UnitSimplex" begin
    x = compare_with_scip(
        function (m)
            x = @variable(m, [1:5], Int)
            @constraint(m, x in Boscia.UnitSimplex(5, 4.0))
            @objective(m, Min, sum((x[i] - a_simplex[i])^2 for i in 1:5))
            return x
        end,
        function (m)
            x = @variable(m, [1:5], Int, lower_bound = 0, upper_bound = 4)
            @constraint(m, sum(x) <= 4)
            @objective(m, Min, sum((x[i] - a_simplex[i])^2 for i in 1:5))
            return x
        end,
    )
    @test x ≈ [0.0, 1.0, 2.0, 0.0, 1.0] atol = 1e-6
end

@testset "HyperSimplex" begin
    x = compare_with_scip(
        function (m)
            x = @variable(m, [1:5], Int)
            @constraint(m, x in Boscia.HyperSimplex(5, 2, 1.0))
            @objective(m, Min, sum((x[i] - a_binary[i])^2 for i in 1:5))
            return x
        end,
        function (m)
            x = @variable(m, [1:5], Bin)
            @constraint(m, sum(x) == 2)
            @objective(m, Min, sum((x[i] - a_binary[i])^2 for i in 1:5))
            return x
        end,
    )
    @test x ≈ [0.0, 1.0, 0.0, 0.0, 1.0] atol = 1e-6
end

@testset "UnitHyperSimplex" begin
    x = compare_with_scip(
        function (m)
            x = @variable(m, [1:5], Int)
            @constraint(m, x in Boscia.UnitHyperSimplex(5, 2, 1.0))
            @objective(m, Min, sum((x[i] - a_binary[i])^2 for i in 1:5))
            return x
        end,
        function (m)
            x = @variable(m, [1:5], Bin)
            @constraint(m, sum(x) <= 2)
            @objective(m, Min, sum((x[i] - a_binary[i])^2 for i in 1:5))
            return x
        end,
    )
    @test x ≈ [0.0, 1.0, 0.0, 0.0, 1.0] atol = 1e-6
end

@testset "ReverseKnapsack" begin
    x = compare_with_scip(
        function (m)
            x = @variable(m, [1:5], Int)
            @constraint(m, x in Boscia.ReverseKnapsack(5, 4.0, 1.0))
            @objective(m, Min, sum((x[i] - a_binary[i])^2 for i in 1:5))
            return x
        end,
        function (m)
            x = @variable(m, [1:5], Bin)
            @constraint(m, sum(x) >= 4)
            @objective(m, Min, sum((x[i] - a_binary[i])^2 for i in 1:5))
            return x
        end,
    )
    @test x ≈ [1.0, 1.0, 1.0, 0.0, 1.0] atol = 1e-6
end
