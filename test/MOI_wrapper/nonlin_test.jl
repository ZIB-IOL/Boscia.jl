using Boscia
using JuMP
using SCIP
using Test
using StableRNGs

# User-defined operators registered with @operator
#   min ∑_i (x_i - d_i)^2  s.t. x ∈ {0,1}^3
# The objective is written as a user-defined operator op_f.
# The optimal solution is round.(d).

@testset "User-defined operator without gradient" begin
    d = [0.3, 0.9, 0.3]
    f(x...) = sum((x[i] - d[i])^2 for i in 1:3)

    model = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model)
    @variable(model, 0 <= x[1:3] <= 1, Int)
    @operator(model, op_f, 3, f)
    @objective(model, Min, op_f(x...))

    @test_throws "gradient" optimize!(model)
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
    @test value.(x) ≈ round.(d) atol = 1e-8
    @test objective_value(model) ≈ f(round.(d)...) atol = 1e-8
end

# f(x) = log(∑_j exp(a_j' x)) + ∑_i (x_i - c_i)^2 + ∑_i sqrt(1 + x_i^2)
@testset "objective oracle against symbolic expression" begin
    rng = StableRNG(111)
    n = 3
    m = 4
    A = randn(rng, m, n)
    c = [0.3, 1.6, 0.8]

    # oracle
    function f(x)
        a = A * x
        s = log(sum(exp.(a)))
        s += sum((x .- c) .^ 2)
        s += sum(sqrt.(1 .+ x .^ 2))
        return s
    end

    function grad!(storage, x)
        a = A * x
        s = exp.(a) ./ sum(exp.(a))
        storage .= A' * s .+ 2 .* (x .- c) .+ x ./ sqrt.(1 .+ x .^ 2)
        return storage
    end

    model = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model)

    @variable(model, 0 <= x[1:n] <= 2, Int)
    @operator(model, op_f, n, (x...) -> f(collect(x)), (g, x...) -> grad!(g, collect(x)))
    @objective(model, Min, op_f(x...))
    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL

    # symbolic
    model2 = Model(() -> Boscia.Optimizer(SCIP.Optimizer))
    set_silent(model2)

    @variable(model2, 0 <= x2[1:n] <= 2, Int)
    @objective(
        model2,
        Min,
        log(sum(exp(sum(A[j, i] * x2[i] for i in 1:n)) for j in 1:m)) +
        sum((x2[i] - c[i])^2 for i in 1:n) +
        sum(sqrt(1 + x2[i]^2) for i in 1:n)
    )
    optimize!(model2)

    @test termination_status(model2) == MOI.OPTIMAL

    @test objective_value(model) ≈ objective_value(model2) atol = 1e-8
    @test value.(x) ≈ value.(x2) atol = 1e-8
end
