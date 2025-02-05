using LinearAlgebra
using Distributions
import Random
using SCIP
using HiGHS
import MathOptInterface
const MOI = MathOptInterface
import Boscia
import FrankWolfe
using Test

# Testing of the interface function solve

n = 20
diffi = Random.rand(Bool, n) * 0.6 .+ 0.3

@testset "Norm hyperbox" begin
    function f(x)
        return sum(0.5 * (x .- diffi) .^ 2)
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    function build_norm_lmo()
        o = SCIP.Optimizer()
        MOI.set(o, MOI.Silent(), true)
        MOI.empty!(o)
        x = MOI.add_variables(o, n)
        for xi in x
            MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
            MOI.add_constraint(o, xi, MOI.LessThan(1.0))
            MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
        end
        return FrankWolfe.MathOptLMO(o)
    end
    x_baseline, _, result =
        Boscia.solve(f, grad!, build_norm_lmo(), verbose=false, dual_tightening=false)
    x_tighten, _, result =
        Boscia.solve(f, grad!, build_norm_lmo(), verbose=false, dual_tightening=true)
    x_strong, _, result = Boscia.solve(
        f,
        grad!,
        build_norm_lmo(),
        verbose=false,
        dual_tightening=true,
        strong_convexity=1.0,
    )

    @test x_baseline == round.(diffi)
    @test f(x_tighten) == f(result[:raw_solution])
    @test f(x_tighten) ≈ f(x_baseline)
    @test f(x_tighten) ≈ f(x_strong)
end

@testset "Norm hyperbox - strong branching" begin
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

    function f(x)
        return sum(0.5 * (x .- diffi) .^ 2)
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    blmo = Boscia.MathOptBLMO(HiGHS.Optimizer())
    branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
    MOI.set(branching_strategy.bounded_lmo.o, MOI.Silent(), true)

    x, _, result = Boscia.solve(f, grad!, lmo, verbose=false, branching_strategy=branching_strategy)

    @test x == round.(diffi)
    @test f(x) == f(result[:raw_solution])
end

@testset "Normbox - Shadow set activation" begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end
    @testset "Using shadow set" begin
        int_vars = collect(1:n)
        lbs = zeros(n)
        ubs = ones(n)

        sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)

        x, _, result =
            Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n)

        @test x == round.(diffi)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end
    @testset "Not using shadow set" begin
        int_vars = collect(1:n)
        lbs = zeros(n)
        ubs = ones(n)

        sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)

        x, _, result =
            Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, use_shadow_set=false)

        @test x == round.(diffi)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end
end

@testset "Normbox - strong convexity and sharpness" begin
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    @testset "Strong convexity" begin
        int_vars = collect(1:n)
        lbs = zeros(n)
        ubs = ones(n)

        sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
        μ = 1.0

        x, _, result =
            Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, strong_convexity=μ)

        @test x == round.(diffi)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Sharpness" begin
        int_vars = collect(1:n)
        lbs = zeros(n)
        ubs = ones(n)

        sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
        θ = 1/2
        M = 2.0

        x, _, result =
            Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, sharpness_constant=M, sharpness_exponent=θ)

        @test x == round.(diffi)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Strong convexity and sharpness" begin
        int_vars = collect(1:n)
        lbs = zeros(n)
        ubs = ones(n)

        sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
        μ = 1.0
        θ = 1/2
        M = 2.0

        x, _, result =
            Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, strong_convexity=μ, sharpness_constant=M, sharpness_exponent=θ)

        @test x == round.(diffi)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end
end

@testset "Start with Active Set" begin

    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    int_vars = collect(1:n)
    lbs = zeros(n)
    ubs = ones(n)

    sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
    direction =rand(n)
    v = Boscia.bounded_compute_extreme_point(sblmo, direction, lbs, ubs, int_vars)
    active_set = FrankWolfe.ActiveSet([(1.0, v)])

    x, _, result = Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, active_set=active_set)

    @test x == round.(diffi)
    @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
end

# Sparse Poisson regression
# min_{w, b, z} ∑_i exp(w x_i + b) - y_i (w x_i + b) + α norm(w)^2
# s.t. -N z_i <= w_i <= N z_i
# b ∈ [-N, N]
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p

n = 30
p = n

# underlying true weights
const ws = rand(Float64, p)
# set 50 entries to 0
for _ in 1:20
    ws[rand(1:p)] = 0
end
const bs = rand(Float64)
const Xs = randn(Float64, n, p)
const ys = map(1:n) do idx
    a = dot(Xs[idx, :], ws) + bs
    return rand(Distributions.Poisson(exp(a)))
end
Ns = 0.1

@testset "Interface - sparse poisson regression" begin
    k = 10
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    w = MOI.add_variables(o, p)
    z = MOI.add_variables(o, p)
    b = MOI.add_variable(o)
    for i in 1:p
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())
    end
    for i in 1:p
        MOI.add_constraint(o, -Ns * z[i] - w[i], MOI.LessThan(0.0))
        MOI.add_constraint(o, Ns * z[i] - w[i], MOI.GreaterThan(0.0))
        # Indicator: z[i] = 1 => -N <= w[i] <= N
        gl = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),
            ],
            [0.0, 0.0],
        )
        gg = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),
            ],
            [0.0, 0.0],
        )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(Ns)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-Ns)))
    end
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(o, sum(z, init=0.0), MOI.GreaterThan(1.0))
    MOI.add_constraint(o, b, MOI.LessThan(Ns))
    MOI.add_constraint(o, b, MOI.GreaterThan(-Ns))
    lmo = FrankWolfe.MathOptLMO(o)

    α = 1.3
    function f(θ)
        w = @view(θ[1:p])
        b = θ[end]
        s = sum(1:n) do i
            a = dot(w, Xs[:, i]) + b
            return 1 / n * (exp(a) - ys[i] * a)
        end
        return s + α * norm(w)^2
    end
    function grad!(storage, θ)
        w = @view(θ[1:p])
        b = θ[end]
        storage[1:p] .= 2α .* w
        storage[p+1:2p] .= 0
        storage[end] = 0
        for i in 1:n
            xi = @view(Xs[:, i])
            a = dot(w, xi) + b
            storage[1:p] .+= 1 / n * xi * exp(a)
            storage[1:p] .-= 1 / n * ys[i] * xi
            storage[end] += 1 / n * (exp(a) - ys[i])
        end
        storage ./= norm(storage)
        return storage
    end

    x, _, result = Boscia.solve(f, grad!, lmo, verbose=false)

    @test sum(x[p+1:2p]) <= k
    @test f(x) <= f(result[:raw_solution])

    x2, _, result = Boscia.solve(f, grad!, lmo, start_solution=x)
    @test sum(x2[p+1:2p]) <= k
    @test f(x2) == f(x)
end

n = 20
diffi = Random.rand(Bool, n) * 0.6 .+ 0.3

@testset "Different FW variants" begin

    function build_model()
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
        return lmo
    end

    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    lmo = build_model()
    x_afw, _, result_afw =
        Boscia.solve(f, grad!, lmo, verbose=false, variant=Boscia.AwayFrankWolfe())

    lmo = build_model()
    x_blended, _, result_blended =
        Boscia.solve(f, grad!, lmo, verbose=false, variant=Boscia.Blended())

    lmo = build_model()
    x_bpcg, _, result_bpcg = Boscia.solve(f, grad!, lmo, verbose=false, variant=Boscia.BPCG())

    lmo = build_model()
    x_vfw, _, result_vfw =
        Boscia.solve(f, grad!, lmo, verbose=false, variant=Boscia.VanillaFrankWolfe())

    @test isapprox(f(x_afw), f(result_afw[:raw_solution]), atol=1e-6, rtol=1e-3)
    @test isapprox(f(x_blended), f(result_blended[:raw_solution]), atol=1e-6, rtol=1e-3)
    @test isapprox(f(x_bpcg), f(result_bpcg[:raw_solution]), atol=1e-6, rtol=1e-3)
    @test isapprox(f(x_vfw), f(result_vfw[:raw_solution]), atol=1e-6, rtol=1e-3)

    @test sum(isapprox.(x_afw, x_blended, atol=1e-6, rtol=1e-3)) == n
    @test sum(isapprox.(x_blended, x_bpcg, atol=1e-6, rtol=1e-3)) == n
    @test sum(isapprox.(x_bpcg, x_vfw, atol=1e-6, rtol=1e-3)) == n
    @test sum(isapprox.(x_vfw, x_afw, atol=1e-6, rtol=1e-3)) == n
end

@testset "Different line search types" begin

    function build_model()
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
        return lmo
    end

    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end


    lmo = build_model()
    line_search = FrankWolfe.Adaptive()
    x_adaptive, _, result_adaptive =
        Boscia.solve(f, grad!, lmo, verbose=false, line_search=line_search)

    lmo = build_model()
    line_search = FrankWolfe.MonotonicStepSize()
    x_monotonic, _, result_monotonic =
        Boscia.solve(f, grad!, lmo, verbose=false, line_search=line_search, time_limit=120)

    lmo = build_model()
    line_search = FrankWolfe.Agnostic()
    x_agnostic, _, result_agnostic =
        Boscia.solve(f, grad!, lmo, verbose=false, line_search=line_search, time_limit=120)

    @test isapprox(f(x_adaptive), f(result_adaptive[:raw_solution]), atol=1e-6, rtol=1e-3)
    @test isapprox(f(x_monotonic), f(result_monotonic[:raw_solution]), atol=1e-6, rtol=1e-3)
    @test isapprox(f(x_agnostic), f(result_agnostic[:raw_solution]), atol=1e-6, rtol=1e-3)

    @test sum(isapprox.(x_adaptive, x_monotonic, atol=1e-6, rtol=1e-3)) == n
    @test sum(isapprox.(x_agnostic, x_monotonic, atol=1e-6, rtol=1e-3)) == n
    @test sum(isapprox.(x_adaptive, x_agnostic, atol=1e-6, rtol=1e-3)) == n
end

n = 20
diffi = Random.rand(Bool, n) * 0.6 .+ 0.3

@testset "Lazification" begin

    function build_model()
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
        return lmo
    end

    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    lmo = build_model()
    x_lazy, _, result_lazy = Boscia.solve(f, grad!, lmo, verbose=false)

    lmo = build_model()
    x_no, _, result_no = Boscia.solve(f, grad!, lmo, verbose=false, lazy=false)

    lmo = build_model()
    x_mid, _, result_mid = Boscia.solve(f, grad!, lmo, verbose=false, lazy=true, lazy_tolerance=1.5)

    @test isapprox(f(x_lazy), f(result_lazy[:raw_solution]), atol=1e-6, rtol=1e-2)
    @test isapprox(f(x_no), f(result_no[:raw_solution]), atol=1e-6, rtol=1e-2)
    @test isapprox(f(x_mid), f(result_mid[:raw_solution]), atol=1e-6, rtol=1e-2)
    @test sum(isapprox.(x_lazy, x_no, atol=1e-6, rtol=1e-2)) == n
    @test sum(isapprox.(x_lazy, x_mid, atol=1e-6, rtol=1e-2)) == n
end
