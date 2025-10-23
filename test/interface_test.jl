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
using StableRNGs
using Suppressor

println("\nInterface Tests")

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

# Testing of the interface function solve

n = 20
diffi = rand(rng, Bool, n) * 0.6 .+ 0.3

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
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.tightening[:dual_tightening] = false
    x_baseline, _, result = Boscia.solve(f, grad!, build_norm_lmo(), settings=settings)
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.tightening[:dual_tightening] = true
    x_tighten, _, result = Boscia.solve(f, grad!, build_norm_lmo(), settings=settings)
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.tightening[:dual_tightening] = true
    settings.tightening[:strong_convexity] = 1.0
    x_strong, _, result = Boscia.solve(f, grad!, build_norm_lmo(), settings=settings)

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

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.branch_and_bound[:branching_strategy] = branching_strategy
    x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

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

        x, _, result = Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n)

        @test x == round.(diffi)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end
    @testset "Not using shadow set" begin
        int_vars = collect(1:n)
        lbs = zeros(n)
        ubs = ones(n)

        sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:use_shadow_set] = false
        x, _, result = Boscia.solve(
            f,
            grad!,
            sblmo,
            lbs[int_vars],
            ubs[int_vars],
            int_vars,
            n,
            settings=settings,
        )

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

        settings = Boscia.create_default_settings()
        settings.tightening[:strong_convexity] = μ
        x, _, result = Boscia.solve(
            f,
            grad!,
            sblmo,
            lbs[int_vars],
            ubs[int_vars],
            int_vars,
            n,
            settings=settings,
        )

        @test x == round.(diffi)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Sharpness" begin
        int_vars = collect(1:n)
        lbs = zeros(n)
        ubs = ones(n)

        sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
        θ = 1 / 2
        M = 2.0

        settings = Boscia.create_default_settings()
        settings.tightening[:sharpness_constant] = M
        settings.tightening[:sharpness_exponent] = θ
        x, _, result = Boscia.solve(
            f,
            grad!,
            sblmo,
            lbs[int_vars],
            ubs[int_vars],
            int_vars,
            n,
            settings=settings,
        )

        @test x == round.(diffi)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Strong convexity and sharpness" begin
        int_vars = collect(1:n)
        lbs = zeros(n)
        ubs = ones(n)

        sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
        μ = 1.0
        θ = 1 / 2
        M = 2.0

        settings = Boscia.create_default_settings()
        settings.tightening[:strong_convexity] = μ
        settings.tightening[:sharpness_constant] = M
        settings.tightening[:sharpness_exponent] = θ
        x, _, result = Boscia.solve(
            f,
            grad!,
            sblmo,
            lbs[int_vars],
            ubs[int_vars],
            int_vars,
            n,
            settings=settings,
        )

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
    direction = rand(n)
    v = Boscia.bounded_compute_extreme_point(sblmo, direction, lbs, ubs, int_vars)
    active_set = FrankWolfe.ActiveSet([(1.0, v)])

    settings = Boscia.create_default_settings()
    settings.domain[:active_set] = active_set
    x, _, result =
        Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, settings=settings)

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
        return storage
    end

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

    @test sum(x[p+1:2p]) <= k
    @test f(x) <= f(result[:raw_solution])

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:start_solution] = x
    x2, _, result = Boscia.solve(f, grad!, lmo, settings=settings)
    @test sum(x2[p+1:2p]) <= k
    @test f(x2) == f(x)
end

n = 20
diffi = rand(rng, Bool, n) * 0.6 .+ 0.3

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
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:variant] = Boscia.AwayFrankWolfe()
    x_afw, _, result_afw = Boscia.solve(f, grad!, lmo, settings=settings)

    lmo = build_model()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:variant] = Boscia.BlendedConditionalGradient()
    x_blended, _, result_blended = Boscia.solve(f, grad!, lmo, settings=settings)

    lmo = build_model()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:variant] = Boscia.BlendedPairwiseConditionalGradient()
    x_bpcg, _, result_bpcg = Boscia.solve(f, grad!, lmo, settings=settings)

    lmo = build_model()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:variant] = Boscia.PairwiseFrankWolfe()
    x_pcg, _, result_pcg = Boscia.solve(f, grad!, lmo, settings=settings)

    lmo = build_model()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()
    settings.frank_wolfe[:fw_verbose] = false
    x_dicg, _, result_dicg = Boscia.solve(f, grad!, lmo, settings=settings)

    lmo = build_model()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:variant] = Boscia.StandardFrankWolfe()
    x_vfw, _, result_vfw = Boscia.solve(f, grad!, lmo, settings=settings)

    @test isapprox(f(x_afw), f(result_afw[:raw_solution]), atol=1e-6, rtol=1e-3)
    @test isapprox(f(x_blended), f(result_blended[:raw_solution]), atol=1e-6, rtol=1e-3)
    @test isapprox(f(x_pcg), f(result_pcg[:raw_solution]), atol=1e-6, rtol=1e-3)
    @test isapprox(f(x_bpcg), f(result_bpcg[:raw_solution]), atol=1e-6, rtol=1e-3)
    @test isapprox(f(x_dicg), f(result_dicg[:raw_solution]), atol=1e-6, rtol=1e-3)
    @test isapprox(f(x_vfw), f(result_vfw[:raw_solution]), atol=1e-6, rtol=1e-3)

    @test sum(isapprox.(x_afw, x_blended, atol=1e-6, rtol=1e-3)) == n
    @test sum(isapprox.(x_blended, x_bpcg, atol=1e-6, rtol=1e-3)) == n
    @test sum(isapprox.(x_bpcg, x_pcg, atol=1e-6, rtol=1e-3)) == n
    @test sum(isapprox.(x_pcg, x_vfw, atol=1e-6, rtol=1e-3)) == n
    @test sum(isapprox.(x_bpcg, x_dicg, atol=1e-6, rtol=1e-3)) == n
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
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:line_search] = line_search
    x_adaptive, _, result_adaptive = @suppress begin
        Boscia.solve(f, grad!, lmo, settings=settings)
    end

    lmo = build_model()
    line_search = FrankWolfe.MonotonicStepSize()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.branch_and_bound[:time_limit] = 60
    settings.frank_wolfe[:line_search] = line_search
    x_monotonic, _, result_monotonic = Boscia.solve(f, grad!, lmo, settings=settings)

    lmo = build_model()
    line_search = FrankWolfe.Agnostic()
    settings = Boscia.create_default_settings()
    settings = merge(
        settings,
        (
            branch_and_bound=merge(
                settings.branch_and_bound,
                Dict(:verbose => false, :time_limit => 60),
            ),
            frank_wolfe=merge(settings.frank_wolfe, Dict(:line_search => line_search)),
        ),
    )
    x_agnostic, _, result_agnostic = Boscia.solve(f, grad!, lmo, settings=settings)

    @test isapprox(f(x_adaptive), f(result_adaptive[:raw_solution]), atol=1e-6, rtol=1e-3)
    @test isapprox(f(x_monotonic), f(result_monotonic[:raw_solution]), atol=1e-6, rtol=1e-3)
    @test isapprox(f(x_agnostic), f(result_agnostic[:raw_solution]), atol=1e-6, rtol=1e-3)

    @test sum(isapprox.(x_adaptive, x_monotonic, atol=1e-6, rtol=1e-3)) == n
    @test sum(isapprox.(x_agnostic, x_monotonic, atol=1e-6, rtol=1e-3)) == n
    @test sum(isapprox.(x_adaptive, x_agnostic, atol=1e-6, rtol=1e-3)) == n

    settings = Boscia.create_default_settings()
    settings.frank_wolfe[:line_search] = line_search
    settings.branch_and_bound[:verbose] = false
    settings.branch_and_bound[:print_iter] = 1
    settings.branch_and_bound[:node_limit] = 2
    x_monotonic, _, result_monotonic_node_limit = Boscia.solve(f, grad!, lmo, settings=settings)

    @test length(result_monotonic_node_limit[:list_ub]) <= 3
    @test result_monotonic_node_limit[:status] == Boscia.NODE_LIMIT_REACHED
end

n = 20
diffi = rand(rng, Bool, n) * 0.6 .+ 0.3

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
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    x_lazy, _, result_lazy = Boscia.solve(f, grad!, lmo, settings=settings)

    lmo = build_model()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:lazy] = false
    x_no, _, result_no = Boscia.solve(f, grad!, lmo, settings=settings)

    lmo = build_model()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:lazy] = true
    settings.frank_wolfe[:lazy_tolerance] = 1.5
    x_mid, _, result_mid = Boscia.solve(f, grad!, lmo, settings=settings)

    @test isapprox(f(x_lazy), f(result_lazy[:raw_solution]), atol=1e-6, rtol=1e-2)
    @test isapprox(f(x_no), f(result_no[:raw_solution]), atol=1e-6, rtol=1e-2)
    @test isapprox(f(x_mid), f(result_mid[:raw_solution]), atol=1e-6, rtol=1e-2)
    @test sum(isapprox.(x_lazy, x_no, atol=1e-6, rtol=1e-2)) == n
    @test sum(isapprox.(x_lazy, x_mid, atol=1e-6, rtol=1e-2)) == n
end

@testset "DICG - Lazification" begin

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

    # testing for weak lazification
    lmo = build_model()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()
    x_lazy, _, result_lazy = Boscia.solve(f, grad!, lmo, settings=settings)

    lmo = build_model()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:lazy] = false
    settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()
    x_no, _, result_no = Boscia.solve(f, grad!, lmo, settings=settings)

    lmo = build_model()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:lazy] = true
    settings.frank_wolfe[:lazy_tolerance] = 1.5
    settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()
    x_warm_start, _, result_warm_start = Boscia.solve(f, grad!, lmo, settings=settings)

    @test isapprox(f(x_lazy), f(result_lazy[:raw_solution]), atol=1e-6, rtol=1e-2)
    @test isapprox(f(x_no), f(result_no[:raw_solution]), atol=1e-6, rtol=1e-2)
    @test isapprox(f(x_warm_start), f(result_warm_start[:raw_solution]), atol=1e-6, rtol=1e-2)
    @test sum(isapprox.(x_lazy, x_no, atol=1e-6, rtol=1e-2)) == n
    @test sum(isapprox.(x_lazy, x_warm_start, atol=1e-6, rtol=1e-2)) == n

    # testing for strong lazification
    lmo = build_model()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:variant] =
        Boscia.DecompositionInvariantConditionalGradient(use_strong_lazy=true)
    x_lazy, _, result_lazy = Boscia.solve(f, grad!, lmo, settings=settings)

    lmo = build_model()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:lazy] = false
    settings.frank_wolfe[:variant] =
        Boscia.DecompositionInvariantConditionalGradient(use_strong_lazy=true)
    x_no, _, result_no = Boscia.solve(f, grad!, lmo, settings=settings)

    lmo = build_model()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:lazy] = true
    settings.frank_wolfe[:lazy_tolerance] = 1.5
    settings.frank_wolfe[:variant] =
        Boscia.DecompositionInvariantConditionalGradient(use_strong_lazy=true)
    x_warm_start, _, result_warm_start = Boscia.solve(f, grad!, lmo, settings=settings)

    @test isapprox(f(x_lazy), f(result_lazy[:raw_solution]), atol=1e-6, rtol=1e-2)
    @test isapprox(f(x_no), f(result_no[:raw_solution]), atol=1e-6, rtol=1e-2)
    @test isapprox(f(x_warm_start), f(result_warm_start[:raw_solution]), atol=1e-6, rtol=1e-2)
    @test sum(isapprox.(x_lazy, x_no, atol=1e-6, rtol=1e-2)) == n
    @test sum(isapprox.(x_lazy, x_warm_start, atol=1e-6, rtol=1e-2)) == n
end

@testset "DICG - warm_start" begin

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
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()
    x_no, _, result_no = Boscia.solve(f, grad!, lmo, settings=settings)

    # testing for weak warm-start
    lmo = build_model()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:variant] =
        Boscia.DecompositionInvariantConditionalGradient(use_DICG_warm_start=true)
    x_weak_warm_start, _, result_weak_warm_start = Boscia.solve(f, grad!, lmo, settings=settings)

    # testing for strong warm_start
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient(
        use_DICG_warm_start=true,
        use_strong_warm_start=true,
    )
    x_strong_warm_start, _, result_strong_warm_start =
        Boscia.solve(f, grad!, lmo, settings=settings)

    @test isapprox(f(x_no), f(result_no[:raw_solution]), atol=1e-6, rtol=1e-2)
    @test isapprox(
        f(x_weak_warm_start),
        f(result_weak_warm_start[:raw_solution]),
        atol=1e-6,
        rtol=1e-2,
    )
    @test isapprox(
        f(x_strong_warm_start),
        f(result_strong_warm_start[:raw_solution]),
        atol=1e-6,
        rtol=1e-2,
    )
    @test sum(isapprox.(x_no, x_weak_warm_start, atol=1e-6, rtol=1e-2)) == n
    @test sum(isapprox.(x_no, x_strong_warm_start, atol=1e-6, rtol=1e-2)) == n
end

@testset "User stop" begin
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

    function callback(
        tree,
        node;
        worse_than_incumbent=false,
        node_infeasible=false,
        lb_update=false,
    )
        if node.id == 2
            tree.root.problem.solving_stage = Boscia.USER_STOP
        end
    end

    lmo = build_model()
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    settings.branch_and_bound[:bnb_callback] = callback
    settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()
    x_no, _, result_no = Boscia.solve(f, grad!, lmo, settings=settings)

    @test result_no[:status] == Boscia.USER_STOP
end

@testset "Linear feasible" begin
    n = 10
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
    end
    MOI.add_constraint(o, 1.0x[1] + 1.0x[2], MOI.LessThan(1.5))
    @test Boscia.is_linear_feasible(o, 2 * ones(n)) == false
    @test Boscia.is_linear_feasible(o, vcat([1.0, 0.5], ones(n - 2)))
    @test Boscia.is_linear_feasible(o, vcat([0.5, 0.5], ones(n - 2)))
    @test Boscia.is_linear_feasible(o, vcat([0.0, 0.0], ones(n - 2)))
end

@testset "Float N test with ProbabilitySimplexSimpleBLMO" begin
    n = 10
    N = 24.5   #Float N
    d = randn(n)
    nint = 4

    lb = zeros(nint)
    ub = ones(nint) * 20.0

    int_vars = collect(1:nint)

    blmo = Boscia.ProbabilitySimplexSimpleBLMO(N)

    x_feas = [1.0, 2.0, 0.0, 2.0, 0.0, 0.0, 4.2, 1.5, 4.1, 9.7] #exactly equal to N


    Q = Matrix(I, n, n)
    b = -Q * x_feas

    function f(x)
        return 0.5 * x' * Q * x + b' * x
    end

    function grad!(storage, x)
        return storage .= Q * x + b
    end

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:time_limit] = 10.0
    settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()

    x, tlmo, result = Boscia.solve(f, grad!, blmo, lb, ub, int_vars, n; settings=settings)


    @test length(x) == n
    @test isfinite(f(x))
    @test Boscia.is_simple_linear_feasible(blmo, x)


    println("Solution x = ", x)
    println("Objective f(x) = ", f(x))
    println("Status = ", result[:status])
end

@testset "Float N test with UnitSimplexSimpleBLMO" begin
    n = 10
    N = 22.4   #Float N
    d = randn(n)
    nint = 3

    lb = zeros(nint)
    ub = ones(nint) * 10.0

    int_vars = collect(1:nint)

    blmo = Boscia.UnitSimplexSimpleBLMO(N)

    x_feas = [1.0, 2.0, 0.0, 2.0, 0.0, 0.0, 4.2, 1.5, 4.1, 0.6] #smaller than N

    Q = Matrix(I, n, n)
    b = -Q * x_feas

    function f(x)
        return 0.5 * x' * Q * x + b' * x
    end

    function grad!(storage, x)
        return storage .= Q * x + b
    end

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:time_limit] = 10.0
    settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()

    x, tlmo, result = Boscia.solve(f, grad!, blmo, lb, ub, int_vars, n; settings=settings)

    @test length(x) == n
    @test isfinite(f(x))
    @test Boscia.is_simple_linear_feasible(blmo, x_feas)

    println("Solution x = ", x)
    println("Objective f(x) = ", f(x))
    println("Status = ", result[:status])
end
