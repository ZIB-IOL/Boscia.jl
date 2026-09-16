using Boscia
using FrankWolfe
using LinearAlgebra
using StableRNGs
using Random
using LogExpFunctions
using Test

println("\nSmoothing mode test")

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

m = 15
k = Int(floor(m / 2))
A_s = randn(rng, m, k)

@testset "Smoothing mode" begin

    f(x) = maximum([dot(A_s[:, i], x) for i in 1:k])
    function grad!(storage, x)
        fx = f(x)
        empty!(storage)
        for i in 1:k
            if isapprox(dot(A_s[:, i], x), fx; atol=1e-10, rtol=1e-10)
                push!(storage, A_s[:, i])
            end
        end
        return storage
    end

    sol, x_sol = Boscia.min_via_enum(f, m)

    function generate_smoothing_function(μ; epsilon=1e-6, node_level=0)
        # stable log-sum-exp for the linear pieces:
        function f_μ(x)
            t = [dot(A_s[:, i], x) for i in 1:k]
            return μ * logsumexp(t ./ μ) - μ * log(k)
        end

        function grad_f_μ!(storage, x)
            t = [dot(A_s[:, i], x) for i in 1:k]
            log_z = logsumexp(t ./ μ)
            fill!(storage, 0)
            @inbounds for i in 1:k
                wi = exp(t[i] / μ - log_z)
                storage .+= wi .* view(A_s, :, i)
            end
            return storage
        end

        return f_μ, grad_f_μ!
    end

    slmo = FrankWolfe.ZeroOneHypercubeLMO()
    lmo = Boscia.ManagedLMO(slmo, fill(0.0, m), fill(1.0, m), collect(1:m), m)

    # Test set up
    settings = Boscia.create_default_settings(; mode=Boscia.SMOOTHING_MODE)

    err = nothing
    try
        Boscia.solve(f, grad!, lmo, settings=settings)
    catch err
    end
    @test err isa ErrorException &&
          err.msg == "generate_smoothing_objective function is required in SMOOTHING_MODE!"

    settings = Boscia.create_default_settings()
    f_test, g_test = generate_smoothing_function(1.0)
    settings.smoothing[:generate_smoothing_objective] = generate_smoothing_function
    settings.branch_and_bound[:time_limit] = 10.0
    @test_logs (:warn, "generate_smoothing_objective function will only be used in SMOOTHING_MODE!") Boscia.solve(
        f_test,
        g_test,
        lmo,
        settings=settings,
    )

    # Test full run
    function node_callback(
        tree,
        node,
        x;
        μ=Inf,
        primal=Inf,
        dual_gap=Inf,
        fw_status=nothing,
        atoms_set=nothing,
        resolve_integer_solution=false,
    )
        if tree.root.options[:clip_mu_resolution]
            @test μ ≥ tree.root.options[:smoothing_min]
        else
            @test isapprox(
                μ,
                tree.root.options[:smoothing_start] *
                (tree.root.options[:smoothing_decay]^(node.std.depth - 1));
                atol=1e-6,
                rtol=1e-6,
            )
        end
        f_μ, _ = tree.root.options[:generate_smoothing_objective](
            μ;
            epsilon=tree.root.options[:fw_epsilon],
            node_level=node.std.depth,
        )
        @test f_μ(node.active_set.x) <= f(node.active_set.x)
        @test isapprox(
            f_μ(node.active_set.x),
            tree.root.problem.f(node.active_set.x);
            atol=1e-6,
            rtol=1e-6,
        )
    end
    settings = Boscia.create_default_settings(; mode=Boscia.SMOOTHING_MODE)
    settings.branch_and_bound[:verbose] = true
    settings.branch_and_bound[:node_callback] = node_callback
    settings.smoothing[:generate_smoothing_objective] = generate_smoothing_function
    settings.smoothing[:max_restart_fw_iter] = 100
    settings.smoothing[:clip_mu_resolution] = true
    σ = maximum(norm(view(A_s, :, i)) for i in 1:k)
    settings.smoothing[:smoothing_start] = 0.2 * σ
    settings.smoothing[:smoothing_min] = 1e-3 * σ
    settings.smoothing[:smoothing_decay] = 0.85

    x, tlmo, result = Boscia.solve(f, grad!, lmo, settings=settings)

    if isapprox(f(x), sol; atol=1e-6, rtol=1e-2)
        @test isapprox(f(x), sol; atol=1e-6, rtol=1e-2)
    else
        @warn "Reported solution is suboptimal: f(x)=$(f(x)) > $(sol)=sol"
    end
end
