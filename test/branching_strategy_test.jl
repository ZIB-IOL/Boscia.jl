using Boscia
using Bonobo
using FrankWolfe
using Test
using Random
using SCIP
using HiGHS
using LinearAlgebra
using StableRNGs

import MathOptInterface
const MOI = MathOptInterface

println("\nBranching Strategy Tests")
seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

verbose = true


function build_examples(o, n, seed)
    Random.seed!(seed)
    A = let
        A = randn(n, n)
        A' * A
    end

    @assert isposdef(A) == true

    y = Random.rand(Bool, n) * 0.6 .+ 0.3

    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.ZeroOne())
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
    end
    lmo = Boscia.MathOptBLMO(o)

    function f(x)
        d = x - y
        return dot(d, A, d)
    end

    function grad!(storage, x)
        # storage = Ax
        mul!(storage, A, x)
        # storage = 2Ax - 2Ay
        return mul!(storage, A, y, -2, 2)
    end
    return f, grad!, lmo
end


@testset "Simple Branching Strategies" begin
    dimension = 20
    seed = 1
    o = SCIP.Optimizer()
    f, grad!, lmo = build_examples(o, dimension, seed)
    time_limit = 60

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = verbose
    settings.branch_and_bound[:time_limit] = time_limit
    settings.branch_and_bound[:branching_strategy] = Bonobo.MOST_INFEASIBLE()
    x_mi, _, result_mi = Boscia.solve(f, grad!, lmo, settings=settings)

    @testset "Largest Gradient Branching" begin
        branching_strategy = Boscia.LargestGradient()
        o = SCIP.Optimizer()
        f, grad!, lmo = build_examples(o, dimension, seed)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = verbose
        settings.branch_and_bound[:time_limit] = time_limit
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test isapprox(f(x_mi), f(x), atol=1e-6, rtol=1e-3)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Largest Most-Infeasible Gradient Branching" begin
        branching_strategy = Boscia.LargestMostInfeasibleGradient()
        o = SCIP.Optimizer()
        f, grad!, lmo = build_examples(o, dimension, seed)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = verbose
        settings.branch_and_bound[:time_limit] = time_limit
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test isapprox(f(x_mi), f(x), atol=1e-6, rtol=1e-3)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Random Branching" begin
        branching_strategy = Boscia.RandomBranching()
        o = SCIP.Optimizer()
        f, grad!, lmo = build_examples(o, dimension, seed)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = verbose
        settings.branch_and_bound[:time_limit] = time_limit
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test isapprox(f(x_mi), f(x), atol=1e-6, rtol=1e-3)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Largest Index" begin
        branching_strategy = Boscia.LargestIndex()
        o = SCIP.Optimizer()
        f, grad!, lmo = build_examples(o, dimension, seed)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = verbose
        settings.branch_and_bound[:time_limit] = time_limit
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test isapprox(f(x_mi), f(x), atol=1e-6, rtol=1e-3)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end
end


@testset "Pseudocost Branching Strategies" begin
    dimension = 30
    seed = 1
    o = SCIP.Optimizer()
    f, grad!, lmo = build_examples(o, dimension, seed)
    time_limit = 60

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = verbose
    settings.branch_and_bound[:time_limit] = time_limit
    settings.branch_and_bound[:branching_strategy] = Bonobo.MOST_INFEASIBLE()
    x_mi, _, result_mi = Boscia.solve(f, grad!, lmo, settings=settings)

    @testset "Pseudocost with Most-Infeasible alternative and weighted_sum decision function" begin
        o = SCIP.Optimizer()
        f, grad!, lmo = build_examples(o, dimension, seed)

        branching_strategy = Boscia.PseudocostBranching(
            lmo;
            alt_f=Boscia.most_infeasible_decision,
            stable_f=Boscia.PseudocostStableSelectionGenerator("weighted_sum", 0.5),
            iterations_until_stable=1,
        )

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = verbose
        settings.branch_and_bound[:time_limit] = time_limit
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test isapprox(f(x_mi), f(x), atol=1e-6, rtol=1e-3)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Pseudocost with Largest Most-Infeasible Gradient alternative and product decision function" begin
        o = SCIP.Optimizer()
        f, grad!, lmo = build_examples(o, dimension, seed)

        branching_strategy = Boscia.PseudocostBranching(
            lmo;
            alt_f=Boscia.largest_most_infeasible_gradient_decision,
            stable_f=Boscia.PseudocostStableSelectionGenerator("product", 1e-6),
            iterations_until_stable=1,
        )

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = verbose
        settings.branch_and_bound[:time_limit] = time_limit
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test isapprox(f(x_mi), f(x), atol=1e-6, rtol=1e-3)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Pseudocost with LargestGradient alternative and minimum decision function" begin
        o = SCIP.Optimizer()
        f, grad!, lmo = build_examples(o, dimension, seed)

        branching_strategy = Boscia.PseudocostBranching(
            lmo;
            alt_f=Boscia.largest_gradient_decision,
            stable_f=Boscia.PseudocostStableSelectionGenerator("minimum", 1e-6),
            iterations_until_stable=1,
        )

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = verbose
        settings.branch_and_bound[:time_limit] = time_limit
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test isapprox(f(x_mi), f(x), atol=1e-6, rtol=1e-3)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Pseudocost with Most-Infeasible alternative and product decision function and 5 visit stability criterion" begin
        o = SCIP.Optimizer()
        f, grad!, lmo = build_examples(o, dimension, seed)

        branching_strategy = Boscia.PseudocostBranching(
            lmo;
            alt_f=Boscia.most_infeasible_decision,
            stable_f=Boscia.PseudocostStableSelectionGenerator("product", 1e-6),
            iterations_until_stable=5,
        )

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = verbose
        settings.branch_and_bound[:time_limit] = time_limit
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test isapprox(f(x_mi), f(x), atol=1e-6, rtol=1e-3)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end
end


function int_sparse_regression(o, n, m, l, k, seed)
    Random.seed!(seed)
    sol_x = rand(1:l, n)
    for _ in 1:(n-k)
        sol_x[rand(1:n)] = 0
    end
    D = rand(m, n)
    y_d = D * sol_x

    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    z = MOI.add_variables(o, n)
    for i in 1:n
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0 * l))
        MOI.add_constraint(o, x[i], MOI.Integer())

        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())

        MOI.add_constraint(o, 1.0 * x[i] - 1.0 * l * z[i], MOI.LessThan(0.0))
    end
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    # MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(zeros(n),x), sum(Float64.(iszero.(x)))), MOI.GreaterThan(1.0*(n-k)))
    # MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n),z), 0.0), MOI.GreaterThan(1.0*k))
    lmo = Boscia.MathOptBLMO(o)

    function f(x)
        xv = @view(x[1:n])
        return 1 / 2 * sum(abs2, y_d - D * xv)  #+ lambda_2*FrankWolfe.norm(x)^2 + lambda_0*sum(x[p+1:2p])
    end

    function grad!(storage, x)
        storage .= 0
        @view(storage[1:n]) .= transpose(D) * (D * @view(x[1:n]) - y_d)
        return storage
    end

    return lmo, f, grad!
end

@testset "Hierarchy Branching Strategies" begin
    dimension = 30
    seed = 1
    o = SCIP.Optimizer()
    n = dimension
    m = 3 * dimension
    l = ceil(dimension / 2)
    k = l - 1
    #lmo, f, grad! = int_sparse_regression(o, n, m, l, k, seed)

    f, grad!, lmo = build_examples(o, dimension, seed)
    time_limit = 60

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = verbose
    settings.branch_and_bound[:time_limit] = time_limit
    settings.branch_and_bound[:branching_strategy] = Bonobo.MOST_INFEASIBLE()
    x_mi, _, result_mi = Boscia.solve(f, grad!, lmo, settings=settings)

    @testset "Hierarchy with Most-Infeasible pseudocost alternative and weighted_sum decision function" begin
        o = SCIP.Optimizer()
        f, grad!, lmo = build_examples(o, dimension, seed)

        stages = Boscia.default_hierarchy_strategies(
            "most_infeasible",
            "most_infeasible",
            1,
            "weighted_sum",
        )
        branching_strategy = Boscia.Hierarchy(lmo; stages)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = verbose
        settings.branch_and_bound[:time_limit] = time_limit
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test isapprox(f(x_mi), f(x), atol=1e-6, rtol=1e-3)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Hierarchy Branching with Largest Gradient pseudocost alternative and product decision function" begin
        o = SCIP.Optimizer()
        f, grad!, lmo = build_examples(o, dimension, seed)

        stages =
            Boscia.default_hierarchy_strategies("most_infeasible", "largest_gradient", 1, "product")
        branching_strategy = Boscia.Hierarchy(lmo; stages)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = verbose
        settings.branch_and_bound[:time_limit] = time_limit
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test isapprox(f(x_mi), f(x), atol=1e-6, rtol=1e-3)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Hierarchy with Largest Most-Infeasible Gradient pseudocost alternative and minimum decision function" begin
        o = SCIP.Optimizer()
        f, grad!, lmo = build_examples(o, dimension, seed)

        stages = Boscia.default_hierarchy_strategies(
            "most_infeasible",
            "largest_most_infeasible_gradient",
            1,
            "minimum",
        )
        branching_strategy = Boscia.Hierarchy(lmo; stages)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = verbose
        settings.branch_and_bound[:time_limit] = time_limit
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test isapprox(f(x_mi), f(x), atol=1e-6, rtol=1e-3)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end

    @testset "Hierarchy with Most-Infeasible pseudocost alternative and product decision function + branch binary first " begin
        o = SCIP.Optimizer()
        f, grad!, lmo = build_examples(o, dimension, seed)

        stages =
            Boscia.default_hierarchy_strategies("most_infeasible", "most_infeasible", 1, "product")
        stages = insert!(stages, 1, Boscia.create_binary_stage(lmo))

        branching_strategy = Boscia.Hierarchy(lmo; stages)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = verbose
        settings.branch_and_bound[:time_limit] = time_limit
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test isapprox(f(x_mi), f(x), atol=1e-6, rtol=1e-3)
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)
    end
end

n = 10
const diff1 = rand(rng, Bool, n) * 0.8 .+ 1.1
@testset "Strong branching" begin
    function f(x)
        return sum((x .- diff1) .^ 2)
    end
    function grad!(storage, x)
        @. storage = 2 * (x - diff1)
    end

    lb = min(sum(round.(diff1)), sum(diff1)) - 0.1
    ub = max(sum(round.(diff1)), sum(diff1)) + 0.1
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(4.0))
        MOI.add_constraint(o, xi, MOI.Integer()) # or MOI.Integer()
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.LessThan(ub),
    )
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.GreaterThan(lb),
    )
    lmo = FrankWolfe.MathOptLMO(o)

    blmo = Boscia.MathOptBLMO(HiGHS.Optimizer())
    branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
    MOI.set(branching_strategy.lmo.o, MOI.Silent(), true)

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = true
    settings.branch_and_bound[:branching_strategy] = branching_strategy
    x, _, result_strong_branching = Boscia.solve(f, grad!, lmo, settings=settings)

    @test isapprox(x, round.(diff1), atol=1e-5, rtol=1e-5)
end

@testset "Hybrid branching" begin
    function f(x)
        return sum((x .- diff1) .^ 2)
    end
    function grad!(storage, x)
        @. storage = 2 * (x - diff1)
    end

    lb = min(sum(round.(diff1)), sum(diff1)) - 0.1
    ub = max(sum(round.(diff1)), sum(diff1)) + 0.1
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(4.0))
        MOI.add_constraint(o, xi, MOI.Integer()) # or MOI.Integer()
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.LessThan(ub),
    )
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.GreaterThan(lb),
    )
    lmo = FrankWolfe.MathOptLMO(o)


    function perform_strong_branch(tree, node)
        return node.level <= length(tree.root.problem.integer_variables) / 3
    end
    blmo = Boscia.MathOptBLMO(HiGHS.Optimizer())
    branching_strategy = Boscia.HybridStrongBranching(10, 1e-3, blmo, perform_strong_branch)
    MOI.set(branching_strategy.pstrong.lmo.o, MOI.Silent(), true)

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = true
    settings.branch_and_bound[:branching_strategy] = branching_strategy
    x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

    @test isapprox(x, round.(diff1), atol=1e-5, rtol=1e-5)
end
