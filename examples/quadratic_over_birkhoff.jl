using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
import MathOptInterface
using CombinatorialLinearOracles
const CLO = CombinatorialLinearOracles
const MOI = MathOptInterface
import HiGHS

println("\nQuadratic over Birkhoff Example")

# min_{X} 1/2 * || X - Xhat ||_F^2
# X ∈ P_n (permutation matrix)

n = 3

function build_objective(n, append_by_column=true)
    # generate random doubly stochastic matrix
    Xstar = rand(n, n)
    while norm(sum(Xstar, dims=1) .- 1) > 1e-6 || norm(sum(Xstar, dims=2) .- 1) > 1e-6
        Xstar ./= sum(Xstar, dims=1)
        Xstar ./= sum(Xstar, dims=2)
    end

    function f(x)
        X = append_by_column ? reshape(x, (n,n)) : transpose(reshape(x, (n,n)))
        return 1/2 * LinearAlgebra.tr(LinearAlgebra.transpose(X .- Xstar)*(X .- Xstar))
    end

    function grad!(storage, x)
        X = append_by_column ? reshape(x, (n,n)) : transpose(reshape(x, (n,n)))
        storage .= if append_by_column
            reduce(vcat, X .- Xstar)
        else
            reduce(vcat, LinearAlgebra.transpose(X .- Xstar))
        end
        #storage .= X .- Xstar
        return storage
    end

    return f, grad!
end


function build_birkhoff_mip(n)
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    X = reshape(MOI.add_variables(o, n^2), n, n)

    MOI.add_constraint.(o, X, MOI.ZeroOne())
    # doubly stochastic constraints
    MOI.add_constraint.(
        o,
        vec(sum(X, dims=1, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
        MOI.EqualTo(1.0),
    )
    MOI.add_constraint.(
        o,
        vec(sum(X, dims=2, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
        MOI.EqualTo(1.0),
    )
    return FrankWolfe.MathOptLMO(o)
end

 @testset "Birkhoff" begin
    f, grad! = build_objective(n)

    x = zeros(n, n)
    int_vars = collect(1:n^2)
    @testset "Birkhoff BLMO (BPCG)" begin
        sblmo = CLO.BirkhoffLMO(n, collect(1:n^2))

        lower_bounds = fill(0.0, n^2)
        upper_bounds = fill(1.0, n^2)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = true
        x, _, result = Boscia.solve(f, grad!, sblmo, lower_bounds, upper_bounds, int_vars, n^2, settings=settings)
        @test f(x) <= f(result[:raw_solution]) + 1e-6
        @test Boscia.is_simple_linear_feasible(sblmo, x)
    end

    x_dicg = zeros(n,n)
    @testset "Birkhoff BLMO (DICG)" begin
        sblmo = CLO.BirkhoffLMO(n, collect(1:n^2))

        lower_bounds = fill(0.0, n^2)
        upper_bounds = fill(1.0, n^2)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = true
        settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()
        x_dicg, _, result_dicg = Boscia.solve(f, grad!, sblmo, lower_bounds, upper_bounds, int_vars, n^2, settings=settings)
        @test f(x_dicg) <= f(result_dicg[:raw_solution]) + 1e-6
        @test Boscia.is_simple_linear_feasible(sblmo, x_dicg)
    end

    x_mip = zeros(n,n)
    @testset "MIP BLMO" begin
        lmo = build_birkhoff_mip(n)

        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = true
        x_mip, _, result_mip = Boscia.solve(f, grad!, lmo, settings=settings)
        @test f(x_mip) <= f(result_mip[:raw_solution]) + 1e-6
        @test Boscia.is_linear_feasible(lmo, x_mip)
    end 

    @show x
    @show x_mip
    @show x_dicg
    @show f(x), f(x_mip), f(x_dicg)
    @test isapprox(f(x_mip), f(x), atol=1e-6, rtol=1e-2)
    @test isapprox(f(x_dicg), f(x), atol=1e-6, rtol=1e-2)
end 