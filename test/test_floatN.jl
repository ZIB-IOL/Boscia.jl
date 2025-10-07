using Test
using Random
using LinearAlgebra
using Boscia
using FrankWolfe
# using SCIP
import MathOptInterface
const MOI = MathOptInterface
using ForwardDiff

@testset "Float N test with ProbabilitySimplexSimpleBLMO" begin
    n = 10
    N = 24.5   #Float N
    d = randn(n)
    nint=9

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
        storage .= ForwardDiff.gradient(f, x)
    end

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:time_limit] = 10.0

    x, tlmo, result = Boscia.solve(
        f,
        grad!,
        blmo,
        lb,
        ub,
        int_vars,
        n;
        settings=settings
    )


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
    nint=3

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
        storage .= ForwardDiff.gradient(f, x)
    end

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:time_limit] = 10.0

    x, tlmo, result = Boscia.solve(
        f,
        grad!,
        blmo,
        lb,
        ub,
        int_vars,
        n;
        settings=settings
    )

    @test length(x) == n
    @test isfinite(f(x))
    @test Boscia.is_simple_linear_feasible(blmo, x_feas)

    println("Solution x = ", x)
    println("Objective f(x) = ", f(x))
    println("Status = ", result[:status])
end