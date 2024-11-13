using Boscia
using Test
using Random
using LinearAlgebra
using FrankWolfe

## Log barrier
# min_x - ∑ log(xi + ϵ) - log(N - ∑ xi + ϵ)
# s.t.  x ∈ {0,1}^n
# Strong convexity: μ = 1 / (1 + ϵ)^2n
# Sharpness: M = sqrt(2/μ), θ = 1/2 

## General convex quadratic
# min_x 1/2 x' * Q * x + b' * x 
# s.t.  ∑ x = N
#       x ∈ {0,1}^n
# Strong convexity: μ = minimum(eigvals(Q))
# Sharpness: M = sqrt(2/μ), θ = 1/2 


seed = 0x5526f8e0e9a68f36
Random.seed!(seed)

@testset "Strong convexity" begin

    @testset "Log barrier" begin
        n = 50
        N = Int(floor(3/4 * n))
        ϵ = 1e-3

        function f(x)
            return - sum(log(xi + ϵ) for xi in x) - log(N - sum(x) + ϵ) 
        end

        function grad!(storage, x)
            storage .= - 1 ./ (x .+ ϵ) .- 1/sum(N - sum(x) + ϵ) 
            return storage

        end

        int_vars = collect(1:n)
        sblmo = Boscia.UnitSimplexSimpleBLMO(N)
        line_search = FrankWolfe.Adaptive()

        x, _, result = Boscia.solve(
            f,
            grad!,
            sblmo,
            fill(0.0, n),
            fill(floor(N/2), n),
            int_vars,
            n,
            verbose=true,
            line_search=line_search,
            time_limit=120,
            print_iter=1000,
        )

        μ = 1/(1 + ϵ)^(2*n)
        x_sc, _, result_sc = Boscia.solve(
            f,
            grad!,
            sblmo,
            fill(0.0, n),
            fill(floor(N/2), n),
            int_vars,
            n,
            verbose=true,
            line_search=line_search,
            strong_convexity=μ,
            time_limit=120,
            print_iter=1000,
        )

        @test f(x_sc) <= f(x) + 1e-6
        @test result_sc[:dual_bound] > result[:dual_bound]
    end

    @testset "General convex quadratic" begin
        n = 20
        N = Int(floor(n/2))
        Q = rand(n, n)
        Q = Q' * Q 
        @assert isposdef(Q)

        b = rand(n)

        function f(x)
            return 1/2 * x' * Q * x - b' * x
        end

        function grad!(storage, x)
            storage .= Q * x - b 
            return storage
        end

        val, sol = Boscia.min_via_enum_prob_simplex(f, n, N)

        blmo = Boscia.ProbabilitySimplexSimpleBLMO(N)
        μ = minimum(eigvals(Q))

        x, _, _ = Boscia.solve(
            f,
            grad!,
            blmo,
            fill(0.0, n),
            fill(1.0, n),
            collect(1:n),
            n,
            strong_convexity=μ,
            verbose=true,
            fw_epsilon=1e-3,
        )

        @test isapprox(f(x), f(sol), atol=1e-5, rtol=1e-2)
    end     
end

@testset "Sharpness" begin

    @testset "Log barrier" begin 
        n = 50
        N = Int(floor(3/4 * n))
        ϵ = 1e-3

        function f(x)
            return - sum(log(xi + ϵ) for xi in x) - log(N - sum(x) + ϵ) 
        end

        function grad!(storage, x)
            storage .= - 1 ./ (x .+ ϵ) .- 1/sum(N - sum(x) + ϵ) 
            return storage

        end

        int_vars = collect(1:n)
        sblmo = Boscia.UnitSimplexSimpleBLMO(N)
        line_search = FrankWolfe.Adaptive()

        x, _, result = Boscia.solve(
            f,
            grad!,
            sblmo,
            fill(0.0, n),
            fill(floor(N/2), n),
            int_vars,
            n,
            verbose=true,
            line_search=line_search,
            time_limit=120,
            print_iter=1000,
        )

        μ = 1/(1 + ϵ)^(2*n)
        θ = 1/2
        M = sqrt(2/μ)
        x_sc, _, result_sc = Boscia.solve(
            f,
            grad!,
            sblmo,
            fill(0.0, n),
            fill(floor(N/2), n),
            int_vars,
            n,
            verbose=true,
            line_search=line_search,
            sharpness_constant=M,
            sharpness_exponent=θ,
            time_limit=120,
            print_iter=1000,
        )

        @test f(x_sc) <= f(x) + 1e-6
        @test result_sc[:dual_bound] >= result[:dual_bound]
    end

    @testset "General convex quadratic" begin
        n = 20
        N = Int(floor(n/2))
        Q = rand(n, n)
        Q = Q' * Q 
        @assert isposdef(Q)

        b = rand(n)

        function f(x)
            return 1/2 * x' * Q * x - b' * x
        end

        function grad!(storage, x)
            storage .= Q * x - b 
            return storage
        end
        val, sol = Boscia.min_via_enum_prob_simplex(f, n, N)

        blmo = Boscia.ProbabilitySimplexSimpleBLMO(N)
        μ = minimum(eigvals(Q))
        θ = 1/2
        M = sqrt(2/μ)

        x, _, _ = Boscia.solve(
            f,
            grad!,
            blmo,
            fill(0.0, n),
            fill(1.0, n),
            collect(1:n),
            n,
            sharpness_constant=M,
            sharpness_exponent=θ,
            verbose=true,
        )

        @test isapprox(f(x), f(sol), atol=1e-5, rtol=1e-2)
    end
end