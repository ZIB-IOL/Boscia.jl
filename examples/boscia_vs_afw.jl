using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface
using CSV
using DataFrames

function boscia_vs_afw(seed=1, dimension=5, iter=3)

    Random.seed!(seed)
    n = dimension
    ri = rand(n)
    ai = rand(n)
    Ωi = rand()
    bi = sum(ai)
    Ai = randn(n, n)
    Ai = Ai' * Ai
    Mi = (Ai + Ai') / 2
    @assert isposdef(Mi)

    # integer set
    #I = 1:(n÷2)
    I = collect(1:n)
    
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    limit = 1800
    # MOI.set(o, MOI.TimeLimitSec(), limit)
    x = MOI.add_variables(o, n)
     
    for i in 1:n
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        if i in I
            MOI.add_constraint(o, x[i], MOI.Integer())
        end
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai, x), 0.0),
        MOI.LessThan(bi),
    )
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.GreaterThan(1.0),
    )
    lmo = FrankWolfe.MathOptLMO(o)
    # println("BOSCIA MODEL")
    # print(o)

    function f(x)
        return 1 / 2 * Ωi * dot(x, Mi, x) - dot(ri, x)
    end
    function grad!(storage, x)
        mul!(storage, Mi, x, Ωi, 0)
        storage .-= ri
        return storage
    end

    intial_status = String(string(MOI.get(o, MOI.TerminationStatus())))
    time_afw = -Inf

    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
    
    for i in 1:iter
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, afw=true)
        @show x, f(x)
        @test dot(ai, x) <= bi + 1e-6
        @test f(x) <= f(result[:raw_solution]) + 1e-6
        time_afw=result[:total_time_in_sec]
        status = result[:status]
        if result[:status] == "Optimal (tolerance reached)"
            status = "OPTIMAL"
        end
        df = DataFrame(seed=seed, dimension=n, time_afw=time_afw, solution_afw=result[:primal_objective], termination_afw=status)
        file_name = joinpath(@__DIR__,"afw_mixed_50.csv")
        if !isfile(file_name)
            CSV.write(file_name, df, append=true, writeheader=true)
        else 
            CSV.write(file_name, df, append=true)
        end
        # display(df)
    end
end 
