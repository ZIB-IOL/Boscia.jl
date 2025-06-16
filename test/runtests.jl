using Boscia
using Test
using FrankWolfe
using Random
using StableRNGs
using SCIP
using Statistics
import Bonobo
import MathOptInterface
const MOI = MathOptInterface
import HiGHS
using Dates
using Aqua

@testset verbose=true failfast=true "Boscia Test Suite" begin

    include("interface_test.jl")
    include("LMO_test.jl")
    include("indicator_test.jl")
    include("heuristics.jl")

    # Takes pretty long, only include if you want to test this specifically
    #include("infeasible_pairwise.jl")

    include("sparse_regression.jl")
    include("poisson.jl")
    include("mean_risk.jl")
    include("time_limit.jl")
    include("strong_convexity_and_sharpness.jl")
    include("branching_strategy_test.jl")

    for file in readdir(joinpath(@__DIR__, "../examples/"), join=true)
        if endswith(file, "jl")
            include(file)
        end
    end

    # Quality control of the code
    Aqua.test_all(Boscia)
end