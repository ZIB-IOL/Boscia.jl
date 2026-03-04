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

@testset verbose = true failfast = true "Boscia Test Suite" begin

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
    include("traverse_strategy_test.jl")

    # Files to exclude from testing (e.g., utilities that require extra dependencies)
    excluded_files = ["plot_utilities.jl"]

    for file in readdir(joinpath(@__DIR__, "../examples/"), join=true)
        filename = basename(file)
        if endswith(file, "jl") && !(filename in excluded_files)
            # Isolate each example in its own module to avoid global name clashes
            m = Module()
            # Provide a local include that resolves relative to this module
            Core.eval(m, :(include(x) = Base.include(@__MODULE__, x)))
            Base.include(m, file)
        end
    end

    # Quality control of the code
    include("aqua.jl")
end
