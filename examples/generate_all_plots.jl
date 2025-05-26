# Script to generate all plots from the paper and appendix
# Make sure to run this from the project directory with: julia --project generate_all_plots.jl

# Include necessary plotting scripts
include("colours.jl")
include("plot_dual_gap.jl")
include("plot_dual_gap_non_accum.jl")
include("plot_boscia_vs_ipopt.jl")
include("plot_boscia_vs_pavito.jl")
include("plot_boscia_vs_scip.jl")
include("plot_tightenings.jl")
include("plot_dual_gap_strong_branching.jl")
include("plot_per_layer.jl")
include("plot_dual_decay.jl")

println("Generating paper figures...")

# Paper Figures
println("Generating Figure 4...")
mode = "default"
# Figure 4a - Poisson regression dual gap
dimension = 70
seed = 1
ns = 1.0
file = joinpath(@__DIR__, "csv/" * mode * "_" * "poisson_" * string(dimension) * "_" * string(ns) * "-" * string(dimension) * "_" * string(floor(dimension/2)) * "_" * string(seed) * ".csv")
dual_gap_plot(file, "default")

# Figure 4b - Sparse log regression dual gap
dimension = 15
seed = 7
var_A = 1
M = 1.0
file = joinpath(@__DIR__, "csv/" * mode * "_" * "sparse_log_regression_" * string(dimension) * "_" * string(M) * "-" * string(var_A) * "_" * string(seed) * ".csv")
dual_gap_plot(file, "default")

println("Generating Figure 5...")
# Figure 5a - Boscia vs Strong Convexity (neos5, 3, 5)
plot_boscia_vs_strong_convexity("neos5", seed = 3, num_v = 5)

# Figure 5b - Boscia vs Strong Convexity (neos5, 3, 8)
plot_boscia_vs_strong_convexity("neos5", seed = 3, num_v = 8)

println("Generating Figure 6...")
# Figure 6 - Portfolio comparison
plot_boscia_vs_pavito("portfolio_mixed", use_shot=true)
plot_boscia_vs_scip("portfolio_mixed")

println("Generating Figure 7...")
# Figure 7a - Boscia vs Ipopt (neos5, 1, 6)
plot_boscia_vs_ipopt("neos5", seed = 1, num_v = 6)

# Figure 7b - Boscia vs Ipopt (neos5, 3, 8)
plot_boscia_vs_ipopt("neos5", seed = 3, num_v = 8)

println("Generating Figure 8...")
# Figure 8 - Portfolio dual gap
file_name = joinpath(@__DIR__, "csv/default_35_10_integer_portfolio.csv")
dual_gap_plot(file_name, "default")

println("Generating Figure 9...")
# Figure 9 - Poisson regression comparison
plot_boscia_vs_pavito("poisson_reg", use_shot=true)
plot_boscia_vs_scip("poisson_reg")

println("Generating Figure 10...")
# Figure 10 - Sparse regression comparison
plot_boscia_vs_pavito("sparse_reg", use_shot=true)
plot_boscia_vs_scip("sparse_reg")

println("Generating Figure 11...")
# Figure 11 - Portfolio Boscia settings
plot_boscia_vs_scip("portfolio_integer")

println("Generating Figure 12...")
# Figure 12a - Sparse regression tightening
plot("sparse_reg", "23_5")

# Figure 12b - Portfolio tightening
plot("portfolio_integer", "120_1")

println("Generating Figure 13...")
# Figure 13 - Branching strategies
dual_gap_plot("mixed_portfolio", 1, 50, "nodes")
dual_gap_plot("mixed_portfolio", 1, 50, "time")

println("\nGenerating appendix figures...")

# Appendix Figures
println("Generating Figures 14-22...")
# Figures 14-22 - Comparison plots for all examples
for example in ["portfolio_integer", "portfolio_mixed", "poisson_reg", "sparse_reg", "sparse_log_reg", 
                "tailed_cardinality", "tailed_cardinality_sparse_log_reg", "miplib_22433", "miplib_neos5"]
    println("Generating comparison plots for $example...")
    plot_boscia_vs_pavito(example, use_shot=true)
    plot_boscia_vs_scip(example)
end

println("Generating Figures 23-27...")
# Figures 23-27a - Dual gap plots
mode = "default"

file = joinpath(@__DIR__, "csv/boscia_" * mode * "_" * string(19) * "_" * string(3) * "_sparse_reg.csv")
dual_gap_plot(file, mode)

file = joinpath(@__DIR__, "csv/" * mode * "_" * string(45) * "_" * string(1) * "_mixed_portfolio.csv")
dual_gap_plot(file, mode)

file = joinpath(@__DIR__, "csv/" * mode * "_" * string(35) * "_" * string(10) * "_integer_portfolio.csv")
dual_gap_plot(file, mode)

var_A = 1
M = 1.0
file = joinpath(@__DIR__, "csv/" * mode * "_" * "sparse_log_regression_" * string(15) * "_" * string(M) * "-" * string(var_A) * "_" * string(7) * ".csv")
dual_gap_plot(file, mode)

ns = 1.0
file = joinpath(@__DIR__, "csv/" * mode * "_" * "poisson_" * string(70) * "_" * string(ns) * "-" * string(70) * "_" * string(floor(70/2)) * "_" * string(1) * ".csv")
dual_gap_plot(file, mode)



# Figures 23-27b - Non-accumulated dual gap plots
file = joinpath(@__DIR__, "csv/boscia_" * mode * "_" * string(15) * "_" * string(7) * "_sparse_reg.csv")
plot_progress_lmo(file, mode)

file = joinpath(@__DIR__, "csv/" * mode * "_" * string(55) * "_" * string(4) * "_mixed_portfolio.csv")
plot_progress_lmo(file, mode)

file = joinpath(@__DIR__, "csv/" * mode * "_" * string(20) * "_" * string(9) * "_integer_portfolio.csv")
plot_progress_lmo(file, mode)

var_A = 5
M = 0.1
file = joinpath(@__DIR__, "csv/" * mode * "_" * "sparse_log_regression_" * string(5) * "_" * string(M) * "-" * string(var_A) * "_" * string(9) * ".csv")
plot_progress_lmo(file, mode)

ns = 10.0
file = joinpath(@__DIR__, "csv/" * mode * "_" * "poisson_" * string(70) * "_" * string(ns) * "-" * string(70) * "_" * string(floor(70/2)) * "_" * string(1) * ".csv")
plot_progress_lmo(file, mode)


println("Generating Figures 28-30...")
file = joinpath(@__DIR__, "csv/boscia_" * mode * "_" * string(15) * "_" * string(4) * "_sparse_reg.csv")
per_layer_plot(file, mode)


file = joinpath(@__DIR__, "csv/" * mode * "_" * string(30) * "_" * string(8) * "_mixed_portfolio.csv")
per_layer_plot(file, mode)


file = joinpath(@__DIR__, "csv/" * mode * "_" * string(20) * "_" * string(6) * "_integer_portfolio.csv")
per_layer_plot(file, mode)


println("Generating Figure 31...")
# Figure 31 - Portfolio tightening comparison
plot("mixed_portfolio", "75_10")

println("Generating Figure 32...")
# Figure 32 - Dual decay comparison
plot_dual_decay(27, 1)  # Using the example parameters from the README

println("Generating Figures 33-34...")
# Figures 33-34 - Strong branching comparison
for (example, seed, dim) in [
    ("sparse_reg", 9, 22),
    ("integer_portfolio", 2, 120)
]
    dual_gap_plot("$(example)", seed, dim, "nodes")
    dual_gap_plot("$(example)", seed, dim, "time")
end

println("\nAll plots have been generated!") 