# Script to generate all plots from the paper and appendix
# Make sure to run this from the project directory with: julia --project generate_all_plots.jl

# Include necessary plotting scripts
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
# Figure 4a - Poisson regression dual gap
file_name = "csv/boscia_default_poisson_reg_1_70.csv"
dual_gap_plot(file_name, "default")

# Figure 4b - Sparse log regression dual gap
file_name = "csv/boscia_default_sparse_log_reg_7_15.csv"
dual_gap_plot(file_name, "default")

println("Generating Figure 5...")
# Figure 5a - Boscia vs Strong Convexity (neos5, 3, 5)
plot_boscia_vs_strong_convexity("neos5", 3, 5)

# Figure 5b - Boscia vs Strong Convexity (neos5, 3, 8)
plot_boscia_vs_strong_convexity("neos5", 3, 8)

println("Generating Figure 6...")
# Figure 6 - Portfolio comparison
plot_boscia_vs_pavito("portfolio", use_shot=false)
plot_boscia_vs_scip("portfolio")

println("Generating Figure 7...")
# Figure 7a - Boscia vs Ipopt (neos5, 1, 6)
plot_boscia_vs_ipopt("neos5", 1, 6)

# Figure 7b - Boscia vs Ipopt (neos5, 3, 8)
plot_boscia_vs_ipopt("neos5", 3, 8)

println("Generating Figure 8...")
# Figure 8 - Portfolio dual gap
file_name = "csv/boscia_default_portfolio_10_35.csv"
dual_gap_plot(file_name, "default")

println("Generating Figure 9...")
# Figure 9 - Poisson regression comparison
plot_boscia_vs_pavito("poisson_reg", use_shot=false)
plot_boscia_vs_scip("poisson_reg")

println("Generating Figure 10...")
# Figure 10 - Sparse regression comparison
plot_boscia_vs_pavito("sparse_reg", use_shot=false)
plot_boscia_vs_scip("sparse_reg")

println("Generating Figure 11...")
# Figure 11 - Portfolio Boscia settings
plot_boscia_vs_scip("portfolio")

println("Generating Figure 12...")
# Figure 12a - Sparse regression tightening
plot("sparse_reg", "23_5")

# Figure 12b - Portfolio tightening
plot("portfolio", "120_1")

println("Generating Figure 13...")
# Figure 13 - Branching strategies
dual_gap_plot("mixed_portfolio", 1, 50, "nodes")
dual_gap_plot("mixed_portfolio", 1, 50, "time")

println("\nGenerating appendix figures...")

# Appendix Figures
println("Generating Figures 14-22...")
# Figures 14-22 - Comparison plots for all examples
for example in ["portfolio", "poisson_reg", "sparse_reg", "sparse_log_reg", 
                "tailed_cardinality_sparse_reg", "tailed_cardinality_sparse_log_reg", "miplib"]
    println("Generating comparison plots for $example...")
    plot_boscia_vs_pavito(example, use_shot=false)
    plot_boscia_vs_scip(example)
end

println("Generating Figures 23-27...")
# Figures 23-27a - Dual gap plots
for (example, seed, dim) in [
    ("sparse_reg", 3, 19),
    ("portfolio", 10, 35),
    ("portfolio", 1, 45),
    ("poisson_reg", 1, 70),
    ("sparse_log_reg", 7, 15)
]
    file_name = "csv/boscia_default_$(example)_$(seed)_$(dim).csv"
    dual_gap_plot(file_name, "default")
end

# Figures 23-27b - Non-accumulated dual gap plots
for (example, seed, dim) in [
    ("sparse_reg", 7, 15),
    ("portfolio", 9, 20),
    ("portfolio", 4, 55),
    ("poisson_reg", 1, 70),
    ("sparse_log_reg", 9, 5)
]
    file_name = "csv/boscia_default_$(example)_$(seed)_$(dim).csv"
    dual_gap_plot(file_name, "default")
end

println("Generating Figures 28-30...")
# Figures 28-30 - Per layer plots
for (example, seed, dim) in [
    ("sparse_reg", 4, 15),
    ("portfolio", 8, 30),
    ("portfolio", 6, 20)
]
    # Note: You may need to adjust the plot_per_layer function call based on its actual implementation
    plot_per_layer(example, seed, dim)
end

println("Generating Figure 31...")
# Figure 31 - Portfolio tightening comparison
plot("portfolio", "75_10")

println("Generating Figure 32...")
# Figure 32 - Dual decay comparison
plot_dual_decay(1, 27)  # Using the example parameters from the README

println("Generating Figures 33-34...")
# Figures 33-34 - Strong branching comparison
for (example, seed, dim) in [
    ("sparse_reg", 9, 22),
    ("portfolio", 2, 120)
]
    dual_gap_plot("$(example)", seed, dim, "nodes")
    dual_gap_plot("$(example)", seed, dim, "time")
end

println("\nAll plots have been generated!") 