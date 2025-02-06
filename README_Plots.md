<a name="readme-top"></a>

<h3 align="center">Plots</h3>

This README summarizes the generating of the plots in the manuscript "Convex mixed-integer optimization with Frank-Wolfe methods".

### Paper:

| Figure | create figure with| build required CSV with |
|---|---|---|
| 4a | plot_dual_gap.jl |  poisson_reg_boscia(1, 70, 1.0, true)
| 4b | plot_dual_gap.jl | sparse_log_reg_boscia(7, 15, 1, 15, 1.0, true)  
| 5a | plot_boscia_vs_ipopt.jl | miplib_boscia("neos5", 3, 5; full_callback=true) -  miplib_boscia("neos5", 3, 5; full_callback=true, bo_mode="strong_convexity")
| 5b | plot_boscia_vs_ipopt.jl | miplib_boscia("neos5", 3, 8; full_callback=true) -  miplib_boscia("neos5", 3, 8; full_callback=true, bo_mode="strong_convexity")
| 6 | plot_boscia_vs_pavito.jl | run_portfolio_slurm.jl for Boscia.jl, SCIP-OA, BnB Ipopt and Pavito.jl
| 7a | plot_boscia_vs_ipopt.jl | miplib_boscia("neos5", 1, 6; full_callback=true) -  miplib_ipopt("neos5", 1, 6; full_callback=true) 
| 7b | plot_boscia_vs_ipopt.jl | miplib_boscia("neos5", 3, 8; full_callback=true) - miplib_ipopt("neos5", 3, 8; full_callback=true) 
| 8 | plot_dual_gap.jl | portfolio_boscia(10, 35, "integer"; full_callback=true) 
| 9 | plot_boscia_vs_pavito.jl | run_poisson_slurm.jl for Boscia.jl, SCIP-OA, BnB Ipopt and Pavito.jl
| 10 | plot_boscia_vs_pavito.jl | run_sparse_reg_slurm.jl for Boscia.jl, SCIP-OA, BnB Ipopt and Pavito.jl
| 11 | plot_boscia_vs_scip.jl | run_portfolio_slurm.jl for the different Boscia Settings
| 12a | plot_tightening.jl | sparse_reg_boscia(5, 23: full_callback=true, bo_mode=mode) with mode in ["default", "global_tightening", "local_tightening", "no_tightening"] 
| 12b | plot_tightening.jl | portfolio_boscia(1, 120, "integer": full_callback=true, bo_mode=mode) with mode in ["default", "global_tightening", "local_tightening", "no_tightening"] 
| 13 | plot_dual_gap_strong_branching.jl | portfolio_boscia(1, 50, "mixed", full_callback=true, bo_mode=mode) with mode=["default, "strong_branching", "hybrid_branching_20"]


To compare Boscia with Ipopt:
```julia
julia --project
include("mps-examples.jl")
include("plot_boscia_vs_ipopt.jl")

miplib_boscia("neos5", 3, 5)
miplib_ipopt("neos5", 3, 5)
plot_boscia_vs_ipopt("neos5", 3, 5)
```

To compare Boscia with and without strong convexity:
```julia
julia --project
include("mps-examples.jl")
include("plot_boscia_vs_ipopt.jl")

miplib_boscia("neos5", 3, 5)
miplib_boscia("neos5", 3, 5; bo_mode="strong_convexity")
plot_boscia_vs_strong_convexity("neos5", 3, 5)
```

To create a convergence plot:
```julia
julia --project
include("sparse_reg.jl")
include("plot_dual_gap.jl")

sparse_reg_boscia(5, 23; full_callback=true, bo_mode=mode)

file_name = "csv/boscia_" * mode * "_sparse_reg_20_1.csv"
dual_gap_plot(file_name, mode)
```

To compare different tightening strategies:
```julia
julia --project
include("sparse_reg.jl")
include("plot_tightenings.jl")

sparse_reg_boscia(5 ,20 , 1; full_callback=true, bo_mode="default")
sparse_reg_boscia(5, 20, 1; full_callback=true, bo_mode="no_tightening")
sparse_reg_boscia(5, 20, 1; full_callback=true, bo_mode="local_tightening")
sparse_reg_boscia(5, 20, 1; full_callback=true, bo_mode="global_tightening")
plot("sparse_reg","20_5")
```

To compare the different branching strategies:
```julia
julia --project
include("portfolio.jl")
include("plot_dual_gap_strong_branching.jl")

portfolio_boscia(1, 50, "mixed"; full_callback=true, bo_mode="default")
portfolio_boscia(1, 50, "mixed"; full_callback=true, bo_mode="strong_branching")
portfolio_boscia(1, 50, "mixed"; full_callback=true, bo_mode="hybrid_branching_20")
dual_gap_plot("mixed_portfolio", 1, 50, "nodes")
dual_gap_plot("mixed_portfolio", 1, 50, "time")
```

To compare the number of terminations:
Run the corresponding slurm file, `run_*example*_slurm.jl`.
```julia
julia --project run_sparse_reg_slurm.jl
```

```julia
julia --project
include("plot_boscia_vs_pavito.jl")
include("plot_boscia_vs_scip.jl")

include("boscia_merge_csvs.jl")
merge_csvs("sparse_reg")
include("ipopt_merge_csvs.jl")
merge_csvs("sparse_reg")
include("pavito_merge_csvs.jl")
merge_csvs("sparse_reg")
include("shot_merge_csvs.jl")
merge_csvs("sparse_reg")
include("scip_merge_csvs.jl")
merge_csvs("sparse_reg")

include("compile_csv.jl")

# comparison
build_non_grouped_csv("comparison", example="sparse_reg")
build_summary_by_difficulty("comparison", example="sparse_reg")

# settings 
build_non_grouped_csv("settings", example="sparse_reg")
build_summary_by_difficulty("settings", example="sparse_reg")

# branching
build_non_grouped_csv("branching", example="sparse_reg")
build_summary_by_difficulty("branching", example="sparse_reg")

plot_boscia_vs_pavito("sparse_reg", use_shot=false)
plot_boscia_vs_scip("sparse_reg")
```

### Appendix: 

| Figure | create figure with | build required CSV with |
|---|---|---|
| 14-22 | plot_boscia_vs_pavito.jl - plot_boscia_vs_scip.jl | run all slurm files with the different solvers |
| (23-27)a | plot_dual_gap.jl | sparse_reg_boscia(3, 19, full_callback=true) - portfolio_boscia(10, 35, "integer", full_callback=true) - portfolio_boscia(1, 45, "mixed", full_callback=true) - poisson_boscia(1, 70, 1.0, full_callback=true) - sparse_log_reg_boscia(7, 15, 1, 15, 1.0, full_callback=true) |
| (23-27)b | plot_dual_gap_non_accum.jl | sparse_reg_boscia(7, 15, full_callback=true) - portfolio_boscia(9, 20, "integer", full_callback=true) - portfolio_boscia(4, 55, "mixed", full_callback=true) - poisson_boscia(1, 70, 10.0, full_callback=true) - sparse_log_reg_boscia(9, 5, 5, 5, 0.1, full_callback=true)  |
| 28-30 | plot_per_layer.jl | sparse_reg_portfolio(4, 15, full_callback=true) - portfolio_boscia(8, 30, "mixed", full_callback=true) - portfolio_boscia(6, 20, "integer", full_callback=true) | 
| 31 | plot_tightenings.jl | portfolio_boscia(10, 75, "mixed", full_callback=true, bo_mode=mode) with mode in ["default", "global_tightening", "local_tightening", "no_tightening"] 
| 32 | plot_dual_decay.jl | run_sparse_reg_slurm.jl with the different combinations of dual decay factors and start Frank-Wolfe epsilon 
| 33-34 | plot_dual_gap_strong_branching.jl | sparse_reg_boscia(9, 22, full_callback=true, bo_mode=mode) - portfolio(2, 120, "integer", full_callback=true. bo_mode=mode) with mode=["default, "strong_branching", "hybrid_branching_20"]

 To create a convergence plot with non accumulated BLMO calls:
```julia
julia --project
include("sparse_reg.jl")
include("plot_dual_gap_non_accum.jl")

sparse_reg_boscia(7, 15; full_callback=true, bo_mode=mode)

file_name = "csv/boscia_" * mode * "_sparse_reg_7_15.csv"
dual_gap_plot(file_name, mode)
```

 To create the comparison of dual decay:
```julia
julia --project
include("sparse_reg.jl")
include("plot_dual_decay.jl")
seed = 1
dim = 27

dual_gap_decay_factors = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
epsilons = [1e-2, 1e-3, 5e-3, 1e-4]
depth = 1

for factor in dual_gap_decay_factors
    for epsilon in epsilons
        @show seed, dimension
            run(`sbatch batch_sparse_reg.sh $seed $dimension $mode $depth $factor $epsilon`)
          end 
      end
end

plot_dual_decay(seed, dim)
```
   <!-- ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
 -->