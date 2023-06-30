<a name="readme-top"></a>

<h3 align="center">Experiments</h3>

This README summarizes the experiments presented in the manuscript "Convex mixed-integer optimization with Frank-Wolfe methods".

<!-- ABOUT THE PROJECT -->
### About The Project

<!-- ### Examples
- portfolio -->

### Tables
|   Table| File  |
|---|---|
| 1-4 | run_portfolio.jl |
| 5-6 | run_poisson.jl  |
| 7-8 | run_sparse_reg.jl |
| 9-10 |run_sparse_log_reg.jl |
| 11-12 |run_tailed_cardinality.jl |
| 13-14 |run_tailed_cardinality_sparse_log_reg.jl |
| 15-16 |run_mip_lib.jl |
| 23-26 |run_bigM_vs_indicator.jl |

To create all csv files required for a table, run: 

```julia
julia --project run_portfolio.jl
```

To run one instance:
```julia
julia --project 
include("portfolio.jl")
portfolio(1, 20, true; bo_mode="boscia",mode="mixed")
```

### Images
#### Paper:

| Figure | File | build required CSV |
|---|---|---|
| 4 |plot_boscia_vs_ipopt.jl | mip-examples.jl
| 5 |plot_dual_gap.jl | sparse_reg.jl, birkhoff.jl |
| 6 |plot_dual_gap_strong_branching.jl
| 7 |plot_boscia_vs_scip.jl | run_sparse_log_reg.jl, run_portfolio.jl, build_csv.jl | run_sparse_reg.jl, run_portfolio.jl
| 8-9 |plot_tightenings.jl | run_portfolio.jl, run_sparse_reg.jl
| 10 |plot_boscia_vs_scip.jl |run_portfolio.jl, build_csv.jl |
  
#### Appendix: 

| Figure | File | CSV |
|---|---|---|
| 11 | plot_per_layer.jl | birkhoff.jl |
| 12 | plot_grid_search_sparse_reg.jl |
| 13 | plot_dual_gap.jl | run_sparse_reg.jl |
| 14-15 | plot_dual_gap_strong_branching.jl |
| 16 | plot_dual_gap_hybrid_branching.jl |
| 17 | plot_per_layer.jl | birkhoff.jl ??? |
| 18 | plot_dual_gap_non_accum.jl | birkhoff.jl |
| 19 | ?
| 20-26 | plot_boscia_vs_scip.jl | run_portfolio.jl, run_poisson.jl, run_sparse_reg.jl, run_sparse_log_reg.jl, run_tailed_cardinality.jl, run_tailed_cardinality_sparse_log_reg.jl, build_csv.jl |
| 27-30 | plot_boscia_vs_ipopt.jl | mip-examples.jl |
| 31-34 | run_bigM_vs_indicator.jl | run_bigM_vs_indicator.jl
| 35 | ? | mip-lib.jl |
| 36-38 | plot_tightening.jl | run_sparse_reg.jl, run_portfolio.jl |
| 39-40 | plot_tightening.jl | run_portfolio.jl |
 
   <!-- ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
 -->







