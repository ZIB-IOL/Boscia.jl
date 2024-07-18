<a name="readme-top"></a>

<h3 align="center">Experiments</h3>

This README summarizes the experiments presented in the manuscript "Convex mixed-integer optimization with Frank-Wolfe methods".

Assuming the `julia` binary is available, first run:
```shell
julia --project "using Pkg; Pkg.update()"
```
in the `Boscia` folder to download and set up all dependencies.

To run some examples, run `test` in the REPL: 
```shell
julia --project
] test
```
The examples tested are all included in the test folder, see `runtests.jl`.

### Pipeline

|  Example | to start sbatch pipeline | configure solver setup in |
|---|---|---| 
| mixed/integer portfolio problems | run_portfolio_slurm.jl | run_portfolio_setup.jl |
| Poisson regression | run_poisson_slurm.jl  | run_poisson_setup.jl |
| sparse regression | run_sparse_reg_slurm.jl | run_sparse_reg_setup.jl |
| sparse log regression |run_sparse_log_reg_slurm.jl | run_sparse_log_reg_setup.jl |
| tailed cardinality | run_tailed_cardinality_sparse_reg_slurm.jl | run_tailed_cardinality_sparse_reg_setup.jl |
| tailed cardinality sparse log regression | run_tailed_cardinality_sparse_log_reg_slurm.jl| run_tailed_cardinality_sparse_log_reg_setup.jl |
| miplib instances |run_miplib_slurm.jl | run_miplib_setup.jl |

To create all csv files, run: 
```julia
julia --project run_portfolio_slurm.jl
```
We are using `julia 1.10.2` for the experiments. You have to set the path to the julia location on the cluster in all `batch*` files.

To run one instance:
```julia
julia --project 
include("portfolio.jl")
portfolio_boscia(1, 20, true; bo_mode="default",mode="mixed")
```

We generate one CSV per instance. Merge instance CSV by running `*_merge_csvs.jl`. The CSV file will be saved in `final_csvs`. The following will create a CSV file for each experiment for the default Boscia setup:
```julia 
julia --project boscia_merge_csvs.jl
```
To generate the summary csvs, run `compile_csv.jl`. This generates summary files for each problem, one comparing the different solver setups, one comparing the different settings in Boscia.jl, and the last comparing the branching strategies. It creates both `*_non_grouped.csv` files where all the csv are just merged into one, and `*_mode_summary_by_difficulty.csv` where the instances are sorted by difficulty (i.e. minimum time taken to solve it). To generate all files call:
```julia 
julia --project compile_csv.jl
```

| Example | Boscia (default) | Pavito |  SHOT | Ipopt | SCIP (OA) | AFW | Boscia variants<span style="color:red">*</span> | Strong Branching |
|---|---|---|---|---|---|---|---|---|
| mixed/integer portfolio problems | <span style="color:red">some mixed instances errored</span> | <span style="color:red">1 mixed instance missing </span> | done | <span style="color:red">some integer instances errored</span>| <span style="color:red">some integer and mixed instance missing</span> | <span style="color:red">missing</span> | <span style="color:red">missing</span> |
| Poisson regression | done | done | done | done | <span style="color:red">some instances missing</span> | <span style="color:red">missing</span> | <span style="color:red">missing</span> |
| sparse regression | <span style="color:red">some instances errored</span> | done | done | done | done | <span style="color:red">missing</span> | <span style="color:red">missing</span> |
| sparse log regression | done | done | done |done | done | <span style="color:red">missing</span> | <span style="color:red">missing</span> |
| tailed cardinality sparse regression<span style="color:purple">*</span> | <span style="color:red">missing</span> | - | - | - | done | <span style="color:red">missing</span> | <span style="color:red">missing</span> |
| tailed cardinality sparse log regression<span style="color:purple">*</span> | <span style="color:red">missing</span> | - | - | - | done | <span style="color:red">missing</span> | <span style="color:red">missing</span> |
| miplib instances | done | done | <span style="color:red">some pg5_34 instances are missing</span> | done | done | <span style="color:red">missing</span> | <span style="color:red">missing</span> |

<span style="color:purple">* only needed for Boscia and SCIP+OA due to indicator constraints</span>
<span style="color:red">* For the Boscia variants, we have to enable `no warm start` on branch.</span>

### Tables
|   Table| file to generate grouped CSV |
|---|---|
| 1-4 | <span style="color:red">missing</span> | 
| 5-6 | <span style="color:red">missing</span>  | 
| 7-8 | <span style="color:red">missing</span> | 
| 9-10 | <span style="color:red">missing</span> | 
| 11-12 | <span style="color:red">missing</span> | 
| 13-14 | <span style="color:red">missing</span> |
| 15-16 | <span style="color:red">missing</span> | 
| 23-26 | ?|

### Images
#### Paper:

| Figure | create figure with| build required CSV with |
|---|---|---|
| 4 | plot_boscia_vs_ipopt.jl |
| 5 | plot_dual_gap.jl | sparse_reg_boscia(2, 10, true, bo_mode="default")
| 6 | plot_dual_gap_strong_branching.jl | 
| 7 | plot_boscia_vs_pavito.jl | -
| 8-9 | plot_tightenings.jl | 
| 10 | plot_boscia_vs_pavito.jl | -

To compare Boscia with Ipopt:
```julia
julia --project
include("mip-examples.jl")
include("plot_boscia_vs_ipopt.jl")

mip_lib(1,4,true, example="22433", bo_mode="boscia")
mip_lib_ipopt(1,4,true, example="22433")
plot_boscia_vs_ipopt("22433")
```

To create a convergence plot:
```julia
julia --project
include("sparse_reg.jl")
include("plot_dual_gap.jl")

sparse_reg(1,20,1,true;bo_mode="boscia")
file_name = "csv/boscia_sparse_reg_20_1.csv"
dual_gap_plot(file_name)
```

To compare different tightening strategies:
```julia
julia --project
include("sparse_reg.jl")
include("plot_tightenings.jl")

sparse_reg(5,20,1,true,bo_mode="boscia")
sparse_reg(5,20,1,true,bo_mode="no_tightening")
sparse_reg(5,20,1,true,bo_mode="local_tightening")
sparse_reg(5,20,1,true,bo_mode="global_tightening")
plot("sparse_reg","20_5")
```

#### Appendix: 

| Figure | create figure with | build required CSV with |
|---|---|---|
| 11 | plot_per_layer.jl | birkhoff_data() |
| 12 | plot_grid_search_sparse_reg.jl | sparse_reg_grid_search() |
| 13 | plot_dual_gap.jl |  |
| 14-15 | plot_dual_gap_strong_branching.jl |  | 
| 16 | plot_dual_gap_hybrid_branching.jl |  |
| 17 | plot_per_layer.jl |  |
| 18 | plot_dual_gap_non_accum.jl |  |
| 19 | plot_dual_gap.jl |  |
| 20-26 | plot_boscia_vs_scip.jl | - |
| 27-30 | plot_boscia_vs_ipopt.jl |  |
| 31-34 | run_bigM_vs_indicator.jl | |
| 35 | plot_tightenings.jl |  |
| 36-38 | plot_tightenings.jl |  |
| 39-40 | plot_tightenings.jl |  |
 
   <!-- ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
 -->







