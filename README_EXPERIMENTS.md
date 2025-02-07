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

### Reproduce the instances

Each solver and problem combination has its own function which takes the seed and the dimension (and potential other input).
This builds the problem in the format the solver requires, solves it and writes the solution and statistics into a CSV file.

Example code for all problems can be found in `reproduce_instances.jl`.

#### MIPLIB
The MPS files can be found in *examples/mps-examples/mps-files*. 
The code can be found in `examples/mps-example.jl`.
To build an instance, one has to state the instance name, e.g. `neos5`, and state a number of vertices. 
In our experiments, we used 4 to 8 as the number of vertices from which the objective is build, seeds used were `1,2,3`.

#### Portfolio
The code for this example can be found in `examples/portfolio.jl`.
The dimensions used are `collect(20:5:120)` and seeds from 1 to 10.
All functions have a parameter `mode` which expects a string and which decides whether the mixed (`"mixed"`) or pure integer (`"integer"`) problem build.

#### Poisson Regression
The code can be found in `examples/poisson_reg.jl`.
The dimensions used are `collect(50:20:100)`, seeds from 1 to 10. 
For the big-M constraints, we chosen the values 0.1, 1, 5 and 10. 

#### Sparse Regression, Sparse Log Regression, Tailed Sparse Regression and Tailed Log Sparse Regression 
The code can be found in `examples/sparse_reg.jl`, `examples/sparse_log_reg.jl`, `examples/tailed_cardinality.jl` and `examples/tailed_cardinality_sparse_log_reg.jl`, respectively.
The dimensions used for both Sparse Regression and Tailed Sparse regression are `collect(15:30)` and seeds go from 1 to 10.
For Sparse Log Regression and Tailed Sparse Log Regression, the dimensions are `collect(5:5:20)`, seeds go from 1 to 10.
For the big-M constraints, we have used 0.1 and 1. 
Input scaling factors were 1 and 5.
