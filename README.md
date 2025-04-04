# Boscia.jl

[![Build Status](https://github.com/ZIB-IOL/Boscia.jl/workflows/CI/badge.svg)](https://github.com/ZIB-IOL/Boscia.jl/actions)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://zib-iol.github.io/Boscia.jl/dev/)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://zib-iol.github.io/Boscia.jl/stable/)
[![Coverage](https://codecov.io/gh/ZIB-IOL/Boscia.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ZIB-IOL/Boscia.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12720675.svg)](https://doi.org/10.5281/zenodo.12720675)

A solver for Mixed-Integer Convex Optimization that uses Frank-Wolfe methods for convex relaxations and a branch-and-bound algorithm.

## Overview

The Boscia.jl solver combines (a variant of) the Frank-Wolfe algorithm with a branch-and-bound-like algorithm to solve mixed-integer convex optimization problems of the form
$\min_{x ∈ C, x_I ∈ \mathbb{Z}^n} f(x)$,
where $f$ is a differentiable convex function, $C$ is a convex and compact set, and $I$ is a set of indices of integral variables.

This approach is especially effective when we have a method to optimize a linear function over $C$ and the integrality constraints in a computationally efficient way.
The set `C` can modelled using the Julia package **MathOptInterface** (or **JuMP**). 
We also implemented simple polytopes like the hypercube, the unit simplex and the probability simplex. Also, we intend to extend this list by combinatorial polytopes, e.g. the matching polytope.

The paper presenting the package with mathematical explanations and numerous examples can be found here:

> Convex mixed-integer optimization with Frank-Wolfe methods: [2208.11010](https://arxiv.org/abs/2208.11010)

`Boscia.jl` uses [`FrankWolfe.jl`](https://github.com/ZIB-IOL/FrankWolfe.jl) for solving the convex subproblems, [`Bonobo.jl`](https://github.com/Wikunia/Bonobo.jl) for managing the search tree, and oracles optimizing linear functions over the feasible set, for instance calling [SCIP](https://scipopt.org) or any MOI-compatible solver to solve MIP subproblems.

## Installation

Add the Boscia stable release with:

```julia
import Pkg
Pkg.add("Boscia")
```

Or get the latest master branch with:
```julia
import Pkg
Pkg.add(url="https://github.com/ZIB-IOL/Boscia.jl", rev="main")
```

For the installation of `SCIP.jl`, see [here](https://github.com/scipopt/SCIP.jl).
Note, for Windows users, you do not need to download the SCIP binaries, you can also use the installer provided by SCIP.


## Getting started

Here is a simple example to get started. For more examples, see the examples folder in the package.


```julia
using Boscia
using FrankWolfe
using Random
using SCIP
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface

n = 6

const diffw = 0.5 * ones(n)
o = SCIP.Optimizer()

MOI.set(o, MOI.Silent(), true)

x = MOI.add_variables(o, n)

for xi in x
    MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
    MOI.add_constraint(o, xi, MOI.LessThan(1.0))
    MOI.add_constraint(o, xi, MOI.ZeroOne())
end

lmo = FrankWolfe.MathOptLMO(o)

function f(x)
    return sum(0.5*(x.-diffw).^2)
end

function grad!(storage, x)
    @. storage = x-diffw
end

x, _, result = Boscia.solve(f, grad!, lmo, verbose = true)

Boscia Algorithm.

Parameter settings.
	 Tree traversal strategy: Move best bound
	 Branching strategy: Most infeasible
	 Absolute dual gap tolerance: 1.000000e-06
	 Relative dual gap tolerance: 1.000000e-02
	 Frank-Wolfe subproblem tolerance: 1.000000e-05
	 Total number of varibales: 6
	 Number of integer variables: 0
	 Number of binary variables: 6
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   Iteration       Open          Bound      Incumbent      Gap (abs)      Gap (rel)       Time (s)      Nodes/sec        FW (ms)       LMO (ms)  LMO (calls c)   FW (Its)   #ActiveSet  Discarded
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*          1          2  -1.202020e-06   7.500000e-01   7.500012e-01            Inf   3.870000e-01   7.751938e+00            237              2              9         13            1          0
         100         27   6.249998e-01   7.500000e-01   1.250002e-01   2.000004e-01   5.590000e-01   2.271914e+02              0              0            641          0            1          0
         127          0   7.500000e-01   7.500000e-01   0.000000e+00   0.000000e+00   5.770000e-01   2.201040e+02              0              0            695          0            1          0
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Postprocessing

Blended Pairwise Conditional Gradient Algorithm.
MEMORY_MODE: FrankWolfe.InplaceEmphasis() STEPSIZE: Adaptive EPSILON: 1.0e-7 MAXITERATION: 10000 TYPE: Float64
GRADIENTTYPE: Nothing LAZY: true lazy_tolerance: 2.0
[ Info: In memory_mode memory iterates are written back into x0!

----------------------------------------------------------------------------------------------------------------
  Type     Iteration         Primal           Dual       Dual Gap           Time         It/sec     #ActiveSet
----------------------------------------------------------------------------------------------------------------
  Last             0   7.500000e-01   7.500000e-01   0.000000e+00   1.086583e-03   0.000000e+00              1
----------------------------------------------------------------------------------------------------------------
    PP             0   7.500000e-01   7.500000e-01   0.000000e+00   1.927792e-03   0.000000e+00              1
----------------------------------------------------------------------------------------------------------------

Solution Statistics.
	 Solution Status: Optimal (tree empty)
	 Primal Objective: 0.75
	 Dual Bound: 0.75
	 Dual Gap (relative): 0.0

Search Statistics.
	 Total number of nodes processed: 127
	 Total number of lmo calls: 699
	 Total time (s): 0.58
	 LMO calls / sec: 1205.1724137931035
	 Nodes / sec: 218.96551724137933
	 LMO calls / node: 5.503937007874016
```
