# Boscia.jl

[![Build Status](https://github.com/ZIB-IOL/Boscia.jl/workflows/CI/badge.svg)](https://github.com/ZIB-IOL/Boscia.jl/actions)
[![Coverage](https://codecov.io/gh/ZIB-IOL/Boscia.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ZIB-IOL/Boscia.jl)
[![Genie Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/Boscia)](https://pkgs.genieframework.com?packages=Boscia)

A package for Branch-and-Bound on top of Frank-Wolfe methods.

## Overview

The Boscia.jl combines (a variant of) the Frank-Wolfe algorithm with a branch-and-bound like algorithms to solve mixed-integer convex optimization problems of the form `min_{x ∈ C, x_I ∈ Z^n} f(x)`, where `f` is a differentiable convex function, `C` is a convex and compact set, and `I` is a set of indices of integral variables.
They are especially useful when we have a method to optimize a linear function over `C` and the integrality constraints in a compuationally efficient way.
`C` is specified using the MathOptInterface API or any DSL like JuMP implementing it.

A paper presenting the package with mathematical explanations and numerous examples can be found here:

> Convex integer optimization with Frank-Wolfe methods: [2208.11010](https://arxiv.org/abs/2208.11010)

`Boscia.jl` uses [`FrankWolfe.jl`](https://github.com/ZIB-IOL/FrankWolfe.jl) for solving the convex subproblems, [`Bonobo.jl`](https://github.com/Wikunia/Bonobo.jl) for managing the search tree, and [SCIP](https://scipopt.org) for the MIP subproblems.

## Installation

Add Boscia in its current state with:
```julia
Pkg.add(url="https://github.com/ZIB-IOL/Boscia.jl", rev="main")
```

## Getting started

Here is a simple example to get started. For more examples see the examples folder in the package.

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

julia> x, _, result,_ = Boscia.solve(f, grad!, lmo, verbose = true)

Boscia Algorithm.

Parameter settings.
	 Tree traversal strategy: Best-first search
	 Branching strategy: Most-infeasible
	 Absolute dual gap tolerance: 1.0e-7
	 Frank-Wolfe subproblem tolerance: 1.0e-5


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 Iteration       Open          Bound      Incumbent      Gap (abs)      Gap (rel)       Time (s)      Nodes/sec        FW (ms)       LMO (ms)  LMO (calls c)   FW (Its) #ActiveSet  Discarded
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
         1          2  -1.332268e-15   7.500000e-01   7.500000e-01            Inf   2.890000e-01   1.038062e+01            158              1              4          3          1          0
       100         27   6.250000e-01   7.500000e-01   1.250000e-01   2.000000e+01   3.830000e-01   3.315927e+02              1              0            326          1          1          0
       127          0   7.500000e-01   7.500000e-01   0.000000e+00   0.000000e+00   4.090000e-01   3.105134e+02              1              0            380          1          1          0
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Solution Statistics.
	 Solution Status: Optimal (tree empty)
	 Primal Objective: 0.75
	 Dual Bound: 0.75
	 Dual Gap (relative in %): 0.0

Search Statistics.
	 Total number of nodes processed: 127
	 Total number of lmo calls: 380
	 Total time (s): 0.409
	 LMO calls / sec: 929.0953545232275
	 Nodes / sec: 310.51344743276286
	 LMO calls / node: 2.9921259842519685
```
