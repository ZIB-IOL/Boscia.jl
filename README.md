# BranchWolfe.jl

[![Build Status](https://github.com/ZIB-IOL/FrankWolfe.jl/workflows/CI/badge.svg)](https://github.com/ZIB-IOL/BranchWolfe.jl/actions)
[![Coverage](https://codecov.io/gh/ZIB-IOL/FrankWolfe.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ZIB-IOL/BranchWolfe.jl)

A package for Branch-and-Bound on top of Frank-Wolfe methods.

## Overview

The BranchWolfe.jl combines (a variant of) the Frank-Wolfe algorithm with a branch-and-bound like algorithms to solve mixed-integer convex optimization problems of the form `min_{x ∈ C, x_I ∈ Z^n} f(x)`, where `f` is a differentiable convex function, `C` is a convex and compact set, and `I` is a set of indices of integral variables.
They are especially useful when we have a method to optimize a linear function over `C` and the integrality constraints in a compuationally efficient way. 

A paper presenting the package with mathematical explanations and numerous examples can be found here:

> [xxx](xxx).

`BranchWolfe.jl` uses [`FrankWolfe.jl`](https://github.com/ZIB-IOL/FrankWolfe.jl) for solving the convex subproblems and [`Bonobo.jl`](https://github.com/Wikunia/Bonobo.jl) for managing the search tree.

## Installation

The most recent release is available via the julia package manager, e.g., with

```julia
using Pkg
Pkg.add("BranchWolfe")
```

or the master branch:

```julia
Pkg.add(url="https://github.com/ZIB-IOL/BranchWolfe.jl", rev="master")
```

## Getting started

Here is a simple example to get started. For more examples see the examples folder in the package.

```julia
julia> using BranchWolfe

julia> using FrankWolfe

julia> using Random

julia> using SCIP

julia> using LinearAlgebra

julia> import MathOptInterface

julia> const MOI = MathOptInterface
MathOptInterface

julia> n = 6
6

julia> const diffw = 0.5 * ones(n)
WARNING: redefinition of constant diffw. This may fail, cause incorrect answers, or produce other errors.
6-element Vector{Float64}:
 0.5
 0.5
 0.5
 0.5
 0.5
 0.5

julia> o = SCIP.Optimizer()
SCIP.Optimizer

julia> MOI.set(o, MOI.Silent(), true)

julia> MOI.empty!(o)

julia> x = MOI.add_variables(o, n)
6-element Vector{MathOptInterface.VariableIndex}:
 MathOptInterface.VariableIndex(1)
 MathOptInterface.VariableIndex(2)
 MathOptInterface.VariableIndex(3)
 MathOptInterface.VariableIndex(4)
 MathOptInterface.VariableIndex(5)
 MathOptInterface.VariableIndex(6)

julia> for xi in x
           MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
           MOI.add_constraint(o, xi, MOI.LessThan(1.0))
           MOI.add_constraint(o, xi, MOI.ZeroOne())
       end

julia> lmo = FrankWolfe.MathOptLMO(o)
FrankWolfe.MathOptLMO{SCIP.Optimizer}(SCIP.Optimizer, true)

julia> function f(x)
           return sum(0.5*(x.-diffw).^2)
       end
f (generic function with 1 method)

julia> function grad!(storage, x)
           @. storage = x-diffw
       end
grad! (generic function with 1 method)

julia> x, _, result,_ = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = true)

BranchWolfe Algorithm.

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

