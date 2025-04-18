# The Bounded Linear Minimization Oracle (BLMO)

The Bounded Linear Minimization Oracle (BLMO) is an oracle solving the problem
```math
\begin{aligned}
v \in \arg\min & \; \langle x, d \rangle \\
\text{s.t} & \; x\in C \\
& \; x_I \in \mathbb{Z}^{|I|} \cap [l,u]   %\text{ where } x \in X \subset \mathbb{R}, \, x_I \in \mathbb{Z}^{|I|}
\end{aligned}
```
where the direction $d$ is usually the gradient evaluated at a certain point.
The bounds are specified at the node level and correspond to bounds obtained by branching.

## General BLMO Interface 

In the following, the functions for the general BLMO interface are listed.
Functions without signature needs to be implemented by a new BLMO type.
Functions with signature are optional and are usually for statistics and additional safety checks.

```@autodocs
Modules = [Boscia]
Pages = ["blmo_interface.jl"]
```

## MathOptInterface (MOI) BLMO

With this BLMO type, any (MIP) solver that provides an interface to `MathOptInterface` and `JuMP` can be used in Boscia.
Note that we only require the feasible region, i.e. no objective has to be set.

```@autodocs
Modules = [Boscia]
Pages = ["MOI_bounded_oracle.jl"]
```

## Managed BLMO

Sometimes the linear problem over the feasible region can be computed via a combinatorial algorithm that is more efficient than formulating the problem as a MIP.
If one does not want to implement the BLMO interface from scratch which requires multiple methods to be provided, we provide the `ManagedBLMO`.
It handles the bound management, so that the user has to implement only a few methods.

```@autodocs
Modules = [Boscia]
Pages = ["managed_blmo.jl"]
```

## Polytopes

Here are some preimplemented polytopes.
We have the hypercube, the unit simplex and the probability simplex.

```@autodocs
Modules = [Boscia]
Pages = ["polytope_blmos.jl"]
```

## Time Tracking BLMO Wrapper 

This wrapper keeps track of the statistics like solving time of the BLMO, the number of calls etc.

```@autodocs
Modules = [Boscia]
Pages = ["time_tracking_lmo.jl"]
```

## Build LMO

Given the global bounds on the integer variables and the bounds at the node level, this builds the BLMO instance for the specific node.
This way, the BLMO can be stored in the tree as opposed to every node having a copy of it.

```@autodocs
Modules = [Boscia]
Pages = ["build_lmo.jl"]
```

## Integer Bounds 

The data structure that records the bounds on the integer/binary variables.

```@autodocs
Modules = [Boscia]
Pages = ["integer_bounds.jl"]
```
