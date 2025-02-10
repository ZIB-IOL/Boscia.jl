# How does it work?

`Boscia.jl` is aimed at mixed-integer convex problems where the nonlinearity stems mostly from the objective function:

```math
\begin{aligned}
\min & \;  f(x)  \\
\text{s.t.} & \; x\in C, \, x_I \in \mathbb{Z}^{|I|}
\end{aligned} %\text{ where } x \in X \subset \mathbb{R}, \, x_I \in \mathbb{Z}^{|I|}
```

where $C$ is a compact, nonconvex set admitting a boundable linear minimization oracle (BLMO), i.e., a set over which optimizing a linear function can be done efficiently (comparatively to the original problem), even when bounds are added or modified. 
Taking lower bounds $l$ and upper bounds $u$, the oracle solves

```math
\begin{aligned}
v \in \arg\min & \; \langle x, d \rangle \\
\text{s.t} & \; x\in C \\
& \; x_I \in \mathbb{Z}^{|I|} \cap [l,u]   %\text{ where } x \in X \subset \mathbb{R}, \, x_I \in \mathbb{Z}^{|I|}
\end{aligned}
```
where $d$ will usually be the gradient of $f$ evaluated at a given point $x_t$, $\nabla f(x_t)$.

The new algorithmic framework is a branch-and-bound approach utilizing Frank-Wolfe (FW), also called Conditional Gradient (CG), methods as the node solver.
A new aspect is that we solve the continuous sub problems in the nodes over the integer hull, i.e. the convex hull of the integer feasible points.
Additionally, we exploit both general properties of the FW methods as well as the recent developments in the field of FW methods to speed up the solution process. 



## Frank-Wolfe variants

The Frank-Wolfe algorithms used in `Boscia.jl` are implemented in [FrankWolfe.jl](https://github.com/ZIB-IOL/FrankWolfe.jl). 
The variants currently available in `Boscia.jl` are Vanilla Frank-Wolfe, Away-Frank-Wolfe (AFW), Blended Conditional Gradient (BCG)
and Blended Pairwise Conditional Gradient (BPCG).
The latter is set as the default variant.


## Branch-and-Bound techniques 

In this section, we present the techniques derived from Frank Wolfe that can be used in our framework .

### Dual gap based termination 

Frank-Wolfe methods produce primal feasible iterates and an FW gap, offering many inexpensive iterations with a gradually increasing dual bound. 
This allows early termination of nodes when the dual bound reaches the best incumbent's objective value, avoiding unnecessary computations. 
Nodes can be stopped anytime to produce a useful dual bound, aiding overall progress. 
This flexibility contrasts with other nonlinear solvers, enabling more efficient optimization.

### Tree state-dependent termination and evolving error 

We implement different termination criteria in the node processing to reduce iterations, prioritizing nodes with promising lower bounds. 
The dual bound provided by Frank-Wolfe is always valid, even if the dual gap is large.
Thus, we solve nodes high in the tree, like the root node, with a coarse precision 
and increase the precision with which a node is solved depending on its depth in the tree. 
This approach balances efficiency and accuracy in solving optimization problems.

### Warm-starting via the active set

Many Frank-Wolfe variants provide a so-called active set, the convex combination of vertices representing the solution. 
This can be used to warm start the children nodes by partitioning the active set of the parent.

### Branching

As default, we use most-infeasible branching which has shown good performance for many problems.
Also, implemented are strong branching and the so-called hybrid branching which performs strong branching until a specific depth and afterwards switches to most-infeasible.
It should be noted that strong branching is only adviseable for problems with very cheap LMO. 
Otherwise, most-infeasible or hybrid branching with a shallow depth is to be preferred.

### Dual fixing and tightening 

In subproblems where variables are at bounds, our approach utilizes convexity and primal solutions to tighten dual bounds effectively. 
Drawing from methods pioneered by Dantzig and extended in various contexts, we leverage Frank-Wolfe methods and FW gaps, adaptable to scenarios without explicit dual solutions, such as those involving MIP-based LMOs.
We can also exploit strong convexity and sharpness to tighten the lowerbound of the tree.


## The Bounded Linear Minimization Oracles (BLMO)

The **Bounded Linear Minimization Oracle (BLMO)** represent the feasible region $C$ with the integrality constraints and handles the computation of mixed-integer linear sub-problems. 
The bound management is also handled by the BLMO. 
There are two options for the BLMO.

### Mixed Integer Linear Solver via JuMP

The first option is a MIP solver like `SCIP` via the `MathOptInterface` or `JuMP` package. 
For examples, see the Poisson Regression in `poisson_reg.jl`, the Sparse Regression with a grouped lasso in `lasso.jl`.
In `mps-example.jl`, the feasible region is encoded in an MPS file.

### Customized BLMO's

In addition, we have implemented some specific BLMOs like the hypercube, the probability and unit simplex.
For examples, see `approx_planted_point.jl`. 
In `cube_blmo.jl`, there is an example on how to implement Boscia's BLMO interface from scratch.


