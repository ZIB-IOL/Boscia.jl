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

First, we introduce the standard Frank-Wolfe algorithm, referred to as *Vanilla*, and the other variants available in `Boscia.jl`. 

### Vanilla Frank-Wolfe

Given an initial start point $x_0 \in C$, the Vanilla variant uses the gradient at the current iterate to solve a linear minimization problem.
The solution of the linear problem is used as the next descend direction and iterate is updated after a line search step.
The algorithm is very simple and able to handle large scale problems. 
Also, it often produces sparse solutions and has the advantages of naturally presenting the solution as a convex combination of extreme points of the feasible region, the so-called **active set**. 

### Away-Frank-Wolfe (AFW)

The Away-Frank-Wolfe (AFW) algorithm is an enhancement of the Vanilla method, designed to improve convergence, particularly in cases where the solution lies at a vertex or along the boundary of the feasible region. 
In addition to the standard Frank-Wolfe steps, AFW introduces an "away step" that allows the algorithm to move away from previously chosen vertices of the feasible set. 
This helps in reducing the zigzagging behavior often observed in Vanilla FW and accelerates convergence, particularly towards solutions on the boundary of the feasible region. 
AFW maintains the simplicity of the original algorithm while achieving faster convergence in many cases.

### Blended Conditional Gradient (BCG)

TODO

### Blended Pairwise Conditional Gradient (BPCG)

The classic FW algorithm's reliance on an LMO is inefficient for polytopes due to expensive calls. 
The Blended Pairwise Conditional Gradient (BPCG) algorithm improves efficiency by combining FW steps with pairwise steps that avoid LMO calls. 
BPCG maintains a smaller active set of vertices, enhancing performance. 
We use a modified lazified BPCG from FrankWolfe.jl.



## Branch-and-Bound techniques 

In this section, we present the techniques derived from Frank Wolfe that can be used in our framework .

### Dual gap based termination 

Frank-Wolfe methods produce primal feasible iterates and an FW gap, offering many inexpensive iterations with a gradually increasing dual bound. 
This allows early termination of nodes when the dual bound reaches the best incumbent's objective value, avoiding unnecessary computations. 
Nodes can be stopped anytime to produce a useful dual bound, aiding overall progress. 
This flexibility contrasts with other nonlinear solvers, enabling more efficient optimization.

### Tree state-dependent termination and evolving error 

We implement termination criteria in node processing to reduce iterations, prioritizing nodes with promising lower bounds. 
An adaptive Frank-Wolfe gap criterion increases precision with depth in the branch-and-bound tree. 
Frank Wolfe's convex combinations of integer extreme points ensure valid dual bound, even with reduced precision runs. 
This approach balances efficiency and accuracy in solving optimization problems.we have Hybrid branching which couples strong branching and most infeasible branching. 
Strong branching is very expensive to do for the whole tree and it's usually more beneficial to utilize strong branching only up to a certain depth.

### Warm-starting via the active set



## The Bounded Linear Minimization Oracles (BLMO)

The **Bounded Linear Minimization Oracle (BLMO)** represent the feasible region $C$ with the integrality constraints and handles the computation of mixed-integer linear sub-problems. 
The bound management is also handled by the BLMO. 
There are two options for the BLMO.

### Mixed Integer Linear Solver via JuMP

The first option is a MIP solver like `SCIP` via the `MathOptInterface` or `JuMP` package. 
For examples, see the Poisson Regression in `poisson_reg.jl`, the Sparse Regression with a grouped lasso in `lasso.jl`.
In `mps-example.jl`, the feasible region is encoded in an MPS file.

### Customized BLMO's

Point to the tutorial



## Strong branching and tightenings 

In tackling large discrete optimization problems, our method utilizes Frank-Wolfe algorithms to approximate lower bound improvements efficiently. 
By relaxing strong branching to continuous LPs for the Linear Minimization Oracle and controlling FW iterations or gap tolerances, we balance computational cost with accuracy. 
This approach optimizes variable selection in branch-and-bound algorithms, enhancing scalability and efficiency in solving complex discrete optimization challenges.

### Dual fixing and tightening 

In subproblems where variables are at bounds, our approach utilizes convexity and primal solutions to tighten dual bounds effectively. 
Drawing from methods pioneered by Dantzig and extended in various contexts, we leverage Frank-Wolfe methods and FW gaps, adaptable to scenarios without explicit dual solutions, such as those involving MIP-based LMOs.









