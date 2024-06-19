# How does it work  ?

Boscia focuses on mixed-integer convex problems in which the nonlinear constraints and objectives are convex and presents a new algorithmic framework for solving these problems that exploit recent advances in so-called Frank-Wolfe (FW) or Conditional Gradient (CG) methods. The problem class we consider is of the type: 
$$
\min ( f(x) ) \text{ where } x \in \mathbb{R}^n, \, x \in X
$$

where \( X \) is a compact nonconvex set admitting a boundable linear minimization oracle (LMO), i.e., a set over which optimizing a linear function can be done efficiently (comparatively to the original problem), even when bounds are added or modified. Formally, we consider we have access to an oracle taking new bounds \( (l, u) \) and a direction \( d \):

Given : 
$$
\( (l, u, d) \in \mathbb{R}^n \times \mathbb{R}^n \times \mathbb{R}^n \),  \( \arg\min_{v \in \mathbb{R}^n} \langle v, d \rangle \) 
 \text {subject to}    \( v \in X \cap [l, u] \).
$$


## BPCG 


The classic FW algorithm's reliance on an LMO is inefficient for polytopes due to expensive calls. The Blended Pairwise Conditional Gradient (BPCG) algorithm improves efficiency by combining FW steps with pairwise steps that avoid LMO calls. BPCG maintains a smaller active set of vertices, enhancing performance. We use a modified lazified BPCG from FrankWolfe.jl.

# Branch and Bound Techniques 

In this section, we present the techniques derived from Frank Wolfe that can be used in our
framework .

## FW gap based termination 

Frank-Wolfe methods produce primal feasible iterates and an FW gap, offering many inexpensive iterations with a gradually increasing dual bound. This allows early termination of nodes when the dual bound reaches the best incumbent's objective value, avoiding unnecessary computations. Nodes can be stopped anytime to produce a useful dual bound, aiding overall progress. This flexibility contrasts with other nonlinear solvers, enabling more efficient optimization.

## Tree stae-dependent termination and evolving error 

We implement termination criteria in node processing to reduce iterations, prioritizing nodes with promising lower bounds. An adaptive Frank-Wolfe gap criterion increases precision with depth in the branch-and-bound tree. BPCG's convex combinations of integer extreme points ensure rapid convergence, even with reduced precision runs. This approach balances efficiency and accuracy in solving optimization problems.

## Strong Branching 

In tackling large discrete optimization problems, our method utilizes Frank-Wolfe algorithms to approximate lower bound improvements efficiently. By relaxing strong branching to continuous LPs for the Linear Minimization Oracle and controlling FW iterations or gap tolerances, we balance computational cost with accuracy. This approach optimizes variable selection in branch-and-bound algorithms, enhancing scalability and efficiency in solving complex discrete optimization challenges.

## Dual fixing and tightening 

In subproblems where variables are at bounds, our approach utilizes convexity and primal solutions to tighten dual bounds effectively. Drawing from methods pioneered by Dantzig and extended in various contexts, we leverage Frank-Wolfe methods and FW gaps, adaptable to scenarios without explicit dual solutions, such as those involving MIP-based LMOs.






