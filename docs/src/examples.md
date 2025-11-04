# Examples

This page provides an overview of example applications of Boscia.jl. Each example demonstrates different use cases and techniques for solving mixed-integer convex optimization problems.

## Featured Examples

### [Network Design Problem](@ref)

This example demonstrates solving a transportation network design problem using Boscia.jl with two approaches:
1. MOI-based LMO: Using MathOptInterface to model the feasible region
2. Custom LMO: Using a customized Linear Minimization Oracle based on shortest path algorithms

[View full example →](@ref)

### [Graph Isomorphism Problem](@ref)

This example shows how to use Boscia to certify whether two graphs are isomorphic. We encode graph isomorphism via a permutation matrix X that reorders the vertices.

[View full example →](@ref)

### [Optimal Design of Experiments](@ref)

This example demonstrates the A-Optimal and D-Optimal Design of Experiments problems, using the Fisher information matrix to quantify information.

[View full example →](@ref)

## Additional Examples

Other example files available in the `examples/` directory include:

- **Poisson Regression** (`poisson_reg.jl`): Example using Poisson regression with integer constraints
- **Sparse Regression** (`sparse_reg.jl`, `int_sparse_reg.jl`, `lasso.jl`): Various sparse regression formulations
- **Portfolio Optimization** (`portfolio.jl`, `strong_branching_portfolio.jl`): Portfolio optimization examples
- **Birkhoff Polytope** (`birkhoff_decomposition.jl`, `quadratic_over_birkhoff.jl`): Examples working with the Birkhoff polytope
- **Custom BLMO** (`cube_blmo.jl`): Example of implementing a custom BLMO from scratch
- **MPS Files** (`mps-example.jl`): Example using MPS file format for feasible regions

