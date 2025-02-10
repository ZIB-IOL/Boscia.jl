# Branch and Bound Tree functionality

Thwe functionality of the Branch-and-Bound implementation extended from Bonobo.jl and some extended features like strong branching and the callbacks.

## Problem 
The main problem structure as stored in the Branch-and-Bound tree.

```@autodocs
Modules = [Boscia]
Pages = ["problem.jl"]
```

## Customized Bonobo structures and functions 
Our adaptations to the functions in `Bonobo.jl`. 

```@autodocs
Modules = [Boscia]
Pages = ["custom_bonobo.jl"]
```

## Node Evaluation
Evaluation of the nodes and handling of branching.

```@autodocs
Modules = [Boscia]
Pages = ["node.jl"]
```

## Callbacks
There are two callbacks. 
One for the Branch-and-Bound tree that records progress data, checks the time limit and prints the logs.
The other is a callback for the Frank-Wolfe runs that runs some checks in each iteration. 
Additionally, the computed vertices are added to the solution pool.
Lastly, the Frank-Wolfe solve can be interrupted if either the dual bound has reached the current incumbent or 
there are enough more promising nodes open.

```@autodocs
Modules = [Boscia]
Pages = ["callbacks.jl"]
```

## Tightenings
Tightenings are performed on node level and can be used either just for the node in question or globally.
If the obejctive is strongly convex and/or sharp, this can also be used to tighten the lower bound at the current node. 

```@autodocs
Modules = [Boscia]
Pages = ["tightenings.jl"]
```

## Strong and Hybrid Branching
We provide a strong branching strategy consisting of running Frank-Wolfe for only a few iterations to get an estimate of the bound increase.
Due to the cost of strong branching, it is usually not advisable to run strong branching through the whole tree.
Hence, we provide a hybrid branching which performs strong branching until a user specified depth and then switches to most-infeasible branching. 

```@autodocs
Modules = [Boscia]
Pages = ["strong_branching.jl"]
```