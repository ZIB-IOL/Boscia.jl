# Algorithm Interface

Boscia's `solve` function only requires the oracles of the objective function `f` and its gradient `g` as well as the BLMO encoding the feasible region.
For the possible settings, see further down the page.

```@autodocs
Modules = [Boscia]
Pages = ["src/interface.jl"]
```

## Optional settings

Boscia has a lot of settings to customize the solving process. These are grouped by 
* general Branch-and-Bound settings 
* settings specific for Frank-Wolfe 
* tolerances settings for both the tree as well as the Frank-Wolfe algorithm 
* settings for the heuristics
* bound tightenings settings
* postprocessing settings
* parameters for the case of a non-trivial domain, i.e. the objective cannot be evaluated at all points of the feasible region

```@autodocs
Modules = [Boscia]
Pages = ["src/settings.jl"]
```

## Definitions

Boscia defines its own solving state. 
Additionally, Boscia has different modes, like the `DEFAULT_MODE` and `HEURISTIC_MODE`.
These have their own default settings for the optional parameters.
