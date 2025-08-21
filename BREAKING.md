# Noteworthy changes from v0.1 to v0.2

- optional parameters in the `solve` function are now grouped into Branch-and-Bound settings, Frank-Wolfe settings, tolerances
postprocessing settings, heuristic settings, tightening settings and settings for non-trivial domains.
- interface for different modes. For now, we have the `DEFAULT_MODE` and `HEURISTIC_MODE`.
- Renaming the Frank-Wolfe variants such that the names are consisent. We favour the full names.
- Decomposition Invariant Conditional Gradient settings are moved to the `DecompositionInvariantConditionalGradient` struct instead of having them as top-level settings.
- removed stale `clean_solutions`and `max_clean_iter` parameters.
- Boscia's own heuristics can now be constructed by just specifiying their probabilities. 