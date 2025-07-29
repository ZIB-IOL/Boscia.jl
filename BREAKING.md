# Noteworthy changes from v0.1 to v0.2

- optional parameters in the `solve` function are now grouped into Branch-and-Bound settings, Frank-Wolfe settings, tolerances
postprocessing settings, heuristic settings, tightening settings and settings for non-trivial domains.
- interface for different modes. For now, we have the DEFAULT mode and HEURISTIC mode.
- Renaming the Frank-Wolfe variants such that the names are consisent. We favour the abbreviations.
- DICG settings are moved to the DICG struct instead of having them as top-level settings.