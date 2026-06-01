# Plot Utilities for Boscia.jl Examples
# 
# This file contains utility functions for creating publication-quality plots
# of optimization progress (bounds over time and nodes).

using PyPlot
using PyCall

"""
    plot_bounds_progress(result::Dict, filename::String; kwargs...)

Creates a two-panel plot showing the evolution of lower and upper bounds
over time and over the number of nodes explored.

# Arguments
- `result::Dict`: Result dictionary from Boscia.solve containing keys 
  `:list_time`, `:list_num_nodes`, `:list_lb`, and `:list_ub`
- `filename::String`: Output filename (should end in .pdf, .png, etc.)

# Keyword Arguments
- `title_prefix::String = ""`: Prefix for plot titles (e.g., "A-Criterion")
- `font_family::String = "serif"`: Font family to use
- `font_size::Int = 11`: Base font size
- `use_latex::Bool = true`: Whether to use LaTeX rendering
- `latex_preamble::String = "\\\\usepackage{charter}\\\\usepackage[charter]{mathdesign}"`: 
  LaTeX preamble for font setup
- `lower_color::String = "C0"`: Color for lower bound line (matplotlib color spec)
- `upper_color::String = "C1"`: Color for upper bound line (matplotlib color spec)
- `linewidth::Real = 2`: Line width for plots
- `figsize::Tuple{Real,Real} = (12, 4)`: Figure size in inches (width, height)
- `dpi::Int = 300`: DPI for saved figure
- `show_grid::Bool = true`: Whether to show grid
- `grid_alpha::Real = 0.3`: Grid transparency (0-1)
- `legend_loc::String = "best"`: Legend location

# Returns
- `fig`: PyPlot figure object

# Example
```julia
using Boscia

# ... run Boscia.solve to get result dictionary ...
x, tlmo, result = Boscia.solve(f, grad!, lmo, settings=settings)

# Create plot with default settings
plot_bounds_progress(result, "output.pdf", title_prefix="My Problem")

# Create plot with custom colors and no LaTeX
plot_bounds_progress(result, "output.png", 
    use_latex=false,
    lower_color="blue",
    upper_color="red",
    font_size=14)
```
"""
function plot_bounds_progress(
    result::Dict,
    filename::String;
    title_prefix::String = "",
    font_family::String = "serif",
    font_size::Int = 11,
    use_latex::Bool = true,
    latex_preamble::String = "\\usepackage{charter}\\usepackage[charter]{mathdesign}",
    lower_color::String = "C0",
    upper_color::String = "C1",
    linewidth::Real = 2,
    figsize::Tuple{Real,Real} = (12, 4),
    dpi::Int = 300,
    show_grid::Bool = true,
    grid_alpha::Real = 0.3,
    legend_loc::String = "best",
)
    # Set up fonts
    PyPlot.rc("text", usetex=use_latex)
    if use_latex
        PyPlot.rc("text.latex", preamble=latex_preamble)
    end
    PyPlot.rc("font", family=font_family, size=font_size)

    # Extract data from result dictionary
    times = result[:list_time] ./ 1000.0  # Convert from milliseconds to seconds
    nodes = result[:list_num_nodes]
    lower_bounds = result[:list_lb]
    upper_bounds = result[:list_ub]

    # Create figure with two subplots
    fig = PyPlot.figure(figsize=figsize)

    # Subplot 1: Bounds over Time
    PyPlot.subplot(1, 2, 1)
    PyPlot.plot(times, lower_bounds, label="Lower Bound", linewidth=linewidth, color=lower_color)
    PyPlot.plot(times, upper_bounds, label="Upper Bound", linewidth=linewidth, color=upper_color)
    PyPlot.xlabel("Time (s)")
    PyPlot.ylabel("Objective Value")
    #if !isempty(title_prefix)
    #    PyPlot.title("$(title_prefix): Bounds over Time")
    #else
    #    PyPlot.title("Bounds over Time")
    #end
    PyPlot.legend(loc=legend_loc)
    if show_grid
        PyPlot.grid(true, alpha=grid_alpha)
    end

    # Subplot 2: Bounds over Nodes
    PyPlot.subplot(1, 2, 2)
    PyPlot.plot(nodes, lower_bounds, label="Lower Bound", linewidth=linewidth, color=lower_color)
    PyPlot.plot(nodes, upper_bounds, label="Upper Bound", linewidth=linewidth, color=upper_color)
    PyPlot.xlabel("Number of Nodes")
    PyPlot.ylabel("Objective Value")
    # if !isempty(title_prefix)
    #    PyPlot.title("$(title_prefix): Bounds over Nodes")
    #else
    #    PyPlot.title("Bounds over Nodes")
    #end
    PyPlot.legend(loc=legend_loc)
    if show_grid
        PyPlot.grid(true, alpha=grid_alpha)
    end

    # Adjust layout and save
    PyPlot.tight_layout()
    PyPlot.savefig(filename, bbox_inches="tight", dpi=dpi)
    println("Saved plot to: $filename")

    return fig
end

"""
    setup_plot_font(; kwargs...)

Convenience function to set up matplotlib fonts globally.

# Keyword Arguments
- `font_family::String = "serif"`: Font family to use
- `font_size::Int = 11`: Base font size
- `use_latex::Bool = true`: Whether to use LaTeX rendering
- `latex_preamble::String = "\\\\usepackage{charter}\\\\usepackage[charter]{mathdesign}"`: 
  LaTeX preamble for font setup

# Example
```julia
setup_plot_font(use_latex=false, font_family="sans-serif", font_size=12)
```
"""
function setup_plot_font(;
    font_family::String = "serif",
    font_size::Int = 11,
    use_latex::Bool = true,
    latex_preamble::String = "\\usepackage{charter}\\usepackage[charter]{mathdesign}",
)
    PyPlot.rc("text", usetex=use_latex)
    if use_latex
        PyPlot.rc("text.latex", preamble=latex_preamble)
    end
    PyPlot.rc("font", family=font_family, size=font_size)
    println("Configured plotting with font_family=$font_family, font_size=$font_size, use_latex=$use_latex")
end

"""
    list_available_fonts()

Lists all available fonts that matplotlib can use.
Useful for debugging font issues.
"""
function list_available_fonts()
    fm = PyCall.pyimport("matplotlib.font_manager")
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    return unique(sort(available_fonts))
end

"""
    find_fonts(pattern::String)

Find fonts matching a pattern (case-insensitive regex).

# Example
```julia
find_fonts("charter")  # Find all fonts with "charter" in the name
find_fonts("serif")    # Find all fonts with "serif" in the name
```
"""
function find_fonts(pattern::String)
    fonts = list_available_fonts()
    return filter(f -> occursin(Regex(pattern, "i"), f), fonts)
end

