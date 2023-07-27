
"""
Frank-Wolfe variant used to compute the problems at node level.
A `FrankWolfeVariant` must implement
```
solve_frank_wolfe(fw::FrankWolfeVariant, f, grad!, lmo, active_set, line_search, epsilon, max_iteration,
    added_dropped_vertices, use_extra_vertex_storage, callback, lazy, timeout, verbose, workspace))
```
It may also implement `build_frank_wolfe_workspace(x)` which creates a
workspace structure that is passed as last argument to `solve_frank_wolfe`.
"""
abstract type FrankWolfeVariant end

# default printing for FrankWolfeVariant is just showing the type
Base.print(io::IO, fw::FrankWolfeVariant) = print(io, split(string(typeof(fw)), ".")[end])

"""
    solve_frank_wolfe(fw::FrankWolfeVariant, f, grad!, lmo, active_set, line_search, epsilon, max_iteration,
    added_dropped_vertices, use_extra_vertex_storage, callback, lazy, timeout, verbose, workspace)

Returns the optimal solution x to the node problem, its primal and dual gap and the active set. 
"""
function solve_frank_wolfe end

build_frank_wolfe_workspace(::FrankWolfeVariant, x) = nothing


"""
    Away-Frank-Wolfe

In every iteration, it computes the worst performing vertex, called away vertex, in the active set with regard to the gradient.
If enough local progress can be made, weight is shifted from the away vertex to all other vertices. 

In case lazification is activated, the FW vertex is only computed if not enough local progress can be guaranteed.
"""
struct AwayFrankWolfe <: FrankWolfeVariant end

function solve_frank_wolfe(
    frank_wolfe_variant::AwayFrankWolfe,
    f,
    grad!, 
    lmo,
    active_set;    
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Adaptive(),
    epsilon=1e-7,
    max_iteration=10000,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    extra_vertex_storage=nothing, 
    callback=nothing,
    lazy=false, 
    lazy_tolerance = 2.0,
    timeout=Inf,
    verbose=false,
    workspace=nothing
)
    x, _, primal, dual_gap, _, active_set = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo,
        active_set,
        epsilon=epsilon,
        max_iteration=max_iteration,
        line_search=line_search,
        callback=callback,
        lazy=lazy,  
        lazy_tolerance = lazy_tolerance,      
        timeout=timeout,
        add_dropped_vertices=add_dropped_vertices,
        use_extra_vertex_storage=use_extra_vertex_storage,
        extra_vertex_storage=extra_vertex_storage,
        verbose=verbose,
    )

    return x, primal, dual_gap, active_set
end

Base.print(io::IO, ::AwayFrankWolfe) = print(io, "Away-Frank-Wolfe")


"""
    Blended Conditional Gradient
"""
struct Blended <: FrankWolfeVariant end

function solve_frank_wolfe(
    frank_wolfe_variant::Blended,
    f,
    grad!, 
    lmo,
    active_set;
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Adaptive(),
    epsilon=1e-7,
    max_iteration=10000,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    extra_vertex_storage=nothing, 
    callback=nothing,
    lazy=false, 
    lazy_tolerance = 2.0,
    timeout=Inf,
    verbose=false,
    workspace=nothing
)
    x,_, primal, dual_gap,_, active_set = blended_conditional_gradient(
        f,
        grad!,
        lmo,
        active_set,
        line_search=line_search,
        epsilon=epsilon,
        max_iteration=max_iteration,
        add_dropped_vertices=add_dropped_vertices,
        use_extra_vertex_storage=use_extra_vertex_storage,
        extra_vertex_storage=extra_vertex_storage,
        callback=callback,
        timeout=timeout,
        verbose=verbose,
        lazy_tolerance = lazy_tolerance,
    )

    return x, primal, dual_gap, active_set
end

Base.print(io::IO, ::Blended) = print(io, "Blended Conditional Gradient")

"""
    Blended Pairwise Conditional Gradient
"""    
struct BPCG <: FrankWolfeVariant end

function solve_frank_wolfe(
    frank_wolfe_variant::BPCG,
    f,
    grad!, 
    lmo,
    active_set;
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Adaptive(),
    epsilon=1e-7,
    max_iteration=10000,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    extra_vertex_storage=nothing, 
    callback=nothing,
    lazy=false, 
    lazy_tolerance = 2.0,
    timeout=Inf,
    verbose=false,
    workspace=nothing
)
    x, _, primal, dual_gap, _, active_set = FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo,
        active_set,
        line_search=line_search,
        epsilon=epsilon,
        max_iteration=max_iteration,
        add_dropped_vertices=add_dropped_vertices,
        use_extra_vertex_storage=use_extra_vertex_storage,
        extra_vertex_storage=extra_vertex_storage,
        callback=callback,
        lazy=lazy,
        lazy_tolerance = lazy_tolerance,
        timeout=timeout,
        verbose=verbose
    )

    return x, primal, dual_gap, active_set
end

Base.print(io::IO, ::BPCG) = print(io, "Blended Pairwise Conditional Gradient")

"""
    Vanilla-Frank-Wolfe

The standard variant of Frank-Wolfe. In each iteration, the vertex v minimizing ∇f * (x-v) is computed. 

Lazification cannot be used in this setting.
"""
struct VanillaFrankWolfe <: FrankWolfeVariant end

function solve_frank_wolfe(
    frank_wolfe_variant::VanillaFrankWolfe,
    f,
    grad!, 
    lmo,
    active_set;
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Adaptive(),
    epsilon=1e-7,
    max_iteration=10000,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    extra_vertex_storage=nothing, 
    callback=nothing,
    lazy=false, 
    lazy_tolerance = 2.0,
    timeout=Inf,
    verbose=false,
    workspace=nothing
)
    # If the flag away_steps is set to false, away_frank_wolfe performs Vanilla.
    # Observe that the lazy flag is only observed if away_steps is set to true, so it can neglected. 
    x, _, primal, dual_gap, _, active_set = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo,
        active_set,
        away_steps=false, 
        epsilon=epsilon,
        max_iteration=max_iteration,
        line_search=line_search,
        callback=callback,
        timeout=timeout,
        add_dropped_vertices=add_dropped_vertices,
        use_extra_vertex_storage=use_extra_vertex_storage,
        extra_vertex_storage=extra_vertex_storage,
        verbose=verbose,
    )

    return x, primal, dual_gap, active_set
end

Base.print(io::IO, ::VanillaFrankWolfe) = print(io, "Vanilla-Frank-Wolfe")
