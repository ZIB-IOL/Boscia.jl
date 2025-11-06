
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
abstract type DecompositionInvariant <: FrankWolfeVariant end

# default printing for FrankWolfeVariant is just showing the type
Base.print(io::IO, fw::FrankWolfeVariant) = print(io, split(string(typeof(fw)), ".")[end])

"""
	solve_frank_wolfe(fw::FrankWolfeVariant, f, grad!, lmo, active_set, line_search, epsilon, max_iteration,
	added_dropped_vertices, use_extra_vertex_storage, callback, lazy, timeout, verbose, workspace)

Returns the optimal solution x to the node problem, its primal and dual gap and the active set. 
"""
function solve_frank_wolfe end

build_frank_wolfe_workspace(::FrankWolfeVariant, x) = nothing


function Base.convert(::Type{V1}, v2::V2) where {V1<:FrankWolfeVariant,V2<:FrankWolfeVariant}
    return V1()
end

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
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Secant(),
    epsilon=1e-7,
    max_iteration=10000,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    extra_vertex_storage=nothing,
    callback=nothing,
    lazy=false,
    lazy_tolerance=2.0,
    timeout=Inf,
    verbose=false,
    workspace=nothing,
    kwargs...,
)
    x, _, primal, dual_gap, status, _, active_set = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo,
        active_set,
        epsilon=epsilon,
        max_iteration=max_iteration,
        line_search=line_search,
        callback=callback,
        lazy=lazy,
        lazy_tolerance=lazy_tolerance,
        timeout=timeout,
        add_dropped_vertices=add_dropped_vertices,
        use_extra_vertex_storage=use_extra_vertex_storage,
        extra_vertex_storage=extra_vertex_storage,
        verbose=verbose,
    )

    return x, primal, dual_gap, status, active_set
end

Base.print(io::IO, ::AwayFrankWolfe) = print(io, "Away-Frank-Wolfe")


"""
    Blended Conditional Gradient
"""
struct BlendedConditionalGradient <: FrankWolfeVariant end

function solve_frank_wolfe(
    frank_wolfe_variant::BlendedConditionalGradient,
    f,
    grad!,
    lmo,
    active_set;
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Secant(),
    epsilon=1e-7,
    max_iteration=10000,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    extra_vertex_storage=nothing,
    callback=nothing,
    lazy=false,
    lazy_tolerance=2.0,
    timeout=Inf,
    verbose=false,
    workspace=nothing,
    kwargs...,
)
    x, _, primal, dual_gap, status, _, active_set = blended_conditional_gradient(
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
        sparsity_control=lazy_tolerance,
    )

    return x, primal, dual_gap, status, active_set
end

Base.print(io::IO, ::BlendedConditionalGradient) = print(io, "Blended Conditional Gradient")

"""
	Blended Pairwise Conditional Gradient
"""
struct BlendedPairwiseConditionalGradient <: FrankWolfeVariant end

function solve_frank_wolfe(
    frank_wolfe_variant::BlendedPairwiseConditionalGradient,
    f,
    grad!,
    lmo,
    active_set;
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Secant(),
    epsilon=1e-7,
    max_iteration=10000,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    extra_vertex_storage=nothing,
    callback=nothing,
    lazy=false,
    lazy_tolerance=2.0,
    timeout=Inf,
    verbose=false,
    workspace=nothing,
    kwargs...,
)
    x, _, primal, dual_gap, status, _, active_set =
        FrankWolfe.blended_pairwise_conditional_gradient(
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
            sparsity_control=lazy_tolerance,
            timeout=timeout,
            verbose=verbose,
        )
    return x, primal, dual_gap, status, active_set
end

Base.print(io::IO, ::BlendedPairwiseConditionalGradient) =
    print(io, "Blended Pairwise Conditional Gradient")

"""
   DICG-Frank-Wolfe

The Decomposition-invariant Frank-Wolfe. 

"""
struct DecompositionInvariantConditionalGradient <: DecompositionInvariant
    use_strong_lazy::Bool
    use_DICG_warm_start::Bool
    use_strong_warm_start::Bool
end

function DecompositionInvariantConditionalGradient(;
    use_strong_lazy=false,
    use_DICG_warm_start=false,
    use_strong_warm_start=false,
)
    return DecompositionInvariantConditionalGradient(
        use_strong_lazy,
        use_DICG_warm_start,
        use_strong_warm_start,
    )
end

function solve_frank_wolfe(
    frank_wolfe_variant::DecompositionInvariantConditionalGradient,
    f,
    grad!,
    lmo,
    active_set;
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Secant(),
    epsilon=1e-7,
    max_iteration=10000,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    extra_vertex_storage=nothing,
    callback=nothing,
    lazy=false,
    lazy_tolerance=2.0,
    timeout=Inf,
    verbose=false,
    workspace=nothing,
    pre_computed_set=nothing,
    domain_oracle=_trivial_domain,
    kwargs...,
)
    x0, DICG_callback = init_decomposition_invariant_state(active_set, pre_computed_set, callback)

    x, _, primal, dual_gap, status, _ = FrankWolfe.decomposition_invariant_conditional_gradient(
        f,
        grad!,
        lmo,
        x0;
        line_search=line_search,
        epsilon=epsilon,
        max_iteration=max_iteration,
        verbose=verbose,
        timeout=timeout,
        lazy=lazy,
        use_strong_lazy=frank_wolfe_variant.use_strong_lazy,
        linesearch_workspace=workspace,
        sparsity_control=lazy_tolerance,
        callback=DICG_callback,
        extra_vertex_storage=pre_computed_set,
    )

    clean_up_pre_computed_set!(lmo, pre_computed_set, x, frank_wolfe_variant)

    return x, primal, dual_gap, status, pre_computed_set
end

Base.print(io::IO, ::DecompositionInvariantConditionalGradient) =
    print(io, "Decompostion-Invariant-Frank-Wolfe")

"""
   BDICG-Frank-Wolfe

The Blended Decomposition-invariant Frank-Wolfe. 

"""
struct BlendedDecompositionInvariantConditionalGradient <: DecompositionInvariant
    use_strong_lazy::Bool
    use_BDICG_warm_start::Bool
    use_strong_warm_start::Bool
end

function BlendedDecompositionInvariantConditionalGradient(;
    use_strong_lazy=false,
    use_BDICG_warm_start=false,
    use_strong_warm_start=false,
)
    return BlendedDecompositionInvariantConditionalGradient(
        use_strong_lazy,
        use_BDICG_warm_start,
        use_strong_warm_start,
    )
end

function solve_frank_wolfe(
    frank_wolfe_variant::BlendedDecompositionInvariantConditionalGradient,
    f,
    grad!,
    lmo,
    active_set;
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Secant(),
    epsilon=1e-7,
    max_iteration=10000,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    extra_vertex_storage=nothing,
    callback=nothing,
    lazy=false,
    lazy_tolerance=2.0,
    timeout=Inf,
    verbose=false,
    workspace=nothing,
    pre_computed_set=nothing,
    domain_oracle=_trivial_domain,
    kwargs...,
)

    x0, BDICG_callback = init_decomposition_invariant_state(active_set, pre_computed_set, callback)

    x, _, primal, dual_gap, status, _ =
        FrankWolfe.blended_decomposition_invariant_conditional_gradient(
            f,
            grad!,
            lmo,
            x0;
            line_search=line_search,
            epsilon=epsilon,
            max_iteration=max_iteration,
            verbose=verbose,
            timeout=timeout,
            lazy=lazy,
            use_strong_lazy=frank_wolfe_variant.use_strong_lazy,
            linesearch_workspace=workspace,
            sparsity_control=lazy_tolerance,
            callback=BDICG_callback,
            extra_vertex_storage=pre_computed_set,
        )

    clean_up_pre_computed_set!(lmo, pre_computed_set, x, frank_wolfe_variant)

    return x, primal, dual_gap, status, pre_computed_set
end

Base.print(io::IO, ::BlendedDecompositionInvariantConditionalGradient) =
    print(io, "Blended-Decompostion-Invariant-Frank-Wolfe")

"""
	Vanilla-Frank-Wolfe

The standard variant of Frank-Wolfe. In each iteration, the vertex v minimizing ∇f * (x-v) is computed. 

Lazification cannot be used in this setting.
"""
struct StandardFrankWolfe <: FrankWolfeVariant end

function solve_frank_wolfe(
    frank_wolfe_variant::StandardFrankWolfe,
    f,
    grad!,
    lmo,
    active_set;
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Secant(),
    epsilon=1e-7,
    max_iteration=10000,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    extra_vertex_storage=nothing,
    callback=nothing,
    lazy=false,
    lazy_tolerance=2.0,
    timeout=Inf,
    verbose=false,
    workspace=nothing,
    kwargs...,
)
    # If the flag away_steps is set to false, away_frank_wolfe performs Vanilla.
    # Observe that the lazy flag is only observed if away_steps is set to true, so it can neglected. 
    x, _, primal, dual_gap, status, _, active_set = FrankWolfe.away_frank_wolfe(
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
    return x, primal, dual_gap, status, active_set
end

Base.print(io::IO, ::StandardFrankWolfe) = print(io, "StandardFrank-Wolfe")
