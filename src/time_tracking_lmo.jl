"""
    TimeTrackingLMO{BLMO<:BoundedLinearMinimizationOracle} <: FrankWolfe.LinearMinimizationOracle

A wrapper for the BLMO tracking the solving time, number of calls etc.
Is created in Boscia itself.
"""
mutable struct TimeTrackingLMO{BLMO<:BoundedLinearMinimizationOracle,D<:Dates.DateTime} <:
               FrankWolfe.LinearMinimizationOracle
    blmo::BLMO
    optimizing_times::Vector{Float64}
    optimizing_nodes::Vector{Int}
    simplex_iterations::Vector{Int}
    ncalls::Int
    int_vars::Vector{Int}
    time_ref::D
    type_moi::Bool
    time_limit::Float64
end

"""
    TimeTrackingLMO(blmo::BoundedLinearMinimizationOracle)

Constructor with just the blmo.
"""
TimeTrackingLMO(blmo::BoundedLinearMinimizationOracle, time_ref, time_limit) = TimeTrackingLMO(
    blmo,
    Float64[],
    Int[],
    Int[],
    0,
    Int[],
    time_ref,
    isa(blmo, MathOptBLMO),
    time_limit,
)

"""
    TimeTrackingLMO(blmo::BoundedLinearMinimizationOracle, int_vars)

Constructor with just the blmo.
"""
TimeTrackingLMO(blmo::BoundedLinearMinimizationOracle, int_vars, time_ref, time_limit) =
    TimeTrackingLMO(
        blmo,
        Float64[],
        Int[],
        Int[],
        0,
        int_vars,
        time_ref,
        isa(blmo, MathOptBLMO),
        time_limit,
    )

is_decomposition_invariant_oracle(tlmo::TimeTrackingLMO) =
    is_decomposition_invariant_oracle(tlmo.blmo)

function is_inface_feasible(tlmo::TimeTrackingLMO, a, x)
    return is_inface_feasible(tlmo.blmo, a, x)
end

function compute_inface_extreme_point(tlmo::TimeTrackingLMO, direction, x; lazy=false, kwargs...)
    tlmo.ncalls += 1
    free_model(tlmo.blmo)
    a = compute_inface_extreme_point(tlmo.blmo, direction, x)

    if !is_linear_feasible(tlmo, a)
        @debug "Vertex not linear feasible $(a)"
        @assert is_linear_feasible(tlmo, a)
    end

    opt_times, numberofnodes, simplex_iterations = get_BLMO_solve_data(tlmo.blmo)

    push!(tlmo.optimizing_times, opt_times)
    push!(tlmo.optimizing_nodes, numberofnodes)
    push!(tlmo.simplex_iterations, simplex_iterations)

    free_model(tlmo.blmo)

    return a
end

function dicg_maximum_step(tlmo::TimeTrackingLMO, direction, x)
    gamma_max = dicg_maximum_step(tlmo.blmo, direction, x)
    return gamma_max
end

"""
    reset!(tlmo::TimeTrackingLMO)
If we want to reset the info between nodes in the Branch-and-Bound tree.
"""
function reset!(tlmo::TimeTrackingLMO)
    empty!(tlmo.optimizing_times)
    empty!(tlmo.optimizing_nodes)
    empty!(tlmo.simplex_iterations)
    return tlmo.ncalls = 0
end

"""
    FrankWolfe.compute_extreme_point(tlmo::TimeTrackingLMO, d; kwargs...)

Compute the extreme point and collect statistics.
"""
function FrankWolfe.compute_extreme_point(tlmo::TimeTrackingLMO, d; kwargs...)
    tlmo.ncalls += 1
    free_model(tlmo.blmo)
    if tlmo.type_moi && isfinite(tlmo.time_limit)
        time_limit = tlmo.time_limit - float(Dates.value(Dates.now() - tlmo.time_ref)) / 1000
        time_limit = time_limit <= 0 ? 1 : time_limit
        MOI.set(tlmo.blmo.o, MOI.TimeLimitSec(), time_limit)
    end
    v = FrankWolfe.compute_extreme_point(tlmo.blmo, d; kwargs)

    if !is_linear_feasible(tlmo, v)
        @debug "Vertex not linear feasible $(v)"
        @assert is_linear_feasible(tlmo, v)
    end
    v[tlmo.int_vars] = round.(v[tlmo.int_vars])

    opt_times, numberofnodes, simplex_iterations = get_BLMO_solve_data(tlmo.blmo)

    push!(tlmo.optimizing_times, opt_times)
    push!(tlmo.optimizing_nodes, numberofnodes)
    push!(tlmo.simplex_iterations, simplex_iterations)

    free_model(tlmo.blmo)
    return v
end
