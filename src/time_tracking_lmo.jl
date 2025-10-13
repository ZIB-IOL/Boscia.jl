"""
    TimeTrackingLMO{LMO<:LinearMinimizationOracle} <: FrankWolfe.LinearMinimizationOracle

A wrapper for the BLMO tracking the solving time, number of calls etc.
Is created in Boscia itself.
"""
mutable struct TimeTrackingLMO{LMO<:LinearMinimizationOracle,D<:Dates.DateTime} <:
               FrankWolfe.LinearMinimizationOracle
    lmo::LMO
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
    TimeTrackingLMO(lmo::LinearMinimizationOracle)

Constructor with just the blmo.
"""
TimeTrackingLMO(lmo::LinearMinimizationOracle, time_ref, time_limit) = TimeTrackingLMO(
    lmo,
    Float64[],
    Int[],
    Int[],
    0,
    Int[],
    time_ref,
    isa(lmo, MathOptBLMO),
    time_limit,
)

"""
    TimeTrackingLMO(lmo::LinearMinimizationOracle, int_vars)

Constructor with just the blmo.
"""
TimeTrackingLMO(lmo::LinearMinimizationOracle, int_vars, time_ref, time_limit) =
    TimeTrackingLMO(
        lmo,
        Float64[],
        Int[],
        Int[],
        0,
        int_vars,
        time_ref,
        isa(lmo, MathOptBLMO),
        time_limit,
    )

is_decomposition_invariant_oracle(tlmo::TimeTrackingLMO) =
    is_decomposition_invariant_oracle(tlmo.lmo)

function is_inface_feasible(tlmo::TimeTrackingLMO, a, x)
    return is_inface_feasible(tlmo.lmo, a, x)
end

function compute_inface_extreme_point(tlmo::TimeTrackingLMO, direction, x; lazy=false, kwargs...)
    tlmo.ncalls += 1
    free_model(tlmo.lmo)
    a = compute_inface_extreme_point(tlmo.lmo, direction, x)

    if !is_linear_feasible(tlmo, a)
        @debug "Vertex not linear feasible $(a)"
        @assert is_linear_feasible(tlmo, a)
    end

    opt_times, numberofnodes, simplex_iterations = get_lmo_solve_data(tlmo.lmo)

    push!(tlmo.optimizing_times, opt_times)
    push!(tlmo.optimizing_nodes, numberofnodes)
    push!(tlmo.simplex_iterations, simplex_iterations)

    free_model(tlmo.lmo)

    return a
end

function dicg_maximum_step(tlmo::TimeTrackingLMO, direction, x)
    gamma_max = dicg_maximum_step(tlmo.lmo, direction, x)
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
    free_model(tlmo.lmo)
    if tlmo.type_moi && isfinite(tlmo.time_limit)
        time_limit = tlmo.time_limit - float(Dates.value(Dates.now() - tlmo.time_ref)) / 1000
        time_limit = time_limit <= 0 ? 1 : time_limit
        MOI.set(tlmo.lmo.o, MOI.TimeLimitSec(), time_limit)
    end
    v = FrankWolfe.compute_extreme_point(tlmo.lmo, d; kwargs)

    if !is_linear_feasible(tlmo, v)
        @debug "Vertex not linear feasible $(v)"
        @assert is_linear_feasible(tlmo, v)
    end
    v[tlmo.int_vars] = round.(v[tlmo.int_vars])

    opt_times, numberofnodes, simplex_iterations = get_lmo_solve_data(tlmo.lmo)

    push!(tlmo.optimizing_times, opt_times)
    push!(tlmo.optimizing_nodes, numberofnodes)
    push!(tlmo.simplex_iterations, simplex_iterations)

    free_model(tlmo.lmo)
    return v
end
