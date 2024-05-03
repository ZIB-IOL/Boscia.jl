"""
    TimeTrackingLMO  <: FW.LMO

An LMO wrapping another one tracking the time, number of nodes and number of calls.

"""
mutable struct TimeTrackingLMO{BLMO<:BoundedLinearMinimizationOracle} <:
               FrankWolfe.LinearMinimizationOracle
    blmo::BLMO
    optimizing_times::Vector{Float64}
    optimizing_nodes::Vector{Int}
    simplex_iterations::Vector{Int}
    ncalls::Int
    int_vars::Vector{Int}
end

TimeTrackingLMO(blmo::BoundedLinearMinimizationOracle) =
    TimeTrackingLMO(blmo, Float64[], Int[], Int[], 0, Int[])

TimeTrackingLMO(blmo::BoundedLinearMinimizationOracle, int_vars) =
    TimeTrackingLMO(blmo, Float64[], Int[], Int[], 0, int_vars)

#========================================================================================#
is_decomposition_invariant_oracle(lmo::TimeTrackingLMO) = is_decomposition_invariant_oracle(lmo.blmo)

function compute_inface_extreme_point(lmo::TimeTrackingLMO, direction, x; lazy, kwargs...)
    tlmo.ncalls += 1
    free_model(tlmo.blmo)
    sblmo = lmo.blmo.simple_lmo
    a = compute_inface_extreme_point(lmo.blmo, direction, x)
    
    if !is_linear_feasible(tlmo, a)
        @debug "Vertex not linear feasible $(v)"
        @assert is_linear_feasible(tlmo, v)
    end

    opt_times, numberofnodes, simplex_iterations = get_BLMO_solve_data(tlmo.blmo)

    push!(tlmo.optimizing_times, opt_times)
    push!(tlmo.optimizing_nodes, numberofnodes)
    push!(tlmo.simplex_iterations, simplex_iterations)

    free_model(tlmo.blmo)
    
    return a
end

function dicg_maximum_step(lmo::TimeTrackingLMO, x, direction)
    sblmo = lmo.blmo.simple_lmo
    gamma_max = dicg_maximum_step(lmo.blmo, x, direction)
    return gamma_max
end
#========================================================================================#

# if we want to reset the info between nodes in Bonobo
function reset!(tlmo::TimeTrackingLMO)
    empty!(tlmo.optimizing_times)
    empty!(tlmo.optimizing_nodes)
    empty!(tlmo.simplex_iterations)
    return tlmo.ncalls = 0
end

function FrankWolfe.compute_extreme_point(tlmo::TimeTrackingLMO, d; kwargs...)
    tlmo.ncalls += 1
    free_model(tlmo.blmo)
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
