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

#MOI.optimize!(time_lmo::TimeTrackingLMO) = MOI.optimize!(time_lmo.lmo.o)
