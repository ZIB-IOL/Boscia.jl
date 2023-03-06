"""
    TimeTrackingLMO  <: FW.LMO

An LMO wrapping another one tracking the time, number of nodes and number of calls.

"""
mutable struct TimeTrackingLMO{LMO<:FrankWolfe.LinearMinimizationOracle} <:
               FrankWolfe.LinearMinimizationOracle
    lmo::LMO
    optimizing_times::Vector{Float64}
    optimizing_nodes::Vector{Int}
    simplex_iterations::Vector{Int}
    ncalls::Int
    int_vars::Vector{Int}
end

TimeTrackingLMO(lmo::FrankWolfe.LinearMinimizationOracle) =
    TimeTrackingLMO(lmo, Float64[], Int[], Int[], 0, Int[])
  
TimeTrackingLMO(lmo::FrankWolfe.LinearMinimizationOracle, int_vars) =
    TimeTrackingLMO(lmo, Float64[], Int[], Int[], 0, int_vars)

# if we want to reset the info between nodes in Bonobo
function reset!(lmo::TimeTrackingLMO)
    empty!(lmo.optimizing_times)
    empty!(lmo.optimizing_nodes)
    empty!(lmo.simplex_iterations)
    return lmo.ncalls = 0
end

function FrankWolfe.compute_extreme_point(lmo::TimeTrackingLMO, d; kwargs...)
    lmo.ncalls += 1
    cleanup_solver(lmo.lmo.o)
    v = FrankWolfe.compute_extreme_point(lmo.lmo, d; kwargs)

    if !is_linear_feasible(lmo, v)
        @debug "Vertex not linear feasible $(v)"
        @assert is_linear_feasible(lmo, v)
    end
    v[lmo.int_vars] = round.(v[lmo.int_vars])

    push!(lmo.optimizing_times, MOI.get(lmo.lmo.o, MOI.SolveTimeSec()))
    numberofnodes = MOI.get(lmo.lmo.o, MOI.NodeCount())
    push!(lmo.optimizing_nodes, numberofnodes)
    push!(lmo.simplex_iterations, MOI.get(lmo.lmo.o, MOI.SimplexIterations()))

    cleanup_solver(lmo.lmo.o)
    return v
end

cleanup_solver(o) = nothing
cleanup_solver(o::SCIP.Optimizer) = SCIP.SCIPfreeTransform(o)

MOI.optimize!(time_lmo::TimeTrackingLMO) = MOI.optimize!(time_lmo.lmo.o)
