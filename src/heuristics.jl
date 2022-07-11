
"""
Finds the best solution in the SCIP solution storage, based on the objective function `f`.
Returns the solution vector and the corresponding best value.
"""
function find_best_solution(f::Function, o::SCIP.Optimizer, vars::Vector{MOI.VariableIndex})
    sols_vec = unsafe_wrap(Vector{Ptr{Cvoid}}, SCIP.LibSCIP.SCIPgetSols(o), SCIP.LibSCIP.SCIPgetNSols(o))
    best_val = Inf
    best_v = nothing
    for sol in sols_vec
        v = SCIP.sol_values(o, vars, sol)
        val = f(v)
        if val < best_val
            best_val = val
            best_v = v
        end
    end
    @assert isfinite(best_val)
    return (best_v, best_val)
end
