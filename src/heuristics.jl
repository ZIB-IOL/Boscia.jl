
"""
Finds the best solution in the SCIP solution storage, based on the objective function `f`.
Returns the solution vector and the corresponding best value.
"""
function find_best_solution(f::Function, o::SCIP.Optimizer, vars::Vector{MOI.VariableIndex}, domain_oracle)
    sols_vec =
        unsafe_wrap(Vector{Ptr{Cvoid}}, SCIP.LibSCIP.SCIPgetSols(o), SCIP.LibSCIP.SCIPgetNSols(o))
    best_val = Inf
    best_v = nothing
    for sol in sols_vec
        v = SCIP.sol_values(o, vars, sol)
        if domain_oracle(v)
            val = f(v)
            if val < best_val
                best_val = val
                best_v = v
            end
        end
    end
    #@assert isfinite(best_val) -> not necessarily the case if the domain oracle is not the default.
    return (best_v, best_val)
end

"""
Finds the best solution in the Optimizer's solution storage, based on the objective function `f`.
Returns the solution vector and the corresponding best value.
"""
function find_best_solution(f::Function, o::MOI.AbstractOptimizer, vars::Vector{MOI.VariableIndex}, domain_oracle)
    nsols = MOI.get(o, MOI.ResultCount())
    @assert nsols > 0
    best_val = Inf
    best_v = nothing
    for sol_idx in 1:nsols
        xv = [MOI.get(o, MOI.VariablePrimal(sol_idx), xi) for xi in vars]
        if domain_oracle(xv)
            val = f(xv)
            if val < best_val
                best_val = val
                best_v = xv
            end
        end
    end
    return (best_v, best_val)
end
