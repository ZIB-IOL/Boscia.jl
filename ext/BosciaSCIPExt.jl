module BosciaSCIPExt

using Boscia
using MathOptInterface
using SCIP
const MOI = MathOptInterface
const MOIU = MOI.Utilities

import MathOptSetDistances as MOD

"""
Finds the best solution in the SCIP solution storage, based on the objective function `f`.
Returns the solution vector and the corresponding best value.
"""
function Boscia.find_best_solution(
    f::Function,
    o::SCIP.Optimizer,
    vars::Vector{MOI.VariableIndex},
    domain_oracle,
)
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
Cleanup internal SCIP model
"""
function Boscia.free_model(o::SCIP.Optimizer)
    println("Free model for SCIP")
    return SCIP.SCIPfreeTransform(o)
end

"""
Get solving tolerance.
"""
function Boscia.get_tol(o::SCIP.Optimizer)
    return MOI.get(o, MOI.RawOptimizerAttribute("numerics/feastol"))
end

end # module
