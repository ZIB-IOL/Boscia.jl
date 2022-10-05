
"""
Finds the best solution in the SCIP solution storage, based on the objective function `f`.
Returns the solution vector and the corresponding best value.
"""
function find_best_solution(f::Function, o::SCIP.Optimizer, vars::Vector{MOI.VariableIndex})
    sols_vec =
        unsafe_wrap(Vector{Ptr{Cvoid}}, SCIP.LibSCIP.SCIPgetSols(o), SCIP.LibSCIP.SCIPgetNSols(o))
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

"""
Finds the best solution in the HiGHS solution storage, based on the objective function `f`.
Returns the solution vector and the corresponding best value.
"""
function find_best_solution(f::Function, o::HiGHS.Optimizer, vars::Vector{MOI.VariableIndex})
    ncol = Highs_getNumCol(o)
    nrow = Highs_getNumRow(o)
    col_value = Vector{Float64}(undef, ncol)
    col_dual = Vector{Float64}(undef, ncol)
    row_value = Vector{Float64}(undef, nrow)
    row_dual = Vector{Float64}(undef, nrow)

    HiGHS.Highs_getSolution(o, col_value, col_dual, row_value, row_dual)
    val = f(col_value)
    best_v = col_value
    
    @assert isfinite(val)
    return (col_value, val)
end