# Result attributes of Boscia.Optimizer: information about the last call to
# `MOI.optimize!` (why it stopped, whether a solution was found, and the
# solution itself).
#
# All values are read from fields of `Optimizer` that are set in
# `MOI.optimize!` and reset in `MOI.empty!`. Result attributes can only be
# queried, so each one has a `MOI.get` method but no `MOI.set` or
# `MOI.supports`.
#
# Boscia keeps a list of solutions internally (`result[:tree_solutions]`),
# but this wrapper only reports the best one, returned by `Boscia.solve`.
# Therefore `MOI.ResultCount` is either 0 or 1.

"""
    MOI.get(optimizer::Optimizer, ::MOI.ResultCount)

Return the number of solutions available from the last solve: `1` if Boscia
found a solution, and `0` otherwise.
"""
function MOI.get(optimizer::Optimizer, ::MOI.ResultCount)
    # A solution is available exactly when the primal status is FEASIBLE_POINT,
    # which is set in `MOI.optimize!` only if Boscia returned a solution.
    if optimizer.primal_status == MOI.FEASIBLE_POINT
        return 1
    else
        return 0
    end
end

"""
    MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)

Return why the last call to `MOI.optimize!` stopped, for example
`MOI.OPTIMAL` or `MOI.TIME_LIMIT`.

Returns `MOI.OPTIMIZE_NOT_CALLED` if `MOI.optimize!` has not been called since
the optimizer was created or emptied.
"""
function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    return optimizer.termination_status
end

"""
    MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)

Return Boscia's own description of why the last solve stopped, for example
`"Optimal (tree empty)"`.

Returns `""` if `MOI.optimize!` has not been called.
"""
function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return optimizer.raw_status_string
end

"""
    MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)

Return whether a primal solution is available: `MOI.FEASIBLE_POINT` if Boscia
found a solution, and `MOI.NO_SOLUTION` otherwise.

Returns `MOI.NO_SOLUTION` if `attr.result_index` is larger than
`MOI.ResultCount`, instead of throwing an error.
"""
function MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)
    # Asking for the status of a solution that does not exist is valid:
    # the answer is NO_SOLUTION, not an error.
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    return optimizer.primal_status
end

"""
    MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue)

Return the objective value of the solution found by Boscia, in the user's
objective sense (the sign is already flipped back for `MOI.MAX_SENSE`).

Throws `MOI.ResultIndexBoundsError` if no solution is available, i.e. if
`attr.result_index` is larger than `MOI.ResultCount`.
"""
function MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.objective_value
end

"""
    MOI.get(optimizer::Optimizer, attr::MOI.VariablePrimal, variable::MOI.VariableIndex)

Return the value of `variable` in the solution found by Boscia.

Throws `MOI.ResultIndexBoundsError` if no solution is available, i.e. if
`attr.result_index` is larger than `MOI.ResultCount`.
"""
function MOI.get(optimizer::Optimizer, attr::MOI.VariablePrimal, variable::MOI.VariableIndex)
    MOI.check_result_index_bounds(optimizer, attr)

    return optimizer.variable_primal[variable.value]
end
