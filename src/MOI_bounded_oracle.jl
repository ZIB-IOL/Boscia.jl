"""

The Bounded Linear Minimization Oracle for solvers supporting MathOptInterface.
"""
struct MathOptBLMO{OT<:MOI.AbstractOptimizer} <: BoundedLinearMinimizationOracle
    o::OT
    use_modify::Bool
    function MathOptLMO(o, use_modify=true)
        MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        return new{typeof(o)}(o, use_modify)
    end
end

"""
Get list of lower bounds.
"""
function get_lower_bounds_list(blmo::MathOptBLMO)
    return MOI.get(blmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.LessThan{Float64}}())
end

"""
Get list of upper bounds.
"""
function get_upper_bounds_list(blmo::MathOptBLMO)
    return MOI.get(blmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.GreaterThan{Float64}}())
end

"""
Get lower bound of variable.
"""
function get_lower_bound(blmo, c_idx) 
    return MOI.get(blmo.o, MOI.ConstraintSet(), c_idx)
end

"""
Get upper bound of variable.
"""
function get_upper_bound(blmo, c_idx) 
    return MOI.get(blmo.o, MOI.ConstraintSet(), c_idx)
end

"""
Set new lower bound.
"""
function set_lower_bound!(blmo, c_idx, value)
    MOI.set(blmo.o, MOI.ConstraintSet(), c_idx, value)
end

"""
Set new upper bound.
"""
function set_upper_bound!(blmo, c_idx, value) 
    MOI.set(blmo.o, MOI.ConstraintSet(), c_idx, value)
end

"""
Delete constraints that are not needed anymore.
"""
function delete_constraints!(blmo, cons_delete) 
    for d_idx in cons_delete
        MOI.delete(blmo, d_idx)
    end
end

"""
Add new bound constraint.
"""
function add_constraint!(blmo, key, value) 
    MOI.add_constraint(blmo, MOI.VariableIndex(key), value)
end

"""
Check if the bound constraint is on an integral variable.
"""
function is_constraint_on_int_var(c_idx, int_vars)
    return c_idx.value in int_vars
end

"""
Is the constraint part of this contraints set. Used to check membership to the global and node bounds.
"""
function is_constraint_in(set, c_idx)
    return haskey(set, c_idx.value)
end

"""
Function for MOI.optimize!
"""
MOI.optimize!(blmo::MathOptBLMO) = MOI.optimize!(blmo.o)

"""
Get the variable of the bound constraint.
"""
function get_variable_index(blmo::MathOptBLMO, c_idx)
    return c_idx.value
end

"""
Check that all node bounds were set correctly. Not necessary
"""
function check_bounds(blmo::MathOptBLMO, node_bounds)
    for list in (node_bounds.lower_bounds, node_bounds.upper_bounds)
        for (idx, set) in list
            c_idx =  MOI.ConstraintIndex{MOI.VariableIndex, typeof(set)}(idx)
            @assert MOI.is_valid(blmo.o, c_idx)
            set2 = MOI.get(blmo.o, MOI.ConstraintSet(), c_idx)
            if !(set == set2)
                MOI.set(blmo.o, MOI.ConstraintSet(), c_idx, set)
                set3 = MOI.get(blmo.o, MOI.ConstraintSet(), c_idx)
                @assert (set3 == set) "$((idx, set3, set))"
            end
        end
    end
    return true
 end

 """
 Check that the problem is feasible, i.e. there are no contradicting constraints, and that it is bounded.
 """
function check_feasibility(blmo::BoundedLinearMinimizationOracle)
    MOI.set(
        blmo.o,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction{Float64}([], 0.0),
    )
    MOI.optimize!(blmo)
    status = MOI.get(blmo.o, MOI.TerminationStatus())
    return status
end

 """
 Compute the extreme point given a direction.
 """
 function compute_extreme_point(
    blmo::MathOptBLMO{OT},
    direction::AbstractVector{T};
    kwargs...,
) where {OT,T<:Real}
    variables = MOI.get(blmo.o, MOI.ListOfVariableIndices())
    if blmo.use_modify
        for i in eachindex(variables)
            MOI.modify(
                blmo.o,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                MOI.ScalarCoefficientChange(variables[i], direction[i]),
            )
        end
    else
        terms = [MOI.ScalarAffineTerm(d, v) for (d, v) in zip(direction, variables)]
        obj = MOI.ScalarAffineFunction(terms, zero(T))
        MOI.set(blmo.o, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    end
    return _optimize_and_return(blmo, variables)
end

"""
Compute extreme point when the given direction is a matrix.
"""
function compute_extreme_point(
    blmo::MathOptBLMO{OT},
    direction::AbstractMatrix{T};
    kwargs...,
) where {OT,T<:Real}
    n = size(direction, 1)
    v = compute_extreme_point(blmo, vec(direction))
    return reshape(v, n, n)
end

"""
Copy BLMO.
"""
function Base.copy(blmo::MathOptBLMO{OT}; ensure_identity=true) where {OT}
    opt = OT() # creates the empty optimizer
    index_map = MOI.copy_to(opt, blmo.o)
    if ensure_identity
        for (src_idx, des_idx) in index_map.var_map
            if src_idx != des_idx
                error("Mapping of variables is not identity")
            end
        end
    end
    return MathOptBLMO(opt)
end
 # is this necessary for us?
function Base.copy(
    blmo::MathOptBLMO{OT};
    ensure_identity=true,
) where {OTI,OT<:MOIU.CachingOptimizer{OTI}}
    opt = MOIU.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        OTI(),
    )
    index_map = MOI.copy_to(opt, blmo.o)
    if ensure_identity
        for (src_idx, des_idx) in index_map.var_map
            if src_idx != des_idx
                error("Mapping of variables is not identity")
            end
        end
    end
    return MathOptBLMO(opt)
end


# Necessary?
function compute_extreme_point(
    blmo::MathOptBLMO{OT},
    direction::AbstractVector{MOI.ScalarAffineTerm{T}};
    kwargs...,
) where {OT,T}
    if blmo.use_modify
        for d in direction
            MOI.modify(
                blmo.o,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                MOI.ScalarCoefficientChange(d.variable, d.coefficient),
            )
        end

        variables = MOI.get(blmo.o, MOI.ListOfVariableIndices())
        variables_to_zero = setdiff(variables, [dir.variable for dir in direction])

        terms = [
            MOI.ScalarAffineTerm(d, v) for
            (d, v) in zip(zeros(length(variables_to_zero)), variables_to_zero)
        ]

        for t in terms
            MOI.modify(
                blmo.o,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                MOI.ScalarCoefficientChange(t.variable, t.coefficient),
            )
        end
    else
        variables = [d.variable for d in direction]
        obj = MOI.ScalarAffineFunction(direction, zero(T))
        MOI.set(blmo.o, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    end
    return _optimize_and_return(blmo, variables)
end

"""
Optimize the given problem.
"""
function _optimize_and_return(blmo, variables)
    MOI.optimize!(blmo.o)
    term_st = MOI.get(blmo.o, MOI.TerminationStatus())
    if term_st ∉ (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.SLOW_PROGRESS)
        @error "Unexpected termination: $term_st"
        return MOI.get.(blmo.o, MOI.VariablePrimal(), variables)
    end
    return MOI.get.(blmo.o, MOI.VariablePrimal(), variables)
end