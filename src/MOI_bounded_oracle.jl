"""
    BoundedLinearMinimizationOracle for solvers supporting MathOptInterface.
"""
struct MathOptBLMO{OT<:MOI.AbstractOptimizer} <: BoundedLinearMinimizationOracle
    o::OT
    use_modify::Bool
    function MathOptBLMO(o, use_modify=true)
        MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        return new{typeof(o)}(o, use_modify)
    end
end

"""
    compute_extreme_point

Is implemented in the FrankWolfe package in file "moi_oracle.jl".
"""

"""
Get the index of the integer variable the bound is working on.
"""
function get_int_var(blmo::MathOptBLMO, c_idx) 
    return c_idx.value
end

"""
Get the list of lower bounds.
"""
function get_lower_bound_list(blmo::MathOptBLMO) 
    return MOI.get(blmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.LessThan{Float64}}())
end
"""
Get the list of upper bounds.
"""
function get_upper_bound_list(blmo::MathOptBLMO)
    return MOI.get(blmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.GreaterThan{Float64}}()) 
end 

"""
Change the value of the bound c_idx.
"""
function set_bound!(blmo::MathOptBLMO, c_idx, value) 
    MOI.set(blmo.o, MOI.ConstraintSet(), c_idx, value)
end

"""
Read bound value for c_idx.
"""
function get_bound(blmo, c_idx) 
    return MOI.get(blmo.o, MOI.ConstraintSet(), c_idx)
end

"""
Check if the subject of the bound c_idx is an integer variable (recorded in int_vars).
"""
function is_constraint_on_int_var(blmo::MathOptBLMO, c_idx, int_vars) 
    return c_idx.value in int_vars
end

"""
To check if there is bound for the variable in the global or node bounds.
"""
function is_bound_in(blmo::MathOptBLMO, c_idx, bounds) 
    return haskey(bounds, c_idx.value)
end

"""
Delete bounds.
"""
function delete_bounds!(blmo::MathOptBLMO, cons_delete) 
    for d_idx in cons_delete
        MOI.delete(lmo.o, d_idx)
    end
end

"""
Add bound constraint.
"""
function add_bound_constraint!(blmo::MathOptBLMO, key, value)
    MOI.add_constraint(blmo.o, MOI.VariableIndex(key), value)
end

"""
Check if the bounds were set correctly in build_LMO.
Safety check only.
"""
function build_LMO_correct(blmo, node_bounds)
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
Free model data from previous solve (if necessary).
"""
function free_model(blmo)
    free_model(blmo.o)
end

# cleanup internal SCIP model
function free_model(o::SCIP.Optimizer)
    SCIP.SCIPfreeTransform(o)
end

# no-op by default
function free_model(o::MOI.AbstractOptimizer)   
    return true
end