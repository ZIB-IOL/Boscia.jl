"""
    MILMO

Supertype for the Bounded Integer Linear Minimization Oracles
"""
abstract type BoundedLinearMinimizationOracle end 

"""
Multiple dispatch of FrankWolfe.compute_extreme_point
"""
function compute_extreme_point end 

"""
"""
function optimize! end 

"""
Get list of lower bounds.
"""
function get_lower_bounds_list end

"""
Get list of upper bounds.
"""
function get_upper_bounds_list end 

"""
Get lower bound of variable.
"""
function get_lower_bound end

"""
Get upper bound of variable.
"""
function get_upper_bound end

"""
Set new lower bound.
"""
function set_lower_bound! end

"""
Set new upper bound.
"""
function set_upper_bound! end

"""
Check if the bound constraint is on an integral variable.
"""
function is_constraint_on_int_var end

"""
Is the constraint part of this contraints set. Used to check membership to the global and node bounds.
"""
function is_constraint_in end

"""
Delete constraints that are not needed anymore.
"""
function delete_constraints! end

"""
Add new bound constraint.
"""
function add_constraint! end

"""
Get the variable of the bound constraint.
"""
function get_variable_index(blmo::BoundedLinearMinimizationOracle, c_idx)
    return c_idx
end

"""
Check that all node bounds were set correctly. Not necessary
"""
function check_bounds(blmo::BoundedLinearMinimizationOracle, node_bounds)
    return true
 end

 """
 Check that the problem is feasible, i.e. there are no contradicting constraints, and that it is bounded.
 """
function check_feasibility(blmo::BoundedLinearMinimizationOracle)
    return 
end

