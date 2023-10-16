"""
    BLMO

Supertype for the Bounded Linear Minimization Oracles
"""
abstract type BoundedLinearMinimizationOracle <: FrankWolfe.LinearMinimizationOracle end 

################## Necessary to implement ##################

"""
Implement `FrankWolfe.compute_extreme_point`

Given a direction d solves the problem
    min_x d^T x
where x has to be an integer feasible point
"""
function compute_extreme_point end

"""
Get list of variables indices. 
If the problem has n variables, they are expected to contiguous and ordered from 1 to n.
"""
function get_list_of_variables end

"""
Get list of binary and integer variables, respectively.
"""
function get_binary_variables end
function get_integer_variables end 

"""
Get the index of the integer variable the bound is working on.
"""
function get_int_var end

"""
Get the list of lower bounds.
"""
function get_lower_bound_list end
"""
Get the list of upper bounds.
"""
function get_upper_bound_list end 

"""
Change the value of the bound c_idx.
"""
function set_bound! end

"""
Read bound value for c_idx.
"""
function get_bound end

"""
Check if the subject of the bound c_idx is an integer variable (recorded in int_vars).
"""
function is_constraint_on_int_var end

"""
To check if there is bound for the variable in the global or node bounds.
"""
function is_bound_in end

"""
Delete bounds.
"""
function delete_bounds! end

"""
Add bound constraint.
"""
function add_bound_constraint! end

"""
Has variable a binary constraint?
"""
function is_binary_constraint end

"""
Has variable an integer constraint?
"""
function is_integer_constraint end


#################### Optional to implement ####################

# These are safety check function, performance and log functions, ..
# They are not strictly necessary for Boscia to run but would be beneficial to add, especially in the case of the safety functions.

"""
Check if the bounds were set correctly in build_LMO.
Safety check only.
"""
function build_LMO_correct(blmo::BoundedLinearMinimizationOracle, node_bounds)
    return true
end

"""
Free model data from previous solve (if necessary).
"""
function free_model(blmo::BoundedLinearMinimizationOracle)
    return true
end

"""
Check if problem is bounded and feasible, i.e. no contradicting constraints.
"""
function check_feasibility(blmo::BoundedLinearMinimizationOracle)
    return MOI.OPTIMAL
end

"""
Check whether a split is valid, i.e. the upper and lower on variable vidx are not the same. 
"""
function is_valid_split(tree::Bonobo.BnBTree, blmo::BoundedLinearMinimizationOracle, vidx::Int)
    return true
end

"""
Get solve time, number of nodes and number of iterations, if applicable.
"""
function get_BLMO_solve_data(blmo::BoundedLinearMinimizationOracle)
    return 0.0, 0.0, 0.0
end

"""
Is a given point v linear feasible for the model?
"""
function is_linear_feasible(blmo::BoundedLinearMinimizationOracle)
    return true
end

"""
Is a given point v indicator feasible, i.e. meets the indicator constraints? If applicable.
"""
function is_indicator_feasible(blmo::BoundedLinearMinimizationOracle, v; atol= 1e-6, rtol=1e-6)
    return true
end

"""
Are indicator constraints present?
"""
function indicator_present(blmo::BoundedLinearMinimizationOracle)
    return false
end


"""
Get solving tolerance for the BLMO.
"""
function get_tol(blmo::BoundedLinearMinimizationOracle)
    return 1e-6
end
