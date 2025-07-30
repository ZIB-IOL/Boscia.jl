"""
    BLMO

Supertype for the Bounded Linear Minimization Oracles
"""
abstract type BoundedLinearMinimizationOracle <: FrankWolfe.LinearMinimizationOracle end

"""
Enum encoding the status of the Bounded Linear Minimization Oracle.
Currently available: `OPTIMAL`, `INFEASIBLE` and `UNBOUNDED`.
"""
@enum BLMOStatus begin
    OPTIMAL = 0
    INFEASIBLE = 1
    UNBOUNDED = 2
end

###################################### Necessary to implement ####################################

"""
Implement `FrankWolfe.compute_extreme_point`

Given a direction d solves the problem
    `min_x d^T x`
where x has to be an integer feasible point
"""
function compute_extreme_point end

"""
Read global bounds from the problem.
"""
function build_global_bounds end

## Read information from problem

"""
Get list of variables indices. 
If the problem has n variables, they are expected to contiguous and ordered from 1 to n.
"""
function get_list_of_variables end

"""
Get list of integer variables.
"""
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
Read bound value for c_idx.
"""
function get_bound end


## Changing the bounds constraints.
"""
Change the value of the bound c_idx.
"""
function set_bound! end

"""
Delete bounds.
"""
function delete_bounds! end

"""
Add bound constraint.
"""
function add_bound_constraint! end


## Checks 
"""
Check if the subject of the bound c_idx is an integer variable (recorded in int_vars).
"""
function is_constraint_on_int_var end

"""
To check if there is bound for the variable in the global or node bounds.
"""
function is_bound_in end

"""
Is a given point v linear feasible for the model?
That means does v satisfy all bounds and other linear constraints?
"""
function is_linear_feasible end

"""
Has variable an integer constraint?
"""
function has_integer_constraint end



#################### Optional to implement ####################

# These are safety check, utilities and log functions.
# They are not strictly necessary for Boscia to run but would be beneficial to add, especially in the case of the safety functions.

## Safety Functions
"""
    build_LMO_correct(blmo::BoundedLinearMinimizationOracle, node_bounds)

Check if the bounds were set correctly in build_LMO.
Safety check only.
"""
function build_LMO_correct(blmo::BoundedLinearMinimizationOracle, node_bounds)
    return true
end

"""
    check_feasibility(blmo::BoundedLinearMinimizationOracle)

Check if problem is bounded and feasible, i.e. no contradicting constraints.
"""
function check_feasibility(blmo::BoundedLinearMinimizationOracle)
    return OPTIMAL
end

"""
    is_valid_split(tree::Bonobo.BnBTree, blmo::BoundedLinearMinimizationOracle, vidx::Int)

Check whether a split is valid, i.e. the upper and lower on variable vidx are not the same. 
"""
function is_valid_split(tree::Bonobo.BnBTree, blmo::BoundedLinearMinimizationOracle, vidx::Int)
    return true
end

"""
    is_indicator_feasible(blmo::BoundedLinearMinimizationOracle, v; atol=1e-6, rtol=1e-6)

Is a given point v indicator feasible, i.e. meets the indicator constraints? If applicable.
"""
function is_indicator_feasible(blmo::BoundedLinearMinimizationOracle, v; atol=1e-6, rtol=1e-6)
    return true
end

"""
    indicator_present(blmo::BoundedLinearMinimizationOracle)

Are indicator constraints present?
"""
function indicator_present(blmo::BoundedLinearMinimizationOracle)
    return false
end

"""
    check_infeasible_vertex(blmo::BoundedLinearMinimizationOracle, tree)

Deal with infeasible vertex if necessary, e.g. check what caused it etc.
"""
function check_infeasible_vertex(blmo::BoundedLinearMinimizationOracle, tree) end


## Utility
"""
    free_model(blmo::BoundedLinearMinimizationOracle)

Free model data from previous solve (if necessary).
"""
function free_model(blmo::BoundedLinearMinimizationOracle)
    return true
end

"""
    get_tol(blmo::BoundedLinearMinimizationOracle)

Get solving tolerance for the BLMO.
"""
function get_tol(blmo::BoundedLinearMinimizationOracle)
    return 1e-6
end

"""
    find_best_solution(f::Function, blmo::BoundedLinearMinimizationOracle, vars, domain_oracle)

Find best solution from the solving process.
"""
function find_best_solution(
    tree::Bonobo.BnBTree,
    f::Function,
    blmo::BoundedLinearMinimizationOracle,
    vars,
    domain_oracle,
)
    return (nothing, Inf)
end

"""
    get_variables_pointers(blmo::BoundedLinearMinimizationOracle, tree)

List of all variable pointers. Depends on how you save your variables internally. In the easy case, this is simply `collect(1:N)`.
Is used in `find_best_solution`.
"""
function get_variables_pointers(blmo::BoundedLinearMinimizationOracle, tree)
    N = tree.root.problem.nvars
    return collect(1:N)
end


## Logs
"""
    get_BLMO_solve_data(blmo::BoundedLinearMinimizationOracle)

Get solve time, number of nodes and number of iterations, if applicable.
"""
function get_BLMO_solve_data(blmo::BoundedLinearMinimizationOracle)
    return 0.0, 0.0, 0.0
end

## These DICG-specific functions are essential for Boscia to run with DICG.
"""
Implement `FrankWolfe.is_decomposition_invariant_oracle`

Check if necessary DICG-specific orcales are implemented.
"""
function is_decomposition_invariant_oracle(blmo::BoundedLinearMinimizationOracle)
    return false
end

"""
Is a given point a on the minimal face containing the given x?
"""
function is_inface_feasible(blmo::BoundedLinearMinimizationOracle, a, x)
    return false
end

"""
Implement `FrankWolfe.compute_inface_extreme_point`

Given a direction d and feasible point x solves the problem
    min_a d^T a
where a has to be an integer feasible point and on the minimal face containing x
"""
function compute_inface_extreme_point(blmo::BoundedLinearMinimizationOracle, d, x)
    return error(
        "To use DICG within Boscia, this function has to be implemented for $(typeof(blmo)).",
    )
end

"""
Implement `FrankWolfe.dicg_maximum_step`

Given a direction d and feasible point x solves the problem
    argmax_γ (x - γ * d) ∈ P
where P is feasible set
"""
function dicg_maximum_step(blmo::BoundedLinearMinimizationOracle, d, x)
    return error(
        "To use DICG within Boscia, this function has to be implemented for $(typeof(blmo)).",
    )
end
