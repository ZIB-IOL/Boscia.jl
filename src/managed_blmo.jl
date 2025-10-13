"""
    SimpleBoundableLinearMinimizationOracle

A "simple" LMO that computes the extreme point given a linear objective and the node specific bounds on the integer variables.
Can be stateless since all of the bound management is done by the `ManagedBoundedLMO`.   
"""
abstract type SimpleBoundableLMO <: FrankWolfe.LinearMinimizationOracle end

"""
    bounded_compute_extreme_point

Computes the extreme point given an direction `d`, the current lower and upper bounds on the integer variables, and the set of indices of integer variables.
"""
function bounded_compute_extreme_point end

"""
    is_simple_linear_feasible

Checks whether a given point `v` is satisfying the constraints on the problem.
Note that the bounds on the integer variables are being checked by the ManagedBoundedLMO and do not have to be check here. 
"""
function is_simple_linear_feasible end


"""
    ManagedBoundedLMO{SBLMO<:SimpleBoundableLMO} <: LinearMinimizationOracle

A Bounded Linear Minimization Oracle that manages the bounds.

- `simple_lmo` an LMO of type Simple Boundable LMO.  
- `lower_bounds` list of lower bounds for the integer variables recorded in `int_vars`. If there is no specific lower bound, set corresponding entry to `-Inf`.
- `upper_bounds` list of upper bounds for the integer variables recorded in `int_vars`. If there is no specific upper bound, set corresponding entry to `Inf`.
- `n` total number of variables.
- `int_vars` list of indices of the integer variables.
- `solving_time` the time to evaluate `compute_extreme_point`.
"""
mutable struct ManagedBoundedLMO{SBLMO<:SimpleBoundableLMO} <: LinearMinimizationOracle
    simple_lmo::SBLMO
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    int_vars::Vector{Int}
    n::Int
    solving_time::Float64
end

function ManagedBoundedLMO(simple_lmo, lb, ub, int_vars::Vector{Int}, n::Int)
    if length(lb) != length(ub) || length(ub) != length(int_vars) || length(lb) != length(int_vars)
        error(
            "Supply lower and upper bounds for all integer variables. If there are no explicit bounds, set entry to Inf and -Inf, respectively. The entries have to match the entries of int_vars!",
        )
    end
    # Check that we have integer bounds
    for (i, _) in enumerate(int_vars)
        @assert isapprox(lb[i], round(lb[i]), atol=1e-6, rtol=1e-2)
        @assert isapprox(ub[i], round(ub[i]), atol=1e-6, rtol=1e-2)
    end
    return ManagedBoundedLMO(simple_lmo, lb, ub, int_vars, n, 0.0)
end

#ManagedBoundedLMO(simple_lmo, lb, ub, n, int_vars) = ManagedBoundedLMO(simple_lmo, lb, ub, n, int_vars, 0.0)

# Overload FrankWolfe.compute_extreme_point
function compute_extreme_point(lmo::ManagedBoundedLMO, d; kwargs...)
    time_ref = Dates.now()
    v = bounded_compute_extreme_point(
        lmo.simple_lmo,
        d,
        lmo.lower_bounds,
        lmo.upper_bounds,
        lmo.int_vars,
    )
    lmo.solving_time = float(Dates.value(Dates.now() - time_ref))
    return v
end

function is_decomposition_invariant_oracle(lmo::ManagedBoundedLMO)
    return is_decomposition_invariant_oracle_simple(lmo.simple_lmo)
end

# Provide FrankWolfe.compute_inface_extreme_point
function compute_inface_extreme_point(lmo::ManagedBoundedLMO, direction, x; kwargs...)
    time_ref = Dates.now()
    a = bounded_compute_inface_extreme_point(
        lmo.simple_lmo,
        direction,
        x,
        lmo.lower_bounds,
        lmo.upper_bounds,
        lmo.int_vars,
    )

    lmo.solving_time = float(Dates.value(Dates.now() - time_ref))
    return a
end

# Check if the given point a is on the minimal face of x
function is_inface_feasible(lmo::ManagedBoundedLMO, a, x)
    return is_simple_inface_feasible(
        lmo.simple_lmo,
        a,
        x,
        lmo.lower_bounds,
        lmo.upper_bounds,
        lmo.int_vars,
    )
end

#Provide FrankWolfe.dicg_maximum_step
function dicg_maximum_step(lmo::ManagedBoundedLMO, x, direction; kwargs...)
    return bounded_dicg_maximum_step(
        lmo.simple_lmo,
        x,
        direction,
        lmo.lower_bounds,
        lmo.upper_bounds,
        lmo.int_vars,
    )
end

# Read global bounds from the problem.
function build_global_bounds(lmo::ManagedBoundedLMO, integer_variables)
    global_bounds = IntegerBounds()
    for (idx, int_var) in enumerate(lmo.int_vars)
        push!(global_bounds, (int_var, lmo.lower_bounds[idx]), :greaterthan)
        push!(global_bounds, (int_var, lmo.upper_bounds[idx]), :lessthan)
    end
    return global_bounds
end

## Read information from problem 

# Get list of variables indices. 
# If the problem has n variables, they are expected to contiguous and ordered from 1 to n.
function get_list_of_variables(lmo::ManagedBoundedLMO)
    return lmo.n, collect(1:lmo.n)
end

# Get list of integer variables
function get_integer_variables(lmo::ManagedBoundedLMO)
    return lmo.int_vars
end

# Get the index of the integer variable the bound is working on.
function get_int_var(lmo::ManagedBoundedLMO, cidx)
    return lmo.int_vars[cidx]
end

# Get the list of lower bounds.
function get_lower_bound_list(lmo::ManagedBoundedLMO)
    return collect(1:length(lmo.lower_bounds))
end

# Get the list of upper bounds.
function get_upper_bound_list(lmo::ManagedBoundedLMO)
    return collect(1:length(lmo.upper_bounds))
end

# Read bound value for c_idx.
function get_bound(lmo::ManagedBoundedLMO, c_idx, sense::Symbol)
    if sense == :lessthan
        return lmo.upper_bounds[c_idx]
    elseif sense == :greaterthan
        return lmo.lower_bounds[c_idx]
    else
        error("Allowed value for sense are :lessthan and :greaterthan!")
    end
end

## Changing the bounds constraints.

# Change the value of the bound c_idx.
function set_bound!(lmo::ManagedBoundedLMO, c_idx, value, sense::Symbol)
    if sense == :greaterthan
        lmo.lower_bounds[c_idx] = value
    elseif sense == :lessthan
        lmo.upper_bounds[c_idx] = value
    else
        error("Allowed values for sense are :lessthan and :greaterthan.")
    end
end

# Delete bounds.
function delete_bounds!(lmo::ManagedBoundedLMO, cons_delete)
    for (d_idx, sense) in cons_delete
        if sense == :greaterthan
            lmo.lower_bounds[d_idx] = -Inf
        else
            lmo.upper_bounds[d_idx] = Inf
        end
    end
end

# Add bound constraint.
function add_bound_constraint!(lmo::ManagedBoundedLMO, key, value, sense::Symbol)
    idx = findfirst(x -> x == key, lmo.int_vars)
    if sense == :greaterthan
        lmo.lower_bounds[idx] = value
    elseif sense == :lessthan
        lmo.upper_bounds[idx] = value
    else
        error("Allowed value of sense are :lessthan and :greaterthan!")
    end
end

## Checks

# Check if the subject of the bound c_idx is an integer variable (recorded in int_vars).
function is_constraint_on_int_var(lmo::ManagedBoundedLMO, c_idx, int_vars)
    return lmo.int_vars[c_idx] in int_vars
end

# To check if there is bound for the variable in the global or node bounds.
function is_bound_in(lmo::ManagedBoundedLMO, c_idx, bounds)
    return haskey(bounds, lmo.int_vars[c_idx])
end

# Is a given point v linear feasible for the model?
# That means does v satisfy all bounds and other linear constraints?
function is_linear_feasible(lmo::ManagedBoundedLMO, v::AbstractVector)
    for (i, int_var) in enumerate(lmo.int_vars)
        if !(
            lmo.lower_bounds[i] ≤ v[int_var] + 1e-6 || !(v[int_var] - 1e-6 ≤ lmo.upper_bounds[i])
        )
            @debug(
                "Variable: $(int_var) Vertex entry: $(v[int_var]) Lower bound: $(lmo.lower_bounds[i]) Upper bound: $(lmo.upper_bounds[i]))"
            )
            return false
        end
    end
    return is_simple_linear_feasible(lmo.simple_lmo, v)
end

# Has variable an integer constraint?
function has_integer_constraint(lmo::ManagedBoundedLMO, idx)
    return idx in lmo.int_vars
end


#################### Optional to implement ####################

## Safety Functions

# Check if the bounds were set correctly in build_LMO.
# Safety check only.
function build_LMO_correct(lmo::ManagedBoundedLMO, node_bounds)
    for key in keys(node_bounds.lower_bounds)
        idx = findfirst(x -> x == key, lmo.int_vars)
        if idx === nothing || lmo.lower_bounds[idx] != node_bounds[key, :greaterthan]
            return false
        end
    end
    for key in keys(node_bounds.upper_bounds)
        idx = findfirst(x -> x == key, lmo.int_vars)
        if idx === nothing || lmo.upper_bounds[idx] != node_bounds[key, :lessthan]
            return false
        end
    end
    return true
end

function check_feasibility(lmo::ManagedBoundedLMO)
    for (lb, ub) in zip(lmo.lower_bounds, lmo.upper_bounds)
        if ub < lb
            return INFEASIBLE
        end
    end
    return check_feasibility(
        lmo.simple_lmo,
        lmo.lower_bounds,
        lmo.upper_bounds,
        lmo.int_vars,
        lmo.n,
    )
end

function check_feasibility(simple_lmo::SimpleBoundableLMO, lb, ub, int_vars, n)
    return true
end

# Check whether a split is valid, i.e. the upper and lower on variable vidx are not the same. 
function is_valid_split(tree::Bonobo.BnBTree, lmo::ManagedBoundedLMO, vidx::Int)
    idx = findfirst(x -> x == vidx, lmo.int_vars)
    return lmo.lower_bounds[idx] != lmo.upper_bounds[idx]
end

## Logs
# Get solve time, number of nodes and number of iterations, if applicable.
function get_lmo_solve_data(lmo::ManagedBoundedLMO)
    return lmo.solving_time, 0.0, 0.0
end

# Solve function that just get a SimpleBoundableLMO and builds the corresponding
# ManagedBoundedLMO
function solve(
    f,
    grad!,
    slmo::SimpleBoundableLMO,
    lower_bounds::Vector{Float64},
    upper_bounds::Vector{Float64},
    int_vars::Vector{Int},
    n::Int;
    settings=create_default_settings(),
    kwargs...,
)
    lmo = ManagedBoundedLMO(slmo, lower_bounds, upper_bounds, int_vars, n)
    return solve(f, grad!, lmo, settings=settings, kwargs...)
end
