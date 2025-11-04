"""
    SimpleBoundableLinearMinimizationOracle

A "simple" BLMO that computes the extreme point given a linear objective and the node specific bounds on the integer variables.
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
    ManagedBoundedLMO{SBLMO<:SimpleBoundableLMO} <: BoundedLinearMinimizationOracle

A Bounded Linear Minimization Oracle that manages the bounds.

- `simple_lmo` an LMO of type Simple Boundable LMO.  
- `lower_bounds` list of lower bounds for the integer variables recorded in `int_vars`. If there is no specific lower bound, set corresponding entry to `-Inf`.
- `upper_bounds` list of upper bounds for the integer variables recorded in `int_vars`. If there is no specific upper bound, set corresponding entry to `Inf`.
- `n` total number of variables.
- `int_vars` list of indices of the integer variables.
- `solving_time` the time to evaluate `compute_extreme_point`.
"""
mutable struct ManagedBoundedLMO{SBLMO<:SimpleBoundableLMO} <:
               BoundedLinearMinimizationOracle
    simple_lmo::SBLMO
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    int_vars::Vector{Int}
    n::Int
    solving_time::Float64
end

function ManagedBoundedLMO(simple_lmo, lb, ub, int_vars::Vector{Int}, n::Int)
    if length(lb) != length(ub) ||
       length(ub) != length(int_vars) ||
       length(lb) != length(int_vars)
        error(
            "Supply lower and upper bounds for all integer variables. If there are no explicit bounds, set entry to Inf and -Inf, respectively. The entries have to match the entries of int_vars!",
        )
    end
    # Check that we have integer bounds
    for (i, _) in enumerate(int_vars)
        @assert isapprox(lb[i], round(lb[i]), atol = 1e-6, rtol = 1e-2)
        @assert isapprox(ub[i], round(ub[i]), atol = 1e-6, rtol = 1e-2)
    end
    return ManagedBoundedLMO(simple_lmo, lb, ub, int_vars, n, 0.0)
end

#ManagedBoundedLMO(simple_lmo, lb, ub, n, int_vars) = ManagedBoundedLMO(simple_lmo, lb, ub, n, int_vars, 0.0)

# Overload FrankWolfe.compute_extreme_point
function compute_extreme_point(blmo::ManagedBoundedLMO, d; kwargs...)
    time_ref = Dates.now()
    v = bounded_compute_extreme_point(
        blmo.simple_lmo,
        d,
        blmo.lower_bounds,
        blmo.upper_bounds,
        blmo.int_vars,
    )
    blmo.solving_time = float(Dates.value(Dates.now() - time_ref))
    return v
end

function is_decomposition_invariant_oracle(blmo::ManagedBoundedLMO)
    return is_decomposition_invariant_oracle_simple(blmo.simple_lmo)
end

# Provide FrankWolfe.compute_inface_extreme_point
function compute_inface_extreme_point(blmo::ManagedBoundedLMO, direction, x; kwargs...)
    time_ref = Dates.now()
    a = bounded_compute_inface_extreme_point(
        blmo.simple_lmo,
        direction,
        x,
        blmo.lower_bounds,
        blmo.upper_bounds,
        blmo.int_vars,
    )

    blmo.solving_time = float(Dates.value(Dates.now() - time_ref))
    return a
end

# Check if the given point a is on the minimal face of x
function is_inface_feasible(blmo::ManagedBoundedLMO, a, x)
    return is_simple_inface_feasible(
        blmo.simple_lmo,
        a,
        x,
        blmo.lower_bounds,
        blmo.upper_bounds,
        blmo.int_vars,
    )
end

#Provide FrankWolfe.dicg_maximum_step
function dicg_maximum_step(blmo::ManagedBoundedLMO, direction, x; kwargs...)
    return bounded_dicg_maximum_step(
        blmo.simple_lmo,
        direction,
        x,
        blmo.lower_bounds,
        blmo.upper_bounds,
        blmo.int_vars,
    )
end

# Read global bounds from the problem.
function build_global_bounds(blmo::ManagedBoundedLMO, integer_variables)
    global_bounds = IntegerBounds()
    for (idx, int_var) in enumerate(blmo.int_vars)
        push!(global_bounds, (int_var, blmo.lower_bounds[idx]), :greaterthan)
        push!(global_bounds, (int_var, blmo.upper_bounds[idx]), :lessthan)
    end
    return global_bounds
end

## Read information from problem 

# Get list of variables indices. 
# If the problem has n variables, they are expected to contiguous and ordered from 1 to n.
function get_list_of_variables(blmo::ManagedBoundedLMO)
    return blmo.n, collect(1:blmo.n)
end

# Get list of integer variables
function get_integer_variables(blmo::ManagedBoundedLMO)
    return blmo.int_vars
end

# Get the index of the integer variable the bound is working on.
function get_int_var(blmo::ManagedBoundedLMO, cidx)
    return blmo.int_vars[cidx]
end

# Get the list of lower bounds.
function get_lower_bound_list(blmo::ManagedBoundedLMO)
    return collect(1:length(blmo.lower_bounds))
end

# Get the list of upper bounds.
function get_upper_bound_list(blmo::ManagedBoundedLMO)
    return collect(1:length(blmo.upper_bounds))
end

# Read bound value for c_idx.
function get_bound(blmo::ManagedBoundedLMO, c_idx, sense::Symbol)
    if sense == :lessthan
        return blmo.upper_bounds[c_idx]
    elseif sense == :greaterthan
        return blmo.lower_bounds[c_idx]
    else
        error("Allowed value for sense are :lessthan and :greaterthan!")
    end
end

## Changing the bounds constraints.

# Change the value of the bound c_idx.
function set_bound!(blmo::ManagedBoundedLMO, c_idx, value, sense::Symbol)
    if sense == :greaterthan
        blmo.lower_bounds[c_idx] = value
    elseif sense == :lessthan
        blmo.upper_bounds[c_idx] = value
    else
        error("Allowed values for sense are :lessthan and :greaterthan.")
    end
end

# Delete bounds.
function delete_bounds!(blmo::ManagedBoundedLMO, cons_delete)
    for (d_idx, sense) in cons_delete
        if sense == :greaterthan
            blmo.lower_bounds[d_idx] = -Inf
        else
            blmo.upper_bounds[d_idx] = Inf
        end
    end
end

# Add bound constraint.
function add_bound_constraint!(blmo::ManagedBoundedLMO, key, value, sense::Symbol)
    idx = findfirst(x -> x == key, blmo.int_vars)
    if sense == :greaterthan
        blmo.lower_bounds[idx] = value
    elseif sense == :lessthan
        blmo.upper_bounds[idx] = value
    else
        error("Allowed value of sense are :lessthan and :greaterthan!")
    end
end

## Checks

# Check if the subject of the bound c_idx is an integer variable (recorded in int_vars).
function is_constraint_on_int_var(blmo::ManagedBoundedLMO, c_idx, int_vars)
    return blmo.int_vars[c_idx] in int_vars
end

# To check if there is bound for the variable in the global or node bounds.
function is_bound_in(blmo::ManagedBoundedLMO, c_idx, bounds)
    return haskey(bounds, blmo.int_vars[c_idx])
end

# Is a given point v linear feasible for the model?
# That means does v satisfy all bounds and other linear constraints?
function is_linear_feasible(blmo::ManagedBoundedLMO, v::AbstractVector)
    for (i, int_var) in enumerate(blmo.int_vars)
        if !(
            blmo.lower_bounds[i] ≤ v[int_var] + 1e-6 &&
            (v[int_var] - 1e-6 ≤ blmo.upper_bounds[i])
        )
            @debug(
                "Variable: $(int_var) Vertex entry: $(v[int_var]) Lower bound: $(blmo.lower_bounds[i]) Upper bound: $(blmo.upper_bounds[i]))"
            )
            return false
        end
    end
    return is_simple_linear_feasible(blmo.simple_lmo, v)
end

# Has variable an integer constraint?
function has_integer_constraint(blmo::ManagedBoundedLMO, idx)
    return idx in blmo.int_vars
end


#################### Optional to implement ####################

## Safety Functions

# Check if the bounds were set correctly in build_LMO.
# Safety check only.
function build_LMO_correct(blmo::ManagedBoundedLMO, node_bounds)
    for key in keys(node_bounds.lower_bounds)
        idx = findfirst(x -> x == key, blmo.int_vars)
        if idx === nothing || blmo.lower_bounds[idx] != node_bounds[key, :greaterthan]
            return false
        end
    end
    for key in keys(node_bounds.upper_bounds)
        idx = findfirst(x -> x == key, blmo.int_vars)
        if idx === nothing || blmo.upper_bounds[idx] != node_bounds[key, :lessthan]
            return false
        end
    end
    return true
end

function check_feasibility(blmo::ManagedBoundedLMO)
    for (lb, ub) in zip(blmo.lower_bounds, blmo.upper_bounds)
        if ub < lb
            return INFEASIBLE
        end
    end
    return check_feasibility(
        blmo.simple_lmo,
        blmo.lower_bounds,
        blmo.upper_bounds,
        blmo.int_vars,
        blmo.n,
    )
end

function check_feasibility(simple_lmo::SimpleBoundableLMO, lb, ub, int_vars, n)
    return true
end

# Check whether a split is valid, i.e. the upper and lower on variable vidx are not the same. 
function is_valid_split(tree::Bonobo.BnBTree, blmo::ManagedBoundedLMO, vidx::Int)
    idx = findfirst(x -> x == vidx, blmo.int_vars)
    return blmo.lower_bounds[idx] != blmo.upper_bounds[idx]
end

## Logs
# Get solve time, number of nodes and number of iterations, if applicable.
function get_BLMO_solve_data(blmo::ManagedBoundedLMO)
    return blmo.solving_time, 0.0, 0.0
end

# Solve function that just get a SimpleBoundableLMO and builds the corresponding
# ManagedBoundedLMO
function solve(
    f,
    grad!,
    sblmo::SimpleBoundableLMO,
    lower_bounds::Vector{Float64},
    upper_bounds::Vector{Float64},
    int_vars::Vector{Int},
    n::Int;
    settings = create_default_settings(),
    kwargs...,
)
    blmo = ManagedBoundedLMO(sblmo, lower_bounds, upper_bounds, int_vars, n)
    return solve(f, grad!, blmo, settings = settings, kwargs...)
end
