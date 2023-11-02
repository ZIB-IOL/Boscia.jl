## How to implement the BLMO Interface using the cube as an example
using Boscia
using Bonobo
using Dates 

"""
    CubeBLMO

A Bounded Linear Minimization Oracle over a cube.
"""
mutable struct CubeBLMO <: Boscia.BoundedLinearMinimizationOracle
    n::Int
    int_vars::Vector{Int}
    bounds::Boscia.IntegerBounds
    solving_time::Float64
end

CubeBLMO(n, int_vars, bounds) = CubeBLMO(n, int_vars, bounds, 0.0)

## Necessary

# computing an extreme point for the cube amounts to checking the sign of the gradient
function Boscia.compute_extreme_point(blmo::CubeBLMO, d; kwargs...)
    time_ref = Dates.now()
    v = zeros(length(d))
    for i in eachindex(d)
        v[i] = d[i] > 0 ? blmo.bounds[i, :greaterthan] : blmo.bounds[i, :lessthan]
    end
    blmo.solving_time = float(Dates.value(Dates.now() - time_ref))
    return v
end

##

function Boscia.build_global_bounds(blmo::CubeBLMO, integer_variables)
    global_bounds = Boscia.IntegerBounds()
    for i in 1:blmo.n
        if i in integer_variables
            push!(global_bounds, (i, blmo.bounds[i, :lessthan]), :lessthan)
            push!(global_bounds, (i, blmo.bounds[i, :greaterthan]), :greaterthan)
        end
    end
    return global_bounds
end


## Read information from problem
function Boscia.get_list_of_variables(blmo::CubeBLMO)
    return blmo.n, collect(1:blmo.n)
end

# Get list of integer variables, respectively.
function Boscia.get_integer_variables(blmo::CubeBLMO)
    return blmo.int_vars
end

function Boscia.get_int_var(blmo::CubeBLMO, cidx)
    return cidx
end

function Boscia.get_lower_bound_list(blmo::CubeBLMO)
    return keys(blmo.bounds.lower_bounds)
end

function Boscia.get_upper_bound_list(blmo::CubeBLMO)
    return keys(blmo.bounds.upper_bounds)
end

function Boscia.get_bound(blmo::CubeBLMO, c_idx, sense::Symbol)
    @assert sense == :lessthan || sense == :greaterthan
    return blmo[c_idx, sense]
end

## Changing the bounds constraints.
function Boscia.set_bound!(blmo::CubeBLMO, c_idx, value, sense::Symbol)
    if sense == :greaterthan
        blmo.bounds.lower_bounds[c_idx] = value
    elseif sense == :lessthan
        blmo.bounds.upper_bounds[c_idx] = value
    else
        error("Allowed values for sense are :lessthan and :greaterthan.")
    end
end

function Boscia.delete_bounds!(blmo::CubeBLMO, cons_delete)
    # For the cube this shouldn't happen! Otherwise we get unbounded!
    if !isempty(cons_delete)
        error("Trying to delete bounds of the cube!")
    end
end

function Boscia.add_bound_constraint!(blmo::CubeBLMO, key, value, sense::Symbol)
    # Should not be necessary
    return error("Trying to add bound constraints of the cube!")
end

## Checks

function Boscia.is_constraint_on_int_var(blmo::CubeBLMO, c_idx, int_vars)
    return c_idx in int_vars
end

function Boscia.is_bound_in(blmo::CubeBLMO, c_idx, bounds)
    return haskey(bounds, c_idx)
end

function Boscia.is_linear_feasible(blmo::CubeBLMO, v::AbstractVector)
    for i in eachindex(v)
        if !(
            blmo.bounds[i, :greaterthan] ≤ v[i] + 1e-6 ||
            !(v[i] - 1e-6 ≤ blmo.bounds[i, :lessthan])
        )
            @debug(
                "Vertex entry: $(v[i]) Lower bound: $(blmo.bounds[i, :greaterthan]) Upper bound: $(blmo.bounds[i, :lessthan]))"
            )
            return false
        end
    end
    return true
end

function Boscia.has_integer_constraint(blmo::CubeBLMO, idx)
    return idx in blmo.int_vars
end


###################### Optional
## Safety Functions

function Boscia.build_LMO_correct(blmo::CubeBLMO, node_bounds)
    for key in keys(node_bounds.lower_bounds)
        if !haskey(blmo.bounds, (key, :greaterthan)) ||
           blmo.bounds[key, :greaterthan] != node_bounds[key, :greaterthan]
            return false
        end
    end
    for key in keys(node_bounds.upper_bounds)
        if !haskey(blmo.bounds, (key, :lessthan)) ||
           blmo.bounds[key, :lessthan] != node_bounds[key, :lessthan]
            return false
        end
    end
    return true
end

function Boscia.check_feasibility(blmo::CubeBLMO)
    for i in 1:blmo.n
        if !haskey(blmo.bounds, (i, :greaterthan)) || !haskey(blmo.bounds, (i, :lessthan))
            return Boscia.UNBOUNDED
        end
        if blmo.bounds[i, :greaterthan] > blmo.bounds[i, :lessthan]
            return Boscia.INFEASIBLE
        end
    end
    return Boscia.OPTIMAL
end

function Boscia.is_valid_split(tree::Bonobo.BnBTree, blmo::CubeBLMO, vidx::Int)
    return blmo.bounds[vidx, :lessthan] != blmo.bounds[vidx, :greaterthan]
end

## Logs
function Boscia.get_BLMO_solve_data(blmo::CubeBLMO)
    return blmo.solving_time, 0.0, 0.0
end