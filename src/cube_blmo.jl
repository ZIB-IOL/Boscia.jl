"""
    CubeBLMO
 
A Bounded Linear Minimization Oracle over a cube.     
"""
mutable struct CubeBLMO <: Boscia.BoundedLinearMinimizationOracle
    n::Int
    int_vars::Vector{Int}
    bin_vars::Vector{Int}
    bounds::Boscia.IntegerBounds
    solving_time
end

CubeBLMO(n, int_vars, bin_vars, bounds) = CubeBLMO(n, int_vars, bin_vars, bounds, 0.0)

## Necessary
"""
Implement `FrankWolfe.compute_extreme_point`

Given a direction d solves the problem
    min_x d^T x
where x has to be an integer feasible point
"""
function Boscia.compute_extreme_point(blmo::CubeBLMO, d; kwargs...)
    time_ref = Dates.now()
    v = zeros(length(d))
    for i in eachindex(d)
        v[i] = d[i] > 0 ? blmo.bounds[i, :greaterthan].lower : blmo.bounds[i, :lessthan].upper
    end 
    blmo.solving_time = float(Dates.value(Dates.now() - time_ref))
    return v
end

## 
"""
Read global bounds from the problem.
"""
function Boscia.build_global_bounds(blmo::CubeBLMO, integer_variables)
    global_bounds = Boscia.IntegerBounds()
    for i in 1:blmo.n
        if i in integer_variables
            push!(global_bounds, (i, blmo.bounds[i, :lessthan]))
            push!(global_bounds, (i, blmo.bounds[i, :greaterthan]))
        end
    end
    return global_bounds
end
"""
Add explicit bounds for binary variables, if not already done from the get-go.
"""
function Boscia.explicit_bounds_binary_var(blmo::CubeBLMO, gb::Boscia.IntegerBounds, binary_vars) 
    nothing
end

## Read information from problem
"""
Get list of variables indices. 
If the problem has n variables, they are expected to contiguous and ordered from 1 to n.
"""
function Boscia.get_list_of_variables(blmo::CubeBLMO)
    return blmo.n, collect(1:blmo.n)
 end
"""
Get list of binary and integer variables, respectively.
"""
function Boscia.get_binary_variables(blmo::CubeBLMO)
    return blmo.bin_vars
end
function Boscia.get_integer_variables(blmo::CubeBLMO) 
    return blmo.int_vars
end 
"""
Get the index of the integer variable the bound is working on.
"""
function Boscia.get_int_var(blmo::CubeBLMO, cidx) 
    return cidx
end
"""
Get the list of lower bounds.
"""
function Boscia.get_lower_bound_list(blmo::CubeBLMO) 
    return keys(blmo.bounds.lower_bounds)
end
"""
Get the list of upper bounds.
"""
function Boscia.get_upper_bound_list(blmo::CubeBLMO) 
    return keys(blmo.bounds.upper_bounds)
end 
"""
Read bound value for c_idx.
"""
function Boscia.get_lower_bound(blmo::CubeBLMO, c_idx)
    return blmo.bounds[c_idx, :greaterthan]
end
function Boscia.get_upper_bound(blmo::CubeBLMO, c_idx)
    return blmo.bounds[c_idx, :lessthan]
end

## Changing the bounds constraints.
"""
Change the value of the bound c_idx.
"""
function Boscia.set_bound!(blmo::CubeBLMO, c_idx, value) 
    if value isa MOI.GreaterThan{Float64}
        blmo.bounds.lower_bounds[c_idx] = value
    elseif value isa MOI.LessThan{Float64}
        blmo.bounds.upper_bounds[c_idx] = value
    else
        error("We expect the value to be of type MOI.GreaterThan or Moi.LessThan!")
    end
end
"""
Delete bounds.
"""
function Boscia.delete_bounds!(blmo::CubeBLMO, cons_delete) 
    # For the cube this shouldn't happen! Otherwise we get unbounded!
    if !isempty(cons_delete)
        error("Trying to delete bounds of the cube!")
    end
end
"""
Add bound constraint.
"""
function Boscia.add_bound_constraint!(blmo::CubeBLMO, key, value) 
    # Should not be necessary
    error("Trying to add bound constraints of the cube!")
end

## Checks 
"""
Check if the subject of the bound c_idx is an integer variable (recorded in int_vars).
"""
function Boscia.is_constraint_on_int_var(blmo::CubeBLMO, c_idx, int_vars) 
    return c_idx in int_vars
end
"""
To check if there is bound for the variable in the global or node bounds.
"""
function Boscia.is_bound_in(blmo::CubeBLMO, c_idx, bounds) 
    return haskey(bounds, c_idx)
end
"""
Is a given point v linear feasible for the model?
That means does v satisfy all bounds and other linear constraints?
"""
function Boscia.is_linear_feasible(blmo::CubeBLMO, v::AbstractVector) 
    for i in eachindex(v)
        if !(blmo.bounds[i, :greaterthan].lower ≤ v[i] + 1e-6 || !(v[i] - 1e-6 ≤ blmo.bounds[i, :lessthan].upper))
            @debug("Vertex entry: $(v[i]) Lower bound: $(blmo.bounds[i, :greaterthan].lower) Upper bound: $(blmo.bounds[i, :lessthan].upper))")
            return false
        end
    end
    return true
end
"""
Does the variable have a binary constraint?
"""
function Boscia.has_binary_constraint(blmo::CubeBLMO, idx) 
    return idx in blmo.int_vars
end
"""
Has variable an integer constraint?
"""
function Boscia.has_integer_constraint(blmo::CubeBLMO, idx) 
    return idx in blmo.bin_vars
end



###################### Optional
## Safety Functions
"""
Check if the bounds were set correctly in build_LMO.
Safety check only.
"""
function Boscia.build_LMO_correct(blmo::CubeBLMO, node_bounds)
    for key in keys(node_bounds.lower_bounds)
        if !haskey(blmo.bounds, (key, :greaterthan)) || blmo.bounds[key, :greaterthan] != node_bounds[key, :greaterthan]
            return false
        end
    end
    for key in keys(node_bounds.upper_bounds)
        if !haskey(blmo.bounds, (key, :lessthan)) || blmo.bounds[key, :lessthan] != node_bounds[key, :lessthan]
            return false
        end
    end  
    return true
end

"""
Check if problem is bounded and feasible, i.e. no contradicting constraints.
"""
function Boscia.check_feasibility(blmo::CubeBLMO)
    for i in 1:blmo.n 
        if !haskey(blmo.bounds, (i, :greaterthan)) || !haskey(blmo.bounds, (i, :lessthan))
            return MOI.DUAL_INFEASIBLE
        end
    end
    return MOI.OPTIMAL
end

"""
Check whether a split is valid, i.e. the upper and lower on variable vidx are not the same. 
"""
function Boscia.is_valid_split(tree::Bonobo.BnBTree, blmo::CubeBLMO, vidx::Int)
    return blmo.bounds[vidx, :lessthan] != blmo.bounds[vidx, :greaterthan]
end

## Logs
"""
Get solve time, number of nodes and number of iterations, if applicable.
"""
function Boscia.get_BLMO_solve_data(blmo::CubeBLMO)
    return blmo.solving_time, 0.0, 0.0
end
