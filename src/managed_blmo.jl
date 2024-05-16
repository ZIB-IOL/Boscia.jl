"""
    SimpleBoundableLinearMinimizationOracle

A simple LMO that computes the extreme point given the node specific bounds on the integer variables.
Can be stateless since all of the bound management is done by the ManagedBoundedLMO.   
"""
abstract type SimpleBoundableLMO end

"""
Computes the extreme point given an direction d, the current lower and upper bounds on the integer variables, and the set of integer variables.
"""
function bounded_compute_extreme_point end

"""
Checks whether a given point v is satisfying the constraints on the problem.
Note that the bounds on the integer variables are being checked by the ManagedBoundedLMO and do not have to be check here. 
"""
function is_simple_linear_feasible end


"""
    ManagedBoundedLinearMinimizationOracle

A Bounded Linear Minimization Oracle that manages the bounds.

simple_lmo   - An LMO of type Simple Boundable LMO (see above).  
lower_bounds - List of lower bounds for the integer variables recorded in int_vars. If there is no specific lower bound, set corresponding entry to `-Inf`.
upper_bounds - List of upper bounds for the integer variables recorded in int_vars. If there is no specific upper bound, set corresponding entry to `Inf`.
n            - Total number of variables.
int_vars     - List of indices of the integer variables.
solving_time - The time evaluate `compute_extreme_point`.
"""
mutable struct ManagedBoundedLMO{SBLMO<:SimpleBoundableLMO} <: BoundedLinearMinimizationOracle
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

#Provide FrankWolfe.dicg_maximum_step
function dicg_maximum_step(blmo::ManagedBoundedLMO, x, direction; kwargs...)
    return bounded_dicg_maximum_step(
                blmo.simple_lmo,
                x,
                direction,
                blmo.lower_bounds, 
                blmo.upper_bounds, 
                blmo.int_vars,
                )
end

# Provide specific active_set split method for simple_lmo in DICG.
function dicg_split_vertices_set!(blmo::ManagedBoundedLMO, active_set::FrankWolfe.ActiveSet{T,R}, tree, vidx::Int, ::IntegerBounds;kwargs...)where {T,R}
    x = FrankWolfe.get_active_set_iterate(active_set)
    x0_left, x0_right = dicg_split_vertices_set_simple(blmo.simple_lmo, x, vidx)
    as_left = FrankWolfe.ActiveSet([(1.0, x0_left)])
    as_right = FrankWolfe.ActiveSet([(1.0, x0_right)])
    return as_left, as_right
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
            blmo.lower_bounds[i] ≤ v[int_var] + 1e-6 || !(v[int_var] - 1e-6 ≤ blmo.upper_bounds[i])
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
    for (lb,ub) in zip(blmo.lower_bounds, blmo.upper_bounds)
        if ub < lb
            return INFEASIBLE
        end
    end
    return check_feasibility(blmo.simple_lmo, blmo.lower_bounds, blmo.upper_bounds, blmo.int_vars, blmo.n)
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
    traverse_strategy=Bonobo.BestFirstSearch(),
    branching_strategy=Bonobo.MOST_INFEASIBLE(),
    variant::FrankWolfeVariant=BPCG(),
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Adaptive(),
    active_set::Union{Nothing,FrankWolfe.ActiveSet}=nothing,
    lazy=true,
    lazy_tolerance=2.0,
    fw_epsilon=1e-2,
    verbose=false,
    dual_gap=1e-6,
    rel_dual_gap=1.0e-2,
    time_limit=Inf,
    print_iter=100,
    dual_gap_decay_factor=0.8,
    max_fw_iter=10000,
    min_number_lower=Inf,
    min_node_fw_epsilon=1e-6,
    use_postsolve=true,
    min_fw_iterations=5,
    max_iteration_post=10000,
    dual_tightening=true,
    global_dual_tightening=true,
    bnb_callback=nothing,
    strong_convexity=0.0,
    domain_oracle=x -> true,
    start_solution=nothing,
    fw_verbose=false,
    use_shadow_set=true,
    custom_heuristics=[Heuristic()],
    rounding_prob=1.0,
    kwargs...,
)
    blmo = ManagedBoundedLMO(sblmo, lower_bounds, upper_bounds, int_vars, n)
    return solve(
        f,
        grad!,
        blmo,
        traverse_strategy=traverse_strategy,
        branching_strategy=branching_strategy,
        variant=variant,
        line_search=line_search,
        active_set=active_set,
        lazy=lazy,
        lazy_tolerance=lazy_tolerance,
        fw_epsilon=fw_epsilon,
        verbose=verbose,
        dual_gap=dual_gap,
        rel_dual_gap=rel_dual_gap,
        time_limit=time_limit,
        print_iter=print_iter,
        dual_gap_decay_factor=dual_gap_decay_factor,
        max_fw_iter=max_fw_iter,
        min_number_lower=min_number_lower,
        min_node_fw_epsilon=min_node_fw_epsilon,
        use_postsolve=use_postsolve,
        min_fw_iterations=min_fw_iterations,
        max_iteration_post=max_iteration_post,
        dual_tightening=dual_tightening,
        global_dual_tightening=global_dual_tightening,
        bnb_callback=bnb_callback,
        strong_convexity=strong_convexity,
        domain_oracle=domain_oracle,
        start_solution=start_solution,
        fw_verbose=fw_verbose,
        use_shadow_set=use_shadow_set,
        custom_heuristics=custom_heuristics,
        rounding_prob=rounding_prob,
        kwargs...,
    )
end
