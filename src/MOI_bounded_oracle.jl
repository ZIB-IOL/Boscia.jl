"""
    MathOptBLMO{OT<:MOI.AbstractOptimizer} <: BoundedLinearMinimizationOracle

BoundedLinearMinimizationOracle for solvers supporting MathOptInterface.
"""

# Store extra information of solving inface extrem points.
# The keys of MOI_attribute should correspond to specific MOI_attribute names.
mutable struct Inface_point_solve_data
    MOI_attribute::Dict
    function Inface_point_solve_data()
        MOI_attribute = Dict()
        return new(MOI_attribute)
    end
end

struct MathOptBLMO{OT<:MOI.AbstractOptimizer} <: BoundedLinearMinimizationOracle
    o::OT
    use_modify::Bool
    inface_point_solve_data::Inface_point_solve_data
    function MathOptBLMO(o, use_modify=true)
        MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        inface_point_solve_data = Inface_point_solve_data()
        return new{typeof(o)}(o, use_modify, inface_point_solve_data)
    end
end

"""
    MathOptBLMO(lmo::FrankWolfe.MathOptLMO)

Build an instance of `MathOptBLMO` from a `FrankWolfe.MathOptLMO`.
"""
function MathOptBLMO(lmo::FrankWolfe.MathOptLMO)
    return MathOptBLMO(lmo.o, lmo.use_modfify)
end

"""
Convert object of Type `FrankWolfe.MathOptLMO` into `Boscia.MathOptBLMO` and viceversa.
"""
function Base.convert(::Type{MathOptBLMO}, lmo::FrankWolfe.MathOptLMO)
    return MathOptBLMO(lmo.o, lmo.use_modify)
end
function Base.convert(::Type{FrankWolfe.MathOptLMO}, blmo::MathOptBLMO)
    return FrankWolfe.MathOptLMO(blmo.o, blmo.use_modify)
end


################## Necessary to implement ####################
"""
    compute_extreme_point(blmo::MathOptBLMO, d; kwargs...)

Is implemented in the FrankWolfe package in file "moi_oracle.jl".
"""
function compute_extreme_point(blmo::MathOptBLMO, d; kwargs...)
    lmo = convert(FrankWolfe.MathOptLMO, blmo)
    v = FrankWolfe.compute_extreme_point(lmo, d; kwargs)
    @assert blmo isa MathOptBLMO
    return v
end

"""
    get_list_of_variables(blmo::MathOptBLMO)

Get list of variables indices and the total number of variables. 
If the problem has n variables, they are expected to contiguous and ordered from 1 to n.
"""
function get_list_of_variables(blmo::MathOptBLMO)
    v_indices = MOI.get(blmo.o, MOI.ListOfVariableIndices())
    n = length(v_indices)
    if v_indices != MOI.VariableIndex.(1:n)
        error("Variables are expected to be contiguous and ordered from 1 to N")
    end
    return n, v_indices
end

"""
    get_binary_variables(blmo::MathOptBLMO)

Get list of binary variables.
"""
function get_binary_variables(blmo::MathOptBLMO)
    return MOI.get(blmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.ZeroOne}())
end

"""
     get_integer_variables(blmo::MathOptBLMO)

Get list of integer variables.
"""
function get_integer_variables(blmo::MathOptBLMO)
    bin_var = get_binary_variables(blmo)
    int_var = MOI.get(blmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}())
    return vcat(getproperty.(int_var, :value), getproperty.(bin_var, :value))
end

"""
     get_int_var(blmo::MathOptBLMO, c_idx)

Get the index of the integer variable the bound is working on.
"""
function get_int_var(blmo::MathOptBLMO, c_idx)
    return c_idx.value
end

"""
    get_lower_bound_list(blmo::MathOptBLMO)

Get the list of lower bounds.
"""
function get_lower_bound_list(blmo::MathOptBLMO)
    return MOI.get(
        blmo.o,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.GreaterThan{Float64}}(),
    )
end

"""
    get_upper_bound_list(blmo::MathOptBLMO)

Get the list of upper bounds.
"""
function get_upper_bound_list(blmo::MathOptBLMO)
    return MOI.get(blmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.LessThan{Float64}}())
end

"""
    set_bound!(blmo::MathOptBLMO, c_idx, value, sense::Symbol)

Change the value of the bound c_idx.
"""
function set_bound!(blmo::MathOptBLMO, c_idx, value, sense::Symbol)
    if sense == :lessthan
        MOI.set(blmo.o, MOI.ConstraintSet(), c_idx, MOI.LessThan(value))
    elseif sense == :greaterthan
        MOI.set(blmo.o, MOI.ConstraintSet(), c_idx, MOI.GreaterThan(value))
    else
        error("Allowed values for sense are :lessthan and :greaterthan!")
    end
end

"""
    get_bound(blmo::MathOptBLMO, c_idx, sense::Symbol)

Read bound value for c_idx.
"""
function get_bound(blmo::MathOptBLMO, c_idx, sense::Symbol)
    return MOI.get(blmo.o, MOI.ConstraintSet(), c_idx)
end

"""
    is_constraint_on_int_var(blmo::MathOptBLMO, c_idx, int_vars)

Check if the subject of the bound c_idx is an integer variable (recorded in int_vars).
"""
function is_constraint_on_int_var(blmo::MathOptBLMO, c_idx, int_vars)
    return c_idx.value in int_vars
end

"""
    is_bound_in(blmo::MathOptBLMO, c_idx, bounds)

To check if there is bound for the variable in the global or node bounds.
"""
function is_bound_in(blmo::MathOptBLMO, c_idx, bounds)
    return haskey(bounds, c_idx.value)
end

"""
    delete_bounds!(blmo::MathOptBLMO, cons_delete)

Delete bounds.
"""
function delete_bounds!(blmo::MathOptBLMO, cons_delete)
    for (d_idx, _) in cons_delete
        MOI.delete(blmo.o, d_idx)
    end
end

"""
    add_bound_constraint!(blmo::MathOptBLMO, key, value, sense::Symbol)

Add bound constraint.
"""
function add_bound_constraint!(blmo::MathOptBLMO, key, value, sense::Symbol)
    if sense == :lessthan
        MOI.add_constraint(blmo.o, MOI.VariableIndex(key), MOI.LessThan(value))
    elseif sense == :greaterthan
        MOI.add_constraint(blmo.o, MOI.VariableIndex(key), MOI.GreaterThan(value))
    end
end

"""
    has_binary_constraint(blmo::MathOptBLMO, idx::Int)

Has variable a binary constraint?
"""
function has_binary_constraint(blmo::MathOptBLMO, idx::Int)
    consB_list = MOI.get(blmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.ZeroOne}())
    for c_idx in consB_list
        if c_idx.value == idx
            return true, c_idx
        end
    end
    return false, -1
end

"""
    has_integer_constraint(blmo::MathOptBLMO, idx::Int)

Does the variable have an integer constraint?
"""
function has_integer_constraint(blmo::MathOptBLMO, idx::Int)
    consB_list = MOI.get(blmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}())
    for c_idx in consB_list
        if c_idx.value == idx
            return true, c_idx
        end
    end
    return false, -1
end

"""
    is_linear_feasible(blmo::MathOptBLMO, v::AbstractVector)

Is a given point v linear feasible for the model?
"""
function is_linear_feasible(blmo::MathOptBLMO, v::AbstractVector)
    return is_linear_feasible(blmo.o, v)
end
function is_linear_feasible(o::MOI.ModelLike, v::AbstractVector)
    valvar(f) = v[f.value]
    for (F, S) in MOI.get(o, MOI.ListOfConstraintTypesPresent())
        isfeasible = is_linear_feasible_subroutine(o, F, S, valvar)
        if !isfeasible
            return false
        end
    end
    # satisfies all constraints
    return true
end
# function barrier for performance
function is_linear_feasible_subroutine(o::MOI.ModelLike, ::Type{F}, ::Type{S}, valvar) where {F,S}
    if S == MOI.ZeroOne || S <: MOI.Indicator || S == MOI.Integer
        return true
    end
    cons_list = MOI.get(o, MOI.ListOfConstraintIndices{F,S}())
    for c_idx in cons_list
        func = MOI.get(o, MOI.ConstraintFunction(), c_idx)
        val = MOIU.eval_variables(valvar, func)
        set = MOI.get(o, MOI.ConstraintSet(), c_idx)
        # @debug("Constraint: $(F)-$(S) $(func) = $(val) in $(set)")
        dist = MOD.distance_to_set(MOD.DefaultDistance(), val, set)
        solve_tol = get_tol(o)
        if dist > 5000.0 * solve_tol
            @debug("Constraint: $(F)-$(S) $(func) = $(val) in $(set)")
            @debug("Distance to set: $(dist)")
            return false
        end
    end
    return true
end

"""
Is a given point v inface feasible for the model?
"""
function is_inface_feasible(blmo::MathOptBLMO, a::AbstractVector, x::AbstractVector)
    o2 = MOI.instantiate(typeof(blmo.o))
    MOI.copy_to(o2, blmo.o)
    MOI.set(o2, MOI.Silent(), true)
    return is_inface_feasible(o2, a, x)
end
function is_inface_feasible(o::MOI.ModelLike, a::AbstractVector, x::AbstractVector)
    variables = MOI.get(o, MOI.ListOfVariableIndices())
    valvar(f) = x[f.value]
    for (F, S) in MOI.get(o, MOI.ListOfConstraintTypesPresent())
        is_inface_feasible_subroutine(o, F, S, valvar)
    end
    return is_linear_feasible(o, x)
end
function is_inface_feasible_subroutine(
    o::MOI.ModelLike,
    ::Type{F},
    ::Type{S},
    valvar;
    atol=1e-6,
) where {F,S}
    const_list = MOI.get(o, MOI.ListOfConstraintIndices{F,S}())
    for c_idx in const_list
        func = MOI.get(o, MOI.ConstraintFunction(), c_idx)
        val = MOIU.eval_variables(valvar, func)
        set = MOI.get(o, MOI.ConstraintSet(), c_idx)
        if S <: MOI.GreaterThan
            if isapprox(set.lower, val; atol=atol)
                MOI.delete(o, c_idx)
                if F <: MOI.VariableIndex
                    check_cidx = MOI.ConstraintIndex{F,MOI.LessThan{Float64}}(c_idx.value)
                    if MOI.is_valid(o, check_cidx)
                        MOI.delete(o, check_cidx)
                    end
                else
                    func_dict =
                        Dict(field => getfield(func, field) for field in fieldnames(typeof(func)))

                    # Get the list of constraints with same ConstraintFunction but LessThan ConstraintSet.
                    const_list_less =
                        MOI.get(o, MOI.ListOfConstraintIndices{F,MOI.LessThan{Float64}}())

                    # Check if the ConstraintFunction has other ConstraintSet.
                    # If exists, delete the constraint to avoid conflict.
                    for c_idx_less in const_list_less
                        func_less = MOI.get(o, MOI.ConstraintFunction(), c_idx_less)
                        func_less_dict = Dict(
                            field => getfield(func_less, field) for
                            field in fieldnames(typeof(func_less))
                        )
                        if func_less_dict == func_dict
                            MOI.delete(o, c_idx_less)
                            break
                        end
                    end
                end
                MOI.add_constraint(o, func, MOI.EqualTo(set.lower))
            end
        elseif S <: MOI.LessThan
            if isapprox(set.upper, val; atol=atol)
                MOI.delete(o, c_idx)
                if F <: MOI.VariableIndex
                    check_cidx = MOI.ConstraintIndex{F,MOI.GreaterThan{Float64}}(c_idx.value)
                    if MOI.is_valid(o, check_cidx)
                        MOI.delete(o, check_cidx)
                    end
                else
                    func_dict =
                        Dict(field => getfield(func, field) for field in fieldnames(typeof(func)))
                    const_list_greater =
                        MOI.get(o, MOI.ListOfConstraintIndices{F,MOI.GreaterThan{Float64}}())
                    for c_idx_greater in const_list_greater
                        func_greater = MOI.get(o, MOI.ConstraintFunction(), c_idx_greater)
                        func_greater_dict = Dict(
                            field => getfield(func_greater, field) for
                            field in fieldnames(typeof(func_greater))
                        )
                        if func_greater_dict == func_dict
                            MOI.delete(o, c_idx_greater)
                            break
                        end
                    end
                end
                MOI.add_constraint(o, func, MOI.EqualTo(set.upper))
            end
        elseif S <: MOI.Interval
            if isapprox(set.upper, val; atol=atol)
                MOI.delete(o, c_idx)
                MOI.add_constraint(o, func, MOI.EqualTo(set.upper))
            elseif isapprox(set.lower, val; atol=atol)
                MOI.delete(o, c_idx)
                MOI.add_constraint(o, func, MOI.EqualTo(set.lower))
            end
        end
    end
    return true
end

"""
    explicit_bounds_binary_var(blmo::MathOptBLMO, global_bounds::IntegerBounds)

Add explicit bounds for binary variables.
"""
function explicit_bounds_binary_var(blmo::MathOptBLMO, global_bounds::IntegerBounds)
    # adding binary bounds explicitly
    binary_variables = get_binary_variables(blmo)
    for idx in binary_variables
        cidx = MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}}(idx.value)
        if !MOI.is_valid(blmo.o, cidx)
            MOI.add_constraint(blmo.o, MOI.VariableIndex(idx.value), MOI.LessThan(1.0))
        end
        @assert MOI.is_valid(blmo.o, cidx)
        cidx = MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}(idx.value)
        if !MOI.is_valid(blmo.o, cidx)
            MOI.add_constraint(blmo.o, MOI.VariableIndex(idx.value), MOI.GreaterThan(0.0))
        end
        global_bounds[idx.value, :greaterthan] = 0.0
        global_bounds[idx.value, :lessthan] = 1.0
    end
end

"""
    build_global_bounds(blmo::MathOptBLMO, integer_variables)

Read global bounds from the problem
"""
function build_global_bounds(blmo::MathOptBLMO, integer_variables)
    global_bounds = IntegerBounds()
    for idx in integer_variables
        for ST in (MOI.LessThan{Float64}, MOI.GreaterThan{Float64})
            cidx = MOI.ConstraintIndex{MOI.VariableIndex,ST}(idx)
            # Variable constraints to not have to be explicitly given, see Buchheim example
            if MOI.is_valid(blmo.o, cidx)
                s = MOI.get(blmo.o, MOI.ConstraintSet(), cidx)
                if ST == MOI.LessThan{Float64}
                    push!(global_bounds, (idx, s.upper), :lessthan)
                else
                    push!(global_bounds, (idx, s.lower), :greaterthan)
                end
            end
        end
        cidx = MOI.ConstraintIndex{MOI.VariableIndex,MOI.Interval{Float64}}(idx)
        if MOI.is_valid(blmo.o, cidx)
            x = MOI.VariableIndex(idx)
            s = MOI.get(blmo.o, MOI.ConstraintSet(), cidx)
            MOI.delete(blmo.o, cidx)
            MOI.add_constraint(blmo.o, x, MOI.GreaterThan(s.lower))
            MOI.add_constraint(blmo.o, x, MOI.LessThan(s.upper))
            push!(global_bounds, (idx, s.lower), :greaterthan)
            push!(global_bounds, (idx, s.upper), :lessthan)
        end
        @assert !MOI.is_valid(blmo.o, cidx)
    end
    explicit_bounds_binary_var(blmo, global_bounds)
    return global_bounds
end


##################### Optional to implement ################

"""
    build_LMO_correct(blmo, node_bounds)

Check if the bounds were set correctly in build_LMO.
Safety check only.
"""
function build_LMO_correct(blmo, node_bounds)
    for list in (node_bounds.lower_bounds, node_bounds.upper_bounds)
        for (idx, set) in list
            c_idx = MOI.ConstraintIndex{MOI.VariableIndex,typeof(set)}(idx)
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
     free_model(blmo::MathOptBLMO)

Free model data from previous solve (if necessary).
"""
function free_model(blmo::MathOptBLMO)
    return free_model(blmo.o)
end

# no-op by default
function free_model(o::MOI.AbstractOptimizer)
    return true
end

"""
    check_feasibility(blmo::MathOptBLMO)
    
Check if problem is bounded and feasible, i.e. no contradicting constraints.
"""
function check_feasibility(blmo::MathOptBLMO)
    MOI.set(
        blmo.o,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction{Float64}([], 0.0),
    )
    MOI.optimize!(blmo.o)
    status = MOI.get(blmo.o, MOI.TerminationStatus())
    return status
end

"""
    is_valid_split(tree::Bonobo.BnBTree, blmo::MathOptBLMO, vidx::Int)

Check whether a split is valid, i.e. the upper and lower on variable vidx are not the same. 
"""
function is_valid_split(tree::Bonobo.BnBTree, blmo::MathOptBLMO, vidx::Int)
    bin_var, _ = has_binary_constraint(blmo, vidx)
    int_var, _ = has_integer_constraint(blmo, vidx)
    if int_var || bin_var
        l_idx = MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}(vidx)
        u_idx = MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}}(vidx)
        l_bound =
            MOI.is_valid(blmo.o, l_idx) ? MOI.get(blmo.o, MOI.ConstraintSet(), l_idx) : nothing
        u_bound =
            MOI.is_valid(blmo.o, u_idx) ? MOI.get(blmo.o, MOI.ConstraintSet(), u_idx) : nothing
        if (l_bound !== nothing && u_bound !== nothing && l_bound.lower === u_bound.upper)
            @debug l_bound.lower, u_bound.upper
            return false
        else
            return true
        end
    else #!bin_var && !int_var
        @debug "No binary or integer constraint here."
        return true
    end
end

"""
    get_BLMO_solve_data(blmo::MathOptBLMO)

Get solve time, number of nodes and number of simplex iterations.
"""
function get_BLMO_solve_data(blmo::MathOptBLMO)
    if !isempty(blmo.inface_point_solve_data.MOI_attribute)
        opt_times = blmo.inface_point_solve_data.MOI_attribute[MOI.SolveTimeSec()]
        numberofnodes = blmo.inface_point_solve_data.MOI_attribute[MOI.NodeCount()]
        simplex_iterations = blmo.inface_point_solve_data.MOI_attribute[MOI.SimplexIterations()]
        empty!(blmo.inface_point_solve_data.MOI_attribute)
    else
        opt_times = MOI.get(blmo.o, MOI.SolveTimeSec())
        numberofnodes = MOI.get(blmo.o, MOI.NodeCount())
        simplex_iterations = MOI.get(blmo.o, MOI.SimplexIterations())
    end
    return opt_times, numberofnodes, simplex_iterations
end

"""
    is_indicator_feasible(blmo::MathOptBLMO, v; atol=1e-6, rtol=1e-6)

Is a given point v indicator feasible, i.e. meets the indicator constraints? If applicable.
"""
function is_indicator_feasible(blmo::MathOptBLMO, v; atol=1e-6, rtol=1e-6)
    return is_indicator_feasible(blmo.o, v; atol, rtol)
end
function is_indicator_feasible(o, x; atol=1e-6, rtol=1e-6)
    valvar(f) = x[f.value]
    for (F, S) in MOI.get(o, MOI.ListOfConstraintTypesPresent())
        if S <: MOI.Indicator
            cons_list = MOI.get(o, MOI.ListOfConstraintIndices{F,S}())
            for c_idx in cons_list
                func = MOI.get(o, MOI.ConstraintFunction(), c_idx)
                val = MOIU.eval_variables(valvar, func)
                set = MOI.get(o, MOI.ConstraintSet(), c_idx)
                # @debug("Constraint: $(F)-$(S) $(func) = $(val) in $(set)")
                dist = MOD.distance_to_set(MOD.DefaultDistance(), val, set)
                if dist > atol
                    @debug("Constraint: $(F)-$(S) $(func) = $(val) in $(set)")
                    @debug("Distance to set: $(dist)")
                    return false
                end
            end
        end
    end
    return true
end

"""
    indicator_present(blmo::MathOptBLMO) 

Are indicator constraints present?
"""
function indicator_present(blmo::MathOptBLMO)
    for (_, S) in MOI.get(blmo.o, MOI.ListOfConstraintTypesPresent())
        if S <: MOI.Indicator
            return true
        end
    end
    return false
end

"""
     get_tol(blmo::MathOptBLMO)

Get solving tolerance for the BLMO.
"""
function get_tol(blmo::MathOptBLMO)
    return get_tol(blmo.o)
end
function get_tol(o::MOI.AbstractOptimizer)
    return 1e-06
end

"""
    find_best_solution(f::Function, blmo::MathOptBLMO, vars, domain_oracle)

Find best solution from the solving process.
"""
function find_best_solution(
    tree::Bonobo.BnBTree,
    f::Function,
    blmo::MathOptBLMO,
    vars,
    domain_oracle,
)
    return find_best_solution(tree, f, blmo.o, vars, domain_oracle)
end

"""
     function find_best_solution(f::Function, o::MOI.AbstractOptimizer, vars::Vector{MOI.VariableIndex}, domain_oracle,)

Finds the best solution in the Optimizer's solution storage, based on the objective function `f`.
Returns the solution vector and the corresponding best value.
"""
function find_best_solution(
    tree::Bonobo.BnBTree,
    f::Function,
    o::MOI.AbstractOptimizer,
    vars::Vector{MOI.VariableIndex},
    domain_oracle,
)
    nsols = MOI.get(o, MOI.ResultCount())
    @assert nsols > 0
    best_val = Inf
    best_v = nothing
    for sol_idx in 1:nsols
        xv = [MOI.get(o, MOI.VariablePrimal(sol_idx), xi) for xi in vars]
        if domain_oracle(xv)
            val = f(xv)
            if tree.root.options[:add_all_solutions]
                node = tree.nodes[tree.root.current_node_id[]]
                add_new_solution!(tree, node, val, xv, :MIPSolver)
            end
            if val < best_val
                best_val = val
                best_v = xv
            end
        end
    end
    return (best_v, best_val)
end

"""
    get_variables_pointers(blmo, tree)

List of all variable pointers. Depends on how you save your variables internally.
Is used in `find_best_solution`.
"""
function get_variables_pointers(blmo::MathOptBLMO, tree)
    return [MOI.VariableIndex(var) for var in 1:(tree.root.problem.nvars)]
end

"""
     check_infeasible_vertex(blmo::MathOptBLMO, tree)

Deal with infeasible vertex if necessary, e.g. check what caused it etc.
"""
function check_infeasible_vertex(blmo::MathOptBLMO, tree)
    node = tree.nodes[tree.root.current_node_id[]]
    node_bounds = node.local_bounds
    for list in (node_bounds.lower_bounds, node_bounds.upper_bounds)
        for (idx, set) in list
            c_idx = MOI.ConstraintIndex{MOI.VariableIndex,typeof(set)}(idx)
            @assert MOI.is_valid(state.tlmo.blmo.o, c_idx)
            set2 = MOI.get(state.tlmo.blmo.o, MOI.ConstraintSet(), c_idx)
            if !(set == set2)
                MOI.set(tlmo.blmo.o, MOI.ConstraintSet(), c_idx, set)
                set3 = MOI.get(tlmo.blmo.o, MOI.ConstraintSet(), c_idx)
                @assert (set3 == set) "$((idx, set3, set))"
            end
        end
    end
end

"""
    Bonobo.get_branching_variable(tree::Bonobo.BnBTree, branching::PartialStrongBranching{MathOptBLMO{OT}}, node::Bonobo.AbstractNode,) where {OT<:MOI.AbstractOptimizer}

Behavior for strong branching. 
Note that in constrast to the `ManagedBLMO` type, we filter out the integer and binary constraints as solving general MIP in strong branching would be very expensive.
"""
function Bonobo.get_branching_variable(
    tree::Bonobo.BnBTree,
    branching::PartialStrongBranching{MathOptBLMO{OT}},
    node::Bonobo.AbstractNode,
) where {OT<:MOI.AbstractOptimizer}
    xrel = Bonobo.get_relaxed_values(tree, node)
    max_lowerbound = -Inf
    max_idx = -1
    # copy problem and remove integer constraints
    filtered_src = MOI.Utilities.ModelFilter(tree.root.problem.tlmo.blmo.o) do item
        if item isa Tuple
            (_, S) = item
            if S <: Union{MOI.Indicator,MOI.Integer,MOI.ZeroOne}
                return false
            end
        end
        return !(item isa MOI.ConstraintIndex{<:Any,<:Union{MOI.ZeroOne,MOI.Integer,MOI.Indicator}})
    end
    index_map = MOI.copy_to(branching.bounded_lmo.o, filtered_src)
    # sanity check, otherwise the functions need permuted indices
    for (v1, v2) in index_map
        if v1 isa MOI.VariableIndex
            @assert v1 == v2
        end
    end
    relaxed_lmo = MathOptBLMO(branching.bounded_lmo.o)
    @assert !isempty(node.active_set)
    active_set = copy(node.active_set)
    empty!(active_set)
    num_frac = 0
    for idx in Bonobo.get_branching_indices(tree.root)
        if !isapprox(xrel[idx], round(xrel[idx]), atol=tree.options.atol, rtol=tree.options.rtol)
            # left node: x_i <=  floor(̂x_i)
            fxi = floor(xrel[idx])
            # create LMO
            boundsLeft = copy(node.local_bounds)
            if haskey(boundsLeft.upper_bounds, idx)
                delete!(boundsLeft.upper_bounds, idx)
            end
            push!(boundsLeft.upper_bounds, (idx => fxi))
            build_LMO(
                relaxed_lmo,
                tree.root.problem.integer_variable_bounds,
                boundsLeft,
                Bonobo.get_branching_indices(tree.root),
            )
            MOI.optimize!(relaxed_lmo.o)
            #MOI.set(relaxed_lmo.o, MOI.Silent(), false)
            if MOI.get(relaxed_lmo.o, MOI.TerminationStatus()) == MOI.OPTIMAL
                empty!(active_set)
                for (λ, v) in node.active_set
                    if v[idx] <= xrel[idx]
                        push!(active_set, ((λ, v)))
                    end
                end
                @assert !isempty(active_set)
                FrankWolfe.active_set_renormalize!(active_set)
                _, _, primal_relaxed, dual_gap_relaxed, _ =
                    FrankWolfe.blended_pairwise_conditional_gradient(
                        tree.root.problem.f,
                        tree.root.problem.g,
                        relaxed_lmo,
                        active_set,
                        verbose=false,
                        epsilon=branching.solving_epsilon,
                        max_iteration=branching.max_iteration,
                    )
                left_relaxed = primal_relaxed - dual_gap_relaxed
            else
                @debug "Left non-optimal status $(MOI.get(relaxed_lmo.o, MOI.TerminationStatus()))"
                left_relaxed = Inf
            end
            #right node: x_i >=  floor(̂x_i)
            cxi = ceil(xrel[idx])
            boundsRight = copy(node.local_bounds)
            if haskey(boundsRight.lower_bounds, idx)
                delete!(boundsRight.lower_bounds, idx)
            end
            push!(boundsRight.lower_bounds, (idx => cxi))
            build_LMO(
                relaxed_lmo,
                tree.root.problem.integer_variable_bounds,
                boundsRight,
                Bonobo.get_branching_indices(tree.root),
            )
            MOI.optimize!(relaxed_lmo.o)
            if MOI.get(relaxed_lmo.o, MOI.TerminationStatus()) == MOI.OPTIMAL
                empty!(active_set)
                for (λ, v) in node.active_set
                    if v[idx] >= xrel[idx]
                        push!(active_set, (λ, v))
                    end
                end
                if isempty(active_set)
                    @show xrel[idx]
                    @show length(active_set)
                    @info [active_set.atoms[idx] for idx in eachindex(active_set)]
                    error("Empty active set, unreachable")
                end
                FrankWolfe.active_set_renormalize!(active_set)
                _, _, primal_relaxed, dual_gap_relaxed, _ =
                    FrankWolfe.blended_pairwise_conditional_gradient(
                        tree.root.problem.f,
                        tree.root.problem.g,
                        relaxed_lmo,
                        active_set,
                        verbose=false,
                        epsilon=branching.solving_epsilon,
                        max_iteration=branching.max_iteration,
                    )
                right_relaxed = primal_relaxed - dual_gap_relaxed
            else
                @debug "Right non-optimal status $(MOI.get(relaxed_lmo.o, MOI.TerminationStatus()))"
                right_relaxed = Inf
            end
            # lowest lower bound on the two branches
            lowerbound_increase = min(left_relaxed, right_relaxed)
            if lowerbound_increase > max_lowerbound
                max_lowerbound = lowerbound_increase
                max_idx = idx
            end
            num_frac += 1
        end
    end
    @debug "strong branching: index $max_idx, lower bound $max_lowerbound"
    if max_idx <= 0 && num_frac != 0
        error("Infeasible node! Please check constraints! node lb: $(node.lb)")
        max_idx = -1
    end
    if max_idx <= 0
        max_idx = -1
    end
    return max_idx
end


function is_decomposition_invariant_oracle(blmo::MathOptBLMO)
    return true
end

function compute_inface_extreme_point(blmo::MathOptBLMO, direction, x; kwargs...)
    MOI_attribute = Dict()
    MOI_attribute[MOI.SolveTimeSec()] = 0.0
    MOI_attribute[MOI.NodeCount()] = 0.0
    MOI_attribute[MOI.SimplexIterations()] = 0.0
    blmo.inface_point_solve_data.MOI_attribute = MOI_attribute
    lmo = convert(FrankWolfe.MathOptLMO, blmo)
    a = FrankWolfe.compute_inface_extreme_point(
        lmo,
        direction,
        x;
        solve_data=blmo.inface_point_solve_data.MOI_attribute,
        kwargs,
    )
    @assert blmo isa MathOptBLMO
    return a
end

function dicg_maximum_step(blmo::MathOptBLMO, direction, x; kwargs...)
    lmo = convert(FrankWolfe.MathOptLMO, blmo)
    return FrankWolfe.dicg_maximum_step(lmo, direction, x; kwargs...)
end

"""
The `solve`  function receiving a `FrankWolfe.MathOptLMO`. 
Converts the lmo into an instance of `Boscia.MathOptBLMO` and calls the main `solve` function.
"""
function solve(
    f,
    g,
    lmo::FrankWolfe.MathOptLMO;
    mode::Mode=DEFAULT_MODE,
    settings_bnb=settings_bnb(mode=mode),
    settings_frank_wolfe=settings_frank_wolfe(mode=mode),
    settings_tolerances=settings_tolerances(mode=mode),
    settings_postprocessing=settings_postprocessing(mode=mode),
    settings_heuristic=settings_heuristic(mode=mode),
    settings_tightening=settings_tightening(mode=mode),
    settings_domain=settings_domain(mode=mode),
    kwargs...,
)
    blmo = convert(MathOptBLMO, lmo)
    return solve(
        f,
        g,
        blmo;
        settings_bnb=settings_bnb,
        settings_frank_wolfe=settings_frank_wolfe,
        settings_tolerances=settings_tolerances,
        settings_postprocessing=settings_postprocessing,
        settings_heuristic=settings_heuristic,
        settings_tightening=settings_tightening,
        settings_domain=settings_domain,
        kwargs...,
    )
end
