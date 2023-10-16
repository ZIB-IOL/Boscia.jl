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
Build MathOptBLMO from FrankWolfe.MathOptLMO.
"""
function MathOptBLMO(lmo::FrankWolfe.MathOptLMO)
    return MathOptBLMO(lmo.o, lmo.use_modfify)
end

"""
Convert object of Type MathOptLMO into MathOptBLMO and viceversa.
"""
function convert(::Type{MathOptBLMO}, lmo::FrankWolfe.MathOptLMO)
    return MathOptBLMO(lmo.o, lmo.use_modify)
end
function convert(::Type{FrankWolfe.MathOptLMO}, nlmo::MathOptBLMO) 
    return FrankWolfe.MathOptLMO(blmo.o, blmo.use_modfify)
end

################## Necessary to implement ####################
"""
    compute_extreme_point

Is implemented in the FrankWolfe package in file "moi_oracle.jl".
"""

"""
Get list of variables indices. 
If the problem has n variables, they are expected to contiguous and ordered from 1 to n.
"""
function get_list_of_variables(blmo::MathOptBLMO) 
    v_indices = MOI.get(blmo.o, MOI.ListOfVariableIndices())
    if v_indices != MOI.VariableIndex.(1:n)
        error("Variables are expected to be contiguous and ordered from 1 to N")
    end
    return v_indices
end

"""
Get list of binary and integer variables, respectively.
"""
function get_binary_variables(blmo::MathOptBLMO) 
    return MOI.get(blmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.ZeroOne}())
end
function get_integer_variables(blmo::MathOptBLMO) 
    return MOI.get(blmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}())
end 

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
Has variable a binary constraint?
"""
function is_binary_constraint(blmo::MathOptBLMO, idx::Int) 
    consB_list = MOI.get(
        blmo.o,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.ZeroOne}(),
    )
    for c_idx in consB_list
        if c_idx.value == idx
            return true, c_idx
        end
    end
    return false, -1
end

"""
Has variable an integer constraint?
"""
function is_integer_constraint(blmo::MathOptBLMO, idx::Int) 
    consB_list = MOI.get(
        blmo.o,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}(),
    )
    for c_idx in consB_list
        if c_idx.value == idx
            return true, c_idx
        end
    end
    return false, -1
end


##################### Optional to implement ################

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

"""
Check if problem is bounded and feasible, i.e. no contradicting constraints.
"""
function check_feasibility(blmo::BoundedLinearMinimizationOracle)
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
Check whether a split is valid, i.e. the upper and lower on variable vidx are not the same. 
"""
function is_valid_split(tree::Bonobo.BnBTree, blmo::MathOptBLMO, vidx::Int)
    bin_var, _ = is_binary_constraint(tree, vidx)
    int_var, _ = is_integer_constraint(tree, vidx)
    if int_var || bin_var
        l_idx = MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}(vidx)
        u_idx = MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}}(vidx)
        l_bound =
            MOI.is_valid(blmo.o, l_idx) ?
            MOI.get(blmo.o, MOI.ConstraintSet(), l_idx) : nothing
        u_bound =
            MOI.is_valid(blmo.o, u_idx) ?
            MOI.get(blmo.o, MOI.ConstraintSet(), u_idx) : nothing
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
Get solve time, number of nodes and number of simplex iterations.
"""
function get_BLMO_solve_data(blmo::MathOptBLMO)
    opt_times = MOI.get(blmo.o, MOI.SolveTimeSec())
    numberofnodes = MOI.get(blmo.o, MOI.NodeCount())
    simplex_iterations = MOI.get(blmo.o, MOI.SimplexIterations())
    return opt_times, numberofnodes, simplex_iterations
end

"""
Is a given point v linear feasible for the model?
"""
function is_linear_feasible(blmo::MathOptBLMO)
    return is_linear_feasible(blmo.o)
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
        scip_tol = 1e-6
        if o isa SCIP.Optimizer
            scip_tol = MOI.get(o, MOI.RawOptimizerAttribute("numerics/feastol"))
        end
        if dist > 5000.0 * scip_tol
            @debug("Constraint: $(F)-$(S) $(func) = $(val) in $(set)")
            @debug("Distance to set: $(dist)")
            return false
        end
    end
    return true
end

"""
Is a given point v indicator feasible, i.e. meets the indicator constraints? If applicable.
"""
function is_indicator_feasible(blmo::MathOptBLMO, v; atol= 1e-6, rtol=1e-6)
    return is_indicator_feasible(blmo.o, v; atol, rtol)
end
function is_indicator_feasible(o, x; atol = 1e-6, rtol=1e-6)
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
Are indicator constraints present?
"""
function indicator_present(blmo::BoundedLinearMinimizationOracle)
    for (_, S) in MOI.get(blmo.o, MOI.ListOfConstraintTypesPresent())
        if S <: MOI.Indicator
            return true
        end
    end
    return false
end

"""
Get solving tolerance for the BLMO.
"""
function get_tol(blmo::BoundedLinearMinimizationOracle)
    return get_tol(blmo.o)
end
function get_tol(o::SCIP.Optimizer)
    return MOI.get(o, MOI.RawOptimizerAttribute("numerics/feastol"))
end
function get_tol(o::MOI.AbstractOptimizer)
    return 1e-06
end
