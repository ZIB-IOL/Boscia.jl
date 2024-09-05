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
function Base.convert(::Type{MathOptBLMO}, lmo::FrankWolfe.MathOptLMO)
    return MathOptBLMO(lmo.o, lmo.use_modify)
end
function Base.convert(::Type{FrankWolfe.MathOptLMO}, blmo::MathOptBLMO)
    return FrankWolfe.MathOptLMO(blmo.o, blmo.use_modify)
end


################## Necessary to implement ####################
"""
    compute_extreme_point

Is implemented in the FrankWolfe package in file "moi_oracle.jl".
"""
function compute_extreme_point(blmo::MathOptBLMO, d; kwargs...)
    lmo = convert(FrankWolfe.MathOptLMO, blmo)
    v = FrankWolfe.compute_extreme_point(lmo, d; kwargs)
    @assert blmo isa MathOptBLMO
    return v
end

"""
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
Get list of binary and integer variables, respectively.
"""
function get_binary_variables(blmo::MathOptBLMO)
    return MOI.get(blmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.ZeroOne}())
end
function get_integer_variables(blmo::MathOptBLMO)
    bin_var = get_binary_variables(blmo)
    int_var = MOI.get(blmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}())
    return vcat(getproperty.(int_var, :value), getproperty.(bin_var, :value))
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
    return MOI.get(
        blmo.o,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.GreaterThan{Float64}}(),
    )
end

"""
Get the list of upper bounds.
"""
function get_upper_bound_list(blmo::MathOptBLMO)
    return MOI.get(blmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.LessThan{Float64}}())
end

"""
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
Read bound value for c_idx.
"""
function get_bound(blmo::MathOptBLMO, c_idx, sense::Symbol)
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
    for (d_idx, _) in cons_delete
        MOI.delete(blmo.o, d_idx)
    end
end

"""
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
Get solve time, number of nodes and number of simplex iterations.
"""
function get_BLMO_solve_data(blmo::MathOptBLMO)
    opt_times = MOI.get(blmo.o, MOI.SolveTimeSec())
    numberofnodes = MOI.get(blmo.o, MOI.NodeCount())
    simplex_iterations = MOI.get(blmo.o, MOI.SimplexIterations())
    return opt_times, numberofnodes, simplex_iterations
end

"""
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
Get solving tolerance for the BLMO.
"""
function get_tol(blmo::MathOptBLMO)
    return get_tol(blmo.o)
end
function get_tol(o::MOI.AbstractOptimizer)
    return 1e-06
end

"""
Find best solution from the solving process.
"""
function find_best_solution(f::Function, blmo::MathOptBLMO, vars, domain_oracle)
    return find_best_solution(f, blmo.o, vars, domain_oracle)
end

"""
Finds the best solution in the Optimizer's solution storage, based on the objective function `f`.
Returns the solution vector and the corresponding best value.
"""
function find_best_solution(
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
            if val < best_val
                best_val = val
                best_v = xv
            end
        end
    end
    return (best_v, best_val)
end

"""
List of all variable pointers. Depends on how you save your variables internally.

Is used in `find_best_solution`.
"""
function get_variables_pointers(blmo::MathOptBLMO, tree)
    return [MOI.VariableIndex(var) for var in 1:(tree.root.problem.nvars)]
end

"""
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
Behavior for strong branching.
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

"""
Solve function in case of MathOptLMO. 
Converts the lmo into a MathOptBLMO and calls the solve function below.
"""
function solve(
    f,
    g,
    lmo::FrankWolfe.MathOptLMO;
    traverse_strategy=Bonobo.BestFirstSearch(),
    branching_strategy=Bonobo.MOST_INFEASIBLE(),
    variant::FrankWolfeVariant=BPCG(),
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Adaptive(),
    active_set::Union{Nothing,FrankWolfe.ActiveSet}=nothing,
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
    clean_solutions=false, 
    max_clean_iter=10,
    kwargs...,
)
    blmo = convert(MathOptBLMO, lmo)
    return solve(
        f,
        g,
        blmo;
        traverse_strategy=traverse_strategy,
        branching_strategy=branching_strategy,
        variant=variant,
        line_search=line_search,
        active_set=active_set,
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
        clean_solutions=clean_solutions,
        max_clean_iter=max_clean_iter,
        kwargs...,
    )
end
