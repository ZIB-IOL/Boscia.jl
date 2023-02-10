mutable struct FrankWolfeSolution{Node<:Bonobo.AbstractNode,Value} <:
               Bonobo.AbstractSolution{Node,Value}
    objective::Float64
    solution::Value
    node::Node
    source::Symbol
end

"""
    AbtractFrankWolfeNode <: Bonobo.AbstractNode 
"""
abstract type AbstractFrankWolfeNode <: Bonobo.AbstractNode end

"""
    FrankWolfeNode <: AbstractFrankWolfeNode

A node in the branch-and-bound tree storing information for a Frank-Wolfe subproblem.

`std` stores the id, lower and upper bound of the node.
`active_set` store the active set structure.
`local_bounds` instead of storing the complete LMO, it just stores the bounds specific to THIS node.
    All other integer bounds are stored in the root.
'level' stores the level in the tree
'fw_dual_gap_limit' set the tolerance for the dual gap in the FW algorithms
"""
mutable struct FrankWolfeNode{
    AT<:FrankWolfe.ActiveSet,
    DVS<:FrankWolfe.DeletedVertexStorage,
    IB<:IntegerBounds,
} <: AbstractFrankWolfeNode
    std::Bonobo.BnBNodeInfo
    active_set::AT
    discarded_vertices::DVS
    local_bounds::IB
    level::Int
    fw_dual_gap_limit::Float64
    fw_time::Millisecond
end

"""
Create the information of the new branching nodes 
based on their parent and the index of the branching variable
"""
function Bonobo.get_branching_nodes_info(tree::Bonobo.BnBTree, node::FrankWolfeNode, vidx::Int)
    if !is_valid_split(tree, vidx)
        error("Splitting on the same index as parent! Abort!")
    end

    # update splitting index
    x = Bonobo.get_relaxed_values(tree, node)

    # split active set
    active_set_left, active_set_right = split_vertices_set!(node.active_set, tree, vidx, node.local_bounds)
    discarded_set_left, discarded_set_right =
        split_vertices_set!(node.discarded_vertices, tree, vidx, x, node.local_bounds)

    @assert isapprox(sum(active_set_left.weights), 1.0)
    @assert sum(active_set_left.weights .< 0) == 0
    @assert isapprox(sum(active_set_right.weights), 1.0)
    @assert sum(active_set_right.weights .< 0) == 0

    # add new bounds to the feasible region left and right
    # copy bounds from parent
    varbounds_left = copy(node.local_bounds)
    varbounds_right = copy(node.local_bounds)

    if haskey(varbounds_left.upper_bounds, vidx)
        delete!(varbounds_left.upper_bounds, vidx)
    end
    if haskey(varbounds_right.lower_bounds, vidx)
        delete!(varbounds_right.lower_bounds, vidx)
    end
    push!(varbounds_left.upper_bounds, (vidx => MOI.LessThan(floor(x[vidx]))))
    push!(varbounds_right.lower_bounds, (vidx => MOI.GreaterThan(ceil(x[vidx]))))

    # compute new dual gap
    fw_dual_gap_limit = tree.root.options[:dual_gap_decay_factor] * node.fw_dual_gap_limit
    fw_dual_gap_limit = max(fw_dual_gap_limit, tree.root.options[:min_node_fw_epsilon])

    for v in active_set_left.atoms
        if !(v[vidx] <= floor(x[vidx]) + 1e-9) 
            error( "active_set_left\n$(v)\n$vidx, $(x[vidx]), $(v[vidx])")
        end
    end
    for v in discarded_set_left.storage
        if !(v[vidx] <= floor(x[vidx]) + 1e-9)
            error("storage left\n$(v)\n$vidx, $(x[vidx]), $(v[vidx])")
        end
    end
    for v in active_set_right.atoms
        if !(v[vidx] >= ceil(x[vidx]) - 1e-9)
            error("active_set_right\n$(v)\n$vidx, $(x[vidx]), $(v[vidx])")
        end
    end
    for v in discarded_set_right.storage
        if !(v[vidx] >= ceil(x[vidx]) - 1e-9)
            error("storage right\n$(v)\n$vidx, $(x[vidx]), $(v[vidx])")
        end
    end
    # update the LMO
    node_info_left = (
        active_set=active_set_left,
        discarded_vertices=discarded_set_left,
        local_bounds=varbounds_left,
        level=node.level + 1,
        fw_dual_gap_limit=fw_dual_gap_limit,
        fw_time=Millisecond(0),
    )
    node_info_right = (
        active_set=active_set_right,
        discarded_vertices=discarded_set_right,
        local_bounds=varbounds_right,
        level=node.level + 1,
        fw_dual_gap_limit=fw_dual_gap_limit,
        fw_time=Millisecond(0),
    )
    return [node_info_left, node_info_right]
end

"""
Computes the relaxation at that node
"""
function Bonobo.evaluate_node!(tree::Bonobo.BnBTree, node::FrankWolfeNode)
    # check if conflict between local bounds and global tightening
    for (j, ub) in tree.root.global_tightenings.upper_bounds
        if !haskey(node.local_bounds.lower_bounds, j)
            continue
        end
        lb = node.local_bounds.lower_bounds[j]
        if ub.upper < lb.lower
            @debug "local lb $(lb.lower) conflicting with global tightening $(ub.upper)"
            return NaN, NaN
        end
    end
    for (j, lb) in tree.root.global_tightenings.lower_bounds
        if !haskey(node.local_bounds.upper_bounds, j)
            continue
        end
        ub = node.local_bounds.upper_bounds[j]
        if ub.upper < lb.lower
            @debug "local ub $(ub.upper) conflicting with global tightening $(lb.lower)"
            return NaN, NaN
        end
    end

    # build up node LMO
    build_LMO(
        tree.root.problem.lmo,
        tree.root.problem.integer_variable_bounds,
        node.local_bounds,
        tree.root.problem.integer_variables,
    )

    # check for feasibility and boundedness
    status = check_feasibility(tree.root.problem.lmo)
    if status == MOI.INFEASIBLE
        return NaN, NaN
    end
    if status == MOI.DUAL_INFEASIBLE
        error("Feasible region unbounded! Please check your constraints!")
        return NaN, NaN
    end

    # set relative accurary for the IP solver
    accurary = node.level >= 2 ? 0.1 / (floor(node.level / 2) * (3 / 4)) : 0.10
    if MOI.get(tree.root.problem.lmo.lmo.o, MOI.SolverName()) == "SCIP"
        MOI.set(tree.root.problem.lmo.lmo.o, MOI.RawOptimizerAttribute("limits/gap"), accurary)
    end

    if isempty(node.active_set)
        consI_list = MOI.get(
            tree.root.problem.lmo.lmo.o,
            MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}(),
        ) + MOI.get(
            tree.root.problem.lmo.lmo.o,
            MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.ZeroOne}(),
        )
        if !isempty(consI_list)
            @error "Unreachable node! Active set is empty!"
        end
        restart_active_set(node, tree.root.problem.lmo.lmo, tree.root.problem.nvars)
    end

    # time tracking FW
    active_set = node.active_set
    time_ref = Dates.now()
    FrankWolfe.compute_active_set_iterate!(node.active_set)
    x = node.active_set.x
    for list in (node.local_bounds.lower_bounds, node.local_bounds.upper_bounds)
        for (idx, set) in list
            dist = MOD.distance_to_set(MOD.DefaultDistance(), x[idx], set)
            if dist > 0.01
                @warn "infeas x $dist"
            end
            for v_idx in eachindex(node.active_set)
                dist_v = MOD.distance_to_set(MOD.DefaultDistance(), node.active_set.atoms[v_idx][idx], set)
                if dist_v > 0.01
                    error("vertex beginning")
                end
            end
            if dist > 0.01
                @warn "infeas x $dist"
                @error("infeasible but vertex okay")
                FrankWolfe.compute_active_set_iterate!(active_set)
                dist2 = MOD.distance_to_set(MOD.DefaultDistance(), x[idx], set)
                if dist2 > 0.01
                    error("$dist, $idx, $set")
                else
                    error("recovered")
                end
            end
        end
    end

    # call blended_pairwise_conditional_gradient
    x, _, primal, dual_gap, _, active_set = FrankWolfe.blended_pairwise_conditional_gradient(
        tree.root.problem.f,
        tree.root.problem.g,
        tree.root.problem.lmo,
        node.active_set,
        epsilon=node.fw_dual_gap_limit,
        max_iteration=tree.root.options[:max_fw_iter],
        line_search=FrankWolfe.Adaptive(verbose=false),
        add_dropped_vertices=true,
        use_extra_vertex_storage=true,
        extra_vertex_storage=node.discarded_vertices,
        callback=tree.root.options[:callback],
        lazy=true,
    )

    node.fw_time = Dates.now() - time_ref

    # update active set of the node
    node.active_set = active_set
    lower_bound = primal - dual_gap

    if tree.root.options[:dual_tightening] && isfinite(tree.incumbent)
        grad = similar(x)
        tree.root.problem.g(grad, x)
        num_tightenings = 0
        for j in tree.root.problem.integer_variables
            lb_global = get(tree.root.problem.integer_variable_bounds, (j, :greaterthan), MOI.GreaterThan(-Inf))
            ub_global = get(tree.root.problem.integer_variable_bounds, (j, :lessthan), MOI.LessThan(Inf))
            lb = get(node.local_bounds.lower_bounds, j, lb_global).lower
            ub = get(node.local_bounds.upper_bounds, j, ub_global).upper
            @assert lb >= lb_global.lower
            @assert ub <= ub_global.upper
            if lb ≈ ub
                # variable already fixed
                continue
            end
            gj = grad[j]
            safety_tolerance = 2.0
            rhs = tree.incumbent - tree.root.problem.f(x) + safety_tolerance * dual_gap
            if gj > 0 && ≈(x[j], lb, atol=tree.options.atol, rtol=tree.options.rtol)
                Mlb = 0
                bound_tightened = true
                @debug "starting tightening ub $(rhs)"
                while Mlb * gj <= rhs
                    Mlb += 1
                    if lb + Mlb -1 == ub
                        bound_tightened = false
                        break
                    end
                end
                if bound_tightened
                    new_bound = lb + Mlb - 1
                    @debug "found UB tightening $ub -> $new_bound"
                    node.local_bounds[j, :lessthan] = MOI.LessThan(new_bound)
                    num_tightenings += 1
                    if haskey(tree.root.problem.integer_variable_bounds, (j, :lessthan))
                        @assert node.local_bounds[j, :lessthan].upper <= tree.root.problem.integer_variable_bounds[j, :lessthan].upper
                    end
                end
            elseif gj < 0 && ≈(x[j], ub, atol=tree.options.atol, rtol=tree.options.rtol)
                Mub = 0
                bound_tightened = true
                @debug "starting tightening lb $(rhs)"
                while -Mub * gj <= rhs
                    Mub += 1
                    if ub - Mub + 1 == lb
                        bound_tightened = false
                        break
                    end
                end
                if bound_tightened
                    new_bound = ub - Mub + 1
                    @debug "found LB tightening $lb -> $new_bound"
                    node.local_bounds[j, :greaterthan] = MOI.GreaterThan(new_bound)
                    num_tightenings += 1
                    if haskey(tree.root.problem.integer_variable_bounds, (j, :greaterthan))
                        @assert node.local_bounds[j, :greaterthan].lower >= tree.root.problem.integer_variable_bounds[j, :greaterthan].lower
                    end
                end
            end
        end
        @debug "# tightenings $num_tightenings"
    end

    # store gradient, dual gap and relaxation
    if tree.root.options[:global_dual_tightening] && node.std.id == 1
        @debug "storing root node info for tightening"
        grad = similar(x)
        tree.root.problem.g(grad, x)
        safety_tolerance = 2.0
        tree.root.global_tightening_rhs[] = -tree.root.problem.f(x) + safety_tolerance * dual_gap
        for j in tree.root.problem.integer_variables
            if haskey(tree.root.problem.integer_variable_bounds.upper_bounds, j)
                ub = tree.root.problem.integer_variable_bounds[j, :lessthan]
                if ≈(x[j], ub.upper, atol=tree.options.atol, rtol=tree.options.rtol) && grad[j] < 0
                    tree.root.global_tightening_root_info.upper_bounds[j] = (grad[j], ub)
                end
            end
            if haskey(tree.root.problem.integer_variable_bounds.lower_bounds, j)
                lb = tree.root.problem.integer_variable_bounds[j, :greaterthan]
                if ≈(x[j], lb.lower, atol=tree.options.atol, rtol=tree.options.rtol) && grad[j] > 0
                    tree.root.global_tightening_root_info.lower_bounds[j] = (grad[j], lb)
                end
            end
        end
    end

    # new incumbent: check global fixings
    if tree.root.options[:global_dual_tightening] && tree.root.updated_incumbent[]
        num_tightenings = 0
        rhs = tree.incumbent + tree.root.global_tightening_rhs[]
        @assert isfinite(rhs)
        for (j, (gj, lb)) in tree.root.global_tightening_root_info.lower_bounds
            ub = get(tree.root.problem.integer_variable_bounds.upper_bounds, j, MOI.LessThan(Inf)).upper
            ub_new = get(tree.root.global_tightening_root_info.upper_bounds, j, MOI.LessThan(Inf)).upper
            ub = min(ub, ub_new)
            Mlb = 0
            bound_tightened = true
            lb = lb.lower
            while Mlb * gj <= rhs
                Mlb += 1
                if lb + Mlb -1 == ub
                    bound_tightened = false
                    break
                end
            end
            if bound_tightened
                new_bound = lb + Mlb - 1
                @debug "found global UB tightening $ub -> $new_bound"
                tree.root.global_tightenings.upper_bounds[j] = MOI.LessThan(new_bound)
                num_tightenings += 1
            end
        end
        for (j, (gj, ub)) in tree.root.global_tightening_root_info.upper_bounds
            lb = get(tree.root.problem.integer_variable_bounds.lower_bounds, j, MOI.GreaterThan(-Inf)).lower
            lb_new = get(tree.root.global_tightening_root_info.lower_bounds, j, MOI.GreaterThan(-Inf)).lower
            lb = max(lb, lb_new)
            Mub = 0
            bound_tightened = true
            while -Mub * gj <= rhs
                Mub += 1
                if ub - Mub + 1 == lb
                    bound_tightened = false
                    break
                end
            end
            if bound_tightened
                new_bound = ub - Mub + 1
                @debug "found global LB tightening $lb -> $new_bound"
                tree.root.global_tightenings.lower_bounds[j] = MOI.GreaterThan(new_bound)
                num_tightenings += 1
            end
        end
    end

    # Found an upper bound?
    if is_integer_feasible(tree, x)
        node.ub = primal
        return lower_bound, primal
    end

    return lower_bound, NaN
end


"""
Returns the solution vector of the relaxed problem at the node
"""
function Bonobo.get_relaxed_values(tree::Bonobo.BnBTree, node::FrankWolfeNode)
    return copy(FrankWolfe.get_active_set_iterate(node.active_set))
end

tree_lb(tree::Bonobo.BnBTree) = min(tree.lb, tree.incumbent)
