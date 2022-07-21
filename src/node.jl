"""
    AbtractFrankWolfeNode <: Bonobo.AbstractNode 
"""
abstract type AbstractFrankWolfeNode  <: Bonobo.AbstractNode end 

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
mutable struct FrankWolfeNode{AT<:FrankWolfe.ActiveSet, DVS<:FrankWolfe.DeletedVertexStorage, IB<:IntegerBounds} <: AbstractFrankWolfeNode
    std::Bonobo.BnBNodeInfo
    active_set::AT
    discarded_vertices::DVS
    local_bounds::IB
    level::Int
    fw_dual_gap_limit::Float64
end

"""
Create the information of the new branching nodes 
based on their parent and the index of the branching variable
"""
function Bonobo.get_branching_nodes_info(tree::Bonobo.BnBTree, node::FrankWolfeNode, vidx::Int; percentage_dual_gap)
    if !is_valid_split(tree, vidx)
        error("Splitting on the same index as parent! Abort!")
    end
    # update splitting index
    x = Bonobo.get_relaxed_values(tree, node)
    # split active set
    active_set_left, active_set_right = split_vertices_set!(node.active_set, tree, vidx)
    discarded_set_left, discarded_set_right = split_vertices_set!(node.discarded_vertices, tree, vidx, x)
    # add new bounds to the feasible region left and right
    # copy bounds from parent
    varBoundsLeft = copy(node.local_bounds)
    varBoundsRight = copy(node.local_bounds)

    if haskey(varBoundsLeft.upper_bounds, vidx)
        delete!(varBoundsLeft.upper_bounds, vidx)
    end
    if haskey(varBoundsRight.lower_bounds, vidx)
        delete!(varBoundsRight.lower_bounds, vidx)
    end
    push!(varBoundsLeft.upper_bounds, (vidx => MOI.LessThan(floor(x[vidx]))))
    push!(varBoundsRight.lower_bounds, (vidx => MOI.GreaterThan(ceil(x[vidx]))))

    # add dual gap
    fw_dual_gap_limit = percentage_dual_gap * node.fw_dual_gap_limit
    #update the LMO's
    node_info_left = (active_set = active_set_left, discarded_vertices = discarded_set_left, local_bounds = varBoundsLeft, level = node.level+1, fw_dual_gap_limit = fw_dual_gap_limit)
    node_info_right = (active_set = active_set_right, discarded_vertices = discarded_set_right, local_bounds = varBoundsRight, level = node.level+1, fw_dual_gap_limit = fw_dual_gap_limit)
    return [node_info_left, node_info_right]
end

"""
Computes the relaxation at that node
"""
function Bonobo.evaluate_node!(tree::Bonobo.BnBTree, node::FrankWolfeNode, fw_callback)
    # build up node LMO
    build_LMO(tree.root.problem.lmo, tree.root.problem.integer_variable_bounds, node.local_bounds, tree.root.problem.integer_variables)

    # check for feasibility and boundedness
    status = check_feasibility(tree.root.problem.lmo)
    if status == MOI.INFEASIBLE
        return NaN, NaN, NaN, NaN
    end
    if status == MOI.DUAL_INFEASIBLE
        error("Feasible region unbounded! Please check your constraints!")
        return NaN, NaN, NaN, NaN
    end

    # set relative accurary for the IP solver
   #= accurary = node.level >= 2 ? 0.1/(floor(node.level/2)*(3/4)) : 0.10
    if MOI.get(tree.root.problem.lmo.lmo.o, MOI.SolverName()) == "SCIP"
        MOI.set(tree.root.problem.lmo.lmo.o, MOI.RawOptimizerAttribute("limits/gap"), accurary)
    end =#

    if isempty(node.active_set)
        consI_list = MOI.get(tree.root.problem.lmo.lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.Integer}())
        if MOI.get(tree.root.problem.lmo.lmo.o, MOI.SolverName()) == "SCIP" && !isempty(consI_list)
            @error "Unreachable node! Active set is empty!"
        end
        restart_active_set(node, tree.root.problem.lmo.lmo, tree.root.problem.nvars)
    end

    # time tracking FW 
    time_ref = Dates.now()
    # time tracking LMO
    len = length(tree.root.problem.lmo.optimizing_times)

    # DEBUG 
    x = FrankWolfe.get_active_set_iterate(node.active_set)
    gradient = randn(Float64, length(x))
    tree.root.problem.g(gradient, x)
    #@show gradient

    # call away_frank_wolfe
    x,_,primal,dual_gap,_ , active_set = FrankWolfe.blended_pairwise_conditional_gradient(
        tree.root.problem.f,
        tree.root.problem.g,
        tree.root.problem.lmo,
        node.active_set,
        epsilon = get(tree.root.options, :FW_tol, -1),
        add_dropped_vertices=true,
        use_extra_vertex_storage=true,
        extra_vertex_storage=node.discarded_vertices,
        callback=fw_callback,
        lazy=true,
        verbose=false,
    ) 

    time_FW = Dates.now() - time_ref
    time_LMO = sum(1000*tree.root.problem.lmo.optimizing_times[len+1:end]) # TODO: no hardcoding of numbers. make parameter or constant

    # update active set of the node
    node.active_set = active_set
    lower_bound = primal - dual_gap

    # Found an upper bound?
    if is_integer_feasible(tree,x)
        #@show lower_bound, primal
        node.ub = primal
        return lower_bound, primal, time_FW, time_LMO
    end

    return lower_bound, NaN, time_FW, time_LMO
end 


"""
Returns the solution vector of the relaxed problem at the node
"""
function Bonobo.get_relaxed_values(tree::Bonobo.BnBTree, node::FrankWolfeNode)
    return copy(FrankWolfe.get_active_set_iterate(node.active_set))
end


