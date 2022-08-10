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
    active_set_left, active_set_right = split_vertices_set!(node.active_set, tree, vidx)
    discarded_set_left, discarded_set_right = split_vertices_set!(node.discarded_vertices, tree, vidx, x)
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

   # update the LMO
   node_info_left = (active_set = active_set_left, discarded_vertices = discarded_set_left, local_bounds = varbounds_left, level = node.level+1, fw_dual_gap_limit = fw_dual_gap_limit, fw_time = Millisecond(0))
   node_info_right = (active_set = active_set_right, discarded_vertices = discarded_set_right, local_bounds = varbounds_right, level = node.level+1, fw_dual_gap_limit = fw_dual_gap_limit, fw_time = Millisecond(0))
   return [node_info_left, node_info_right]
end

"""
Computes the relaxation at that node
"""
function Bonobo.evaluate_node!(tree::Bonobo.BnBTree, node::FrankWolfeNode)
    # build up node LMO
    build_LMO(tree.root.problem.lmo, tree.root.problem.integer_variable_bounds, node.local_bounds, tree.root.problem.integer_variables)

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
    accurary = node.level >= 2 ? 0.1/(floor(node.level/2)*(3/4)) : 0.10
    if MOI.get(tree.root.problem.lmo.lmo.o, MOI.SolverName()) == "SCIP"
        MOI.set(tree.root.problem.lmo.lmo.o, MOI.RawOptimizerAttribute("limits/gap"), accurary)
    end 

    if isempty(node.active_set)
        consI_list = MOI.get(tree.root.problem.lmo.lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.Integer}())
        if MOI.get(tree.root.problem.lmo.lmo.o, MOI.SolverName()) == "SCIP" && !isempty(consI_list)
            @error "Unreachable node! Active set is empty!"
        end
        restart_active_set(node, tree.root.problem.lmo.lmo, tree.root.problem.nvars)
    end

    # time tracking FW 
    time_ref = Dates.now()

    if node.id === 168
        nothing
        println("Here!")
    end

    # DEBUG 
    # Commented out old debug code
    # x = FrankWolfe.get_active_set_iterate(node.active_set)
    # gradient = randn(Float64, length(x))
    # tree.root.problem.g(gradient, x)
    #@show gradient

    # call blended_pairwise_conditional_gradient
    x,_,primal,dual_gap,_ , active_set = FrankWolfe.blended_pairwise_conditional_gradient(
        tree.root.problem.f,
        tree.root.problem.g,
        tree.root.problem.lmo,
        node.active_set,
        epsilon = node.fw_dual_gap_limit,
        max_iteration = tree.root.options[:max_fw_iter],
        add_dropped_vertices=true,
        use_extra_vertex_storage=true,
        extra_vertex_storage=node.discarded_vertices,
        callback=tree.root.options[:callback],
        lazy=true,
        verbose=false,
    ) 

    node.fw_time = Dates.now() - time_ref

   #= if tree.incumbent != Inf
        println("\n")
        @show node.id
        @show tree.incumbent
        #@show tree.solutions[1].solution
        @show tree.root.problem.f(tree.incumbent_solution.solution)
        @show tree.incumbent === tree.root.problem.f(tree.incumbent_solution.solution)
    end =#

    # update active set of the node
    node.active_set = active_set
    lower_bound = primal - dual_gap

    # check check_feasibility
    if !is_linear_feasible(tree.root.problem.lmo, x)
        @error "Solution not linear feasible!"
    end

    # Found an upper bound?
    if is_integer_feasible(tree,x)
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

#=function Bonobo.terminated(tree::Bonobo.BnBTree{<:FrankWolfeNode})
    dual_gap = relative_gap(tree.incumbent,tree_lb(tree))
    return isempty(tree.node_queue) || dual_gap ≤ tree.root.options[:dual_gap]
end=#

tree_lb(tree::Bonobo.BnBTree) = min(tree.lb, tree.incumbent)
