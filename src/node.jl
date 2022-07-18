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
 'sidx' stores splitting index of the parent before splitting and updates it to the splitting index of this node in branch!
"""
mutable struct FrankWolfeNode{AT<:FrankWolfe.ActiveSet, DVS<:FrankWolfe.DeletedVertexStorage, IB<:IntegerBounds} <: AbstractFrankWolfeNode
    std::Bonobo.BnBNodeInfo
    active_set::AT
    discarded_vertices::DVS
    local_bounds::IB
    level::Int
    sidx::Int
    fw_dual_gap_limit::Float64
    FW_time::Millisecond
end

"""
    InfeasibleFrankWolfeNode <: AbstractFrankWolfeNode

A node in the branch-and-bound tree storing information for a Frank-Wolfe subproblem.

`std` stores the id, lower and upper bound of the node.
`valid_active` vector of booleans indicating which vertices in the global active set are valid for the node.
`lmo` is the minimization oracle capturing the feasible region.   
"""
mutable struct InfeasibleFrankWolfeNode{IB<:IntegerBounds} <: AbstractFrankWolfeNode
    std::Bonobo.BnBNodeInfo
    valid_active::Vector{Bool}
    local_bounds::IB
end

"""
Create the information of the new branching nodes 
based on their parent and the index of the branching variable
"""
function Bonobo.get_branching_nodes_info(tree::Bonobo.BnBTree, node::FrankWolfeNode, vidx::Int)
    if vidx == node.sidx && is_binary_constraint(tree, vidx)
        error("Splitting on the same index as parent! Abort!")
    end
    # update splitting index
    node.sidx = vidx
    x = Bonobo.get_relaxed_values(tree, node)
    # split active set
   #temp_active_set = copy(node.active_set)
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
   fw_dual_gap_limit = tree.root.options[:percentage_dual_gap] * node.fw_dual_gap_limit
   #update the LMO's
   node_info_left = (active_set = active_set_left, discarded_vertices = discarded_set_left, local_bounds = varBoundsLeft, level = node.level+1, sidx = vidx, fw_dual_gap_limit = fw_dual_gap_limit, FW_time = Millisecond(0))
   node_info_right = (active_set = active_set_right, discarded_vertices = discarded_set_right, local_bounds = varBoundsRight, level = node.level+1, sidx = vidx, fw_dual_gap_limit = fw_dual_gap_limit, FW_time = Millisecond(0))
   return [node_info_left, node_info_right]
end

"""
InfeasibleFrankWolfeNode: Create the information of the new branching nodes 
based on their parent and the index of the branching variable
"""
function Bonobo.get_branching_nodes_info(tree::Bonobo.BnBTree, node::InfeasibleFrankWolfeNode, vidx::Int)
   # get solution
   x = Bonobo.get_relaxed_values(tree, node)
  
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

   
   #valid_active is set at evaluation time
   node_info_left = (valid_active = Bool[], local_bounds = varBoundsLeft) 
   node_info_right = (valid_active = Bool[],local_bounds = varBoundsRight)
   
   return [node_info_left, node_info_right]

end


"""
Split an active set between left and right children.
"""
function split_vertices_set!(active_set::FrankWolfe.ActiveSet{T, R}, tree, var::Int; atol = 1e-5, rtol = 1e-5) where {T,R}
    #right_as = FrankWolfe.ActiveSet{Vector{T}, R, Vector{T}}([], [], active_set.x)
    x = FrankWolfe.get_active_set_iterate(active_set)
    right_as = FrankWolfe.ActiveSet{Vector{Float64}, Float64, Vector{Float64}}([], [], similar(active_set.x))  # works..
    # indices to remove later from the left active set
    left_del_indices = BitSet()
    for (idx, tup) in enumerate(active_set)
        (λ, a) = tup
        # if variable set to 1 in the atom,
        # place in right branch, delete from left
        
        if a[var] >= ceil(x[var]) || isapprox(a[var], ceil(x[var]),atol=atol, rtol=rtol)
            push!(right_as, tup)
            push!(left_del_indices, idx)
        elseif a[var] <= floor(x[var]) || isapprox(a[var], floor(x[var]),atol=atol, rtol=rtol)
            # keep in left, don't add to right
        else #floor(x[var]) < a[var] < ceil(x[var])
            # if you are in middle, delete from the left and do not add to the right!
            consI_list = MOI.get(tree.root.problem.lmo.lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.Integer}())
            if MOI.get(tree.root.problem.lmo.lmo.o, MOI.SolverName()) == "SCIP" && !isempty(consI_list)
                @warn "Attention! Vertex in the middle."
            end
            push!(left_del_indices, idx)

        end
    end
    deleteat!(active_set, left_del_indices)
    # renormalize active set and recompute new iterates 
    if !isempty(active_set)
        FrankWolfe.active_set_renormalize!(active_set)
        FrankWolfe.compute_active_set_iterate!(active_set)
    end
    if !isempty(right_as)
        FrankWolfe.active_set_renormalize!(right_as)
        FrankWolfe.compute_active_set_iterate!(right_as)
    end
    return (active_set, right_as)
end

"""
Split a discarded vertices set between left and right children.
"""
function split_vertices_set!(discarded_set::FrankWolfe.DeletedVertexStorage{T}, tree, var::Int, x; atol = 1e-5, rtol = 1e-5) where {T}
    right_as = FrankWolfe.DeletedVertexStorage{}(Vector{Float64}[], discarded_set.return_kth)
    # indices to remove later from the left active set
    left_del_indices = BitSet()
    for (idx, vertex) in enumerate(discarded_set.storage)
        if vertex[var] >= ceil(x[var]) || isapprox(vertex[var], ceil(x[var]),atol=atol, rtol=rtol)
            push!(right_as.storage, vertex)
            push!(left_del_indices, idx)
        elseif vertex[var] <= floor(x[var]) || isapprox(vertex[var], floor(x[var]),atol=atol, rtol=rtol)
            # keep in left, don't add to right
        else #floor(x[var]) < vertex[var] < ceil(x[var])
            # if you are in middle, delete from the left and do not add to the right!
            push!(left_del_indices, idx)
        end
    end
    deleteat!(discarded_set.storage, left_del_indices)
    return (discarded_set, right_as)
end

"""
Computes the relaxation at that node
"""
function Bonobo.evaluate_node!(tree::Bonobo.BnBTree, node::FrankWolfeNode)
   # @show node.id
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
        callback=tree.root.options[:callback],
        lazy=true,
        verbose=false,
    ) 
    # @show x,primal,dual_gap, active_set

    # println("after fw: ", node.discarded_vertices)
    node.FW_time = Dates.now() - time_ref

    # update active set of the node
    node.active_set = active_set
    #@show primal, dual_gap
    lower_bound = primal - dual_gap
    
    # Found an upper bound?
    if is_integer_feasible(tree,x)
        #@show lower_bound, primal
        node.ub = primal
        return lower_bound, primal
    end

    return lower_bound, NaN
end 

"""
Check feasibility and boundedness
"""
function check_feasibility(lmo::TimeTrackingLMO)
    MOI.optimize!(lmo)
    status = MOI.get(lmo.lmo.o, MOI.TerminationStatus())
    return status
end 


"""
Returns the solution vector of the relaxed problem at the node
"""
function Bonobo.get_relaxed_values(tree::Bonobo.BnBTree, node::FrankWolfeNode)
    return copy(FrankWolfe.get_active_set_iterate(node.active_set))
end

function Bonobo.get_relaxed_values(tree::Bonobo.BnBTree, node::InfeasibleFrankWolfeNode)
    return copy(FrankWolfe.get_active_set_iterate(tree.root.problem.active_set))
end

"""
Check if at a given index we have a binary and integer constraint respectivily.
"""
function is_binary_constraint(tree::Bonobo.BnBTree, idx::Int)
    consB_list = MOI.get(tree.root.problem.lmo.lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.ZeroOne}())
    for c_idx in consB_list
        if c_idx.value == idx
            return true
        end
    end
    return false
end

function is_integer_constraint(tree::Bonobo.BnBTree, idx::Int)
    consB_list = MOI.get(tree.root.problem.lmo.lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.Integer}())
    for c_idx in consB_list
        if c_idx.value == idx
            return true
        end
    end
    return false
end

"""
Build up valid_active; is called whenever the global active_set changes
"""
function  populate_valid_active!(active_set::FrankWolfe.ActiveSet, node::InfeasibleFrankWolfeNode, lmo::FrankWolfe.LinearMinimizationOracle)
    empty!(node.valid_active)
    for i in eachindex(active_set)
        push!(node.valid_active, is_linear_feasible(lmo, active_set.atoms[i]))
    end
end

"""
Call this if the active set is empty after splitting.
Remark: This should not happen when using SCIP as IP solver for the nodes!
"""
function restart_active_set(node::FrankWolfeNode, lmo::FrankWolfe.LinearMinimizationOracle, nvars::Int)
    direction = Vector{Float64}(undef,nvars)
    Random.rand!(direction)
    v = compute_extreme_point(lmo, direction)
    push!(node.active_set, (1.0, v))
    return node.active_set
end

"""
Sort active set by weight and feasibility.

DOES NOT WORK; matrix not a good idea, try vector and append?
"""
function sort_active_set!(as::FrankWolfe.ActiveSet, node::InfeasibleFrankWolfeNode)
    if isempty(as)
        return nothing
    end
    mergedVec = rand(2, size(node.bool_active,1))
    mergedVec[1,:]=node.bool_active
    mergedVec[2,:]=active_set
    #sort by weight and feasibility
    sort(mergedVec, by = x->x[2], rev=true)
    sort(mergedVec, by = x->x[1], rev=true)

    node.bool_active .= mergedVec[1]
    as .= mergedVec[2:size(mergedVec,1)]

    return (node.bool_active, as)
end

# function Bonobo.optimize!(tree::Bonobo.BnBTree; min_number_lower=20, percentage_dual_gap=0.7, callback=(args...; kwargs...)->(),)
#     println("OWN OPTIMIZE FUNCTION USED")
#     time_ref = Dates.now()
#     list_lb = Float64[] 
#     list_ub = Float64[]
#     FW_iterations = []
#     iteration = 0
#     time = 0.0
#     list_time = Float64[]
#     list_num_nodes = Int64[]
#     list_lmo_calls = Int64[]
    
#     fw_callback = build_FW_callback(tree, min_number_lower, true, FW_iterations)
#     callback = build_bnb_callback(tree)
#     while !Bonobo.terminated(tree)
#         node = Bonobo.get_next_node(tree, tree.options.traverse_strategy)
#         # println("current node: ", node.id)
#         # println("nodes : ", tree.num_nodes)
#         # if node.id == 25
#         #     return false
#         # end
#         # @show(tree.lb, tree.incumbent)
#         tree.root.current_node_id[] = node.id
#         lb, ub, FW_time, LMO_time = Bonobo.evaluate_node!(tree, node, fw_callback) 
        
#         # if the problem was infeasible we simply close the node and continue
#         # println(FW_iterations)
#         if isnan(lb) && isnan(ub)
#             Bonobo.close_node!(tree, node)
#             list_lb, list_ub, iteration, time, list_time, list_num_nodes, list_lmo_calls = callback(tree, node; FW_time=FW_time, LMO_time=LMO_time, FW_iterations=FW_iterations, node_infeasible=true)
#             continue
#         end

#         Bonobo.set_node_bound!(tree.sense, node, lb, ub)
#         # if the evaluated lower bound is worse than the best incumbent -> close and continue
#         if node.lb >= tree.incumbent
#             Bonobo.close_node!(tree, node)
#             list_lb, list_ub, iteration, time, list_time, list_num_nodes, list_lmo_calls = callback(tree, node; FW_time=FW_time, LMO_time=LMO_time, FW_iterations=FW_iterations, worse_than_incumbent=true)
#             continue
#         end
#         updated = Bonobo.update_best_solution!(tree, node)
#         updated && Bonobo.bound!(tree, node.id)

#         Bonobo.close_node!(tree, node)
#         #println("branch node")
#         Bonobo.branch!(tree, node; percentage_dual_gap=percentage_dual_gap)
#         list_lb, list_ub, iteration, time, list_time, list_num_nodes, list_lmo_calls = callback(tree, node; FW_time=FW_time, LMO_time=LMO_time, FW_iterations=FW_iterations,)
#     end
#      if get(tree.root.options, :verbose, -1)
#         println("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
#         x = Bonobo.get_solution(tree)
#         println("objective: ", tree.root.problem.f(x))
#         println("number of nodes: $(tree.num_nodes)")
#         println("number of lmo calls: ", tree.root.problem.lmo.ncalls)
#         println("time in seconds: ", (Dates.value(Dates.now()-time_ref))/1000)
#         append!(list_ub, copy(tree.incumbent))
#         append!(list_lb, copy(tree.lb))
#     end
#     return list_lb::Vector{Float64}, list_ub::Vector{Float64}, iteration::Int, time, list_time::Vector{Float64}, list_num_nodes::Vector{Int64}, list_lmo_calls::Vector{Int64}
# end
