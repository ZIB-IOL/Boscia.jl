# Ultilities function 

"""
Compute relative gap consistently everywhere
"""
function relative_gap(primal,dual)
    gap = if signbit(primal) != signbit(dual)
        Inf
    elseif primal == dual
        0.0
    else
        (primal - dual) / min(abs(primal), abs(dual))
    end
    return gap
end

"""
Check feasibility and boundedness
"""
function check_feasibility(lmo::TimeTrackingLMO)
    MOI.set(lmo.lmo.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction{Float64}([], 0.0))
    MOI.optimize!(lmo)
    status = MOI.get(lmo.lmo.o, MOI.TerminationStatus())
    return status
end 


"""
Check if at a given index we have a binary and integer constraint respectivily.
"""
function is_binary_constraint(tree::Bonobo.BnBTree, idx::Int)
    consB_list = MOI.get(tree.root.problem.lmo.lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.ZeroOne}())
    for c_idx in consB_list
        if c_idx.value == idx
            return true, c_idx
        end
    end
    return false, -1
end

function is_integer_constraint(tree::Bonobo.BnBTree, idx::Int)
    consB_list = MOI.get(tree.root.problem.lmo.lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.Integer}())
    for c_idx in consB_list
        if c_idx.value == idx
            return true, c_idx
        end
    end
    return false, -1
end


"""
Check wether a split is valid. It is 
"""
function is_valid_split(tree::Bonobo.BnBTree, vidx::Int)
    bin_var, _ = is_binary_constraint(tree, vidx)
    int_var, _ = is_integer_constraint(tree, vidx)
    if int_var || bin_var
        l_idx = MOI.ConstraintIndex{MOI.VariableIndex, MOI.GreaterThan{Float64}}(vidx)
        u_idx = MOI.ConstraintIndex{MOI.VariableIndex, MOI.LessThan{Float64}}(vidx)
        l_bound = MOI.is_valid(get_optimizer(tree), l_idx) ? MOI.get(get_optimizer(tree), MOI.ConstraintSet(), l_idx) : nothing
        u_bound = MOI.is_valid(get_optimizer(tree), u_idx) ? MOI.get(get_optimizer(tree), MOI.ConstraintSet(), u_idx) : nothing
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
Split an active set between left and right children.
"""
function split_vertices_set!(active_set::FrankWolfe.ActiveSet{T, R}, tree, var::Int; atol = 1e-5, rtol = 1e-5) where {T,R}
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


"""
Checks if the branch and bound can be stopped.
By default (in Bonobo) stops then the priority queue is empty. 
"""
function Bonobo.terminated(tree::Bonobo.BnBTree{FrankWolfeNode})
    dual_gap = get(tree.root.options, :dual_gap, -1)
    if tree.incumbent - tree_lb(tree) < dual_gap || isempty(tree.nodes)
        return true
    end
    return false
end


"""
Naive optimization by enumeration.
Default uses binary values.
Otherwise, third argument should be a vector of n sets of possible values for the variables.
"""
function min_via_enum(f, n, values = fill(0:1,n))
    solutions = Iterators.product(values...)
    best_val = Inf
    best_sol = nothing
    for sol in solutions
        sol_vec = collect(sol)
        val = f(sol_vec)
        if best_val > val
            best_val = val
            best_sol = sol_vec
        end
    end
    return best_val, best_sol
end
