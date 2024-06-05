struct PSEUDO_COST{BLMO<:BoundedLinearMinimizationOracle} <: Bonobo.AbstractBranchStrategy
    iterations_until_stable::Int
    stable::Bool
    bounded_lmo::BLMO
end

# """ Function that keeps track of which branching candidates are stable """
# function is_stable(idx::Int, branching::PSEUDO_COST{BLMO})
#     local branch_tracker = Dict{Int, Float64}(idx=> 0 for idx in Boscia.get_integer_variables(branching.bounded_lmo))
#     if branch_tracker[idx] >= branching.iterations_until_stable
#         return true
#     else 
#         return false
#     end
# end


function pseudo_weight_update!(
    tree::Bonobo.BnBTree, 
    node::Bonobo.AbstractNode, 
    idx::Int,
    values,
    pseudos::Dict{Int,Array{Float64}},
    branch_tracker::Dict{Int, Int},
    branching::PSEUDO_COST{BLMO}
) where BLMO <: BoundedLinearMinimizationOracle
    @assert !isempty(node.active_set)
    active_set = copy(node.active_set)
    empty!(active_set)
    current_dual_gap = node.dual_gap# if this is a fw_node this should be the fw_dual_gap

    fx = floor(values[idx])
    # create LMO
    boundsLeft = copy(node.local_bounds)
    if haskey(boundsLeft.upper_bounds, idx)
        delete!(boundsLeft.upper_bounds, idx)
    end
    push!(boundsLeft.upper_bounds, (idx => fx))
    build_LMO(
                branching.bounded_lmo,
                tree.root.problem.integer_variable_bounds,
                boundsLeft,
                Bonobo.get_branching_indices(tree.root),
            )
    status = check_feasibility(branching.bounded_lmo)
    if status == OPTIMAL
        empty!(active_set)
        for (λ, v) in node.active_set
            if v[idx] <= values[idx]
                push!(active_set, ((λ, v)))
            end
        end
        @assert !isempty(active_set)
        FrankWolfe.active_set_renormalize!(active_set)
        x, primal, dual_gap, active_set
        _, _, primal_relaxed, dual_gap_relaxed, _ =
            FrankWolfe.blended_pairwise_conditional_gradient(
                tree.root.problem.f,
                tree.root.problem.g,
                branching.bounded_lmo,
                active_set,
                verbose=false,
                epsilon=branching.solving_epsilon,
                max_iteration=5,
            )
        left_update = current_dual_gap - dual_gap_relaxed
    else
        @debug "Left non-optimal status $(status)"
        left_update = Inf
    end

    #right node: x_i >=  floor(̂x_i)
    cx = ceil(values[idx])
    boundsRight = copy(node.local_bounds)
    if haskey(boundsRight.lower_bounds, idx)
        delete!(boundsRight.lower_bounds, idx)
    end
    push!(boundsRight.lower_bounds, (idx => cx))
    build_LMO(
        branching.bounded_lmo,
        tree.root.problem.integer_variable_bounds,
        boundsRight,
        Bonobo.get_branching_indices(tree.root),
    )
    status = check_feasibility(branching.bounded_lmo)
    if status == OPTIMAL
        empty!(active_set)
        for (λ, v) in node.active_set
            if v[idx] >= values[idx]
                push!(active_set, (λ, v))
            end
        end
        if isempty(active_set)
            @show values[idx]
            @show length(active_set)
            @info [active_set.atoms[idx] for idx in eachindex(active_set)]
            error("Empty active set, unreachable")
        end
        FrankWolfe.active_set_renormalize!(active_set)
        _, _, primal_relaxed, dual_gap_relaxed, _ =
            FrankWolfe.blended_pairwise_conditional_gradient(
                tree.root.problem.f,
                tree.root.problem.g,
                branching.bounded_lmo,
                active_set,
                verbose=false,
                epsilon=branching.solving_epsilon,
                max_iteration=5,
            )
        right_update = current_dual_gap	 - dual_gap_relaxed
    else
        @debug "Right non-optimal status $(status)"
        right_update = Inf
    end
    # reset LMO
    build_LMO(
        branching.bounded_lmo,
        tree.root.problem.integer_variable_bounds,
        node.local_bounds,
        Bonobo.get_branching_indices(tree.root),
    )
    pseudos[idx][1] = update_avg(left_update, pseudos[idx][1], branch_tracker[idx])
    pseudos[idx][2] = update_avg(right_update, pseudos[idx][2], branch_tracker[idx])
    branch_tracker[idx] += 1
end

function best_pseudo_choice(# currently not in use. To be implemented later in order to improve readability
    tree::Bonobo.BnBTree,
)
    branching_candidates = Bonobo.get_branching_indices(tree.root)
    return argmax(map(idx-> maximum(pseudos[idx]), branching_candidates))    
end


"""
    get_branching_variable(
    tree::Bonobo.BnBTree, 
    branching::PSEUDO_COST,
    node::Bonobo.AbstractNode
)

Get branching variable using Pseudocost branching after costs have stabilized. 
Prior to stabilization an adaptation of the Bonobo MOST_INFEASIBLE is used.

"""
function Bonobo.get_branching_variable(
    tree::Bonobo.BnBTree, 
    branching::PSEUDO_COST{BLMO},
    node::Bonobo.AbstractNode,
) where BLMO <: BoundedLinearMinimizationOracle
    #the following should create the dictionaries only on first function call and then use existing ones
    local pseudos = Dict{Int,Array{Float64}}(idx=>zeros(2) for idx in Boscia.get_integer_variables(branching.bounded_lmo))
    local branch_tracker = Dict{Int, Int}(idx=> 0 for idx in Boscia.get_integer_variables(branching.bounded_lmo))
    local call_tracker = 0
    
    best_idx = -1
    all_stable = true
    for idx in Bonobo.get_branching_indices(tree.root)
        if branch_tracker[idx] >= branching.iterations_until_stable
            all_stable = true
        else 
            all_stable = false
            break
        end
    end
    if !all_stable# THEN Use Most Infeasible
        values = Bonobo.get_relaxed_values(tree, node)
        max_distance_to_feasible = 0.0
        for i in  Bonobo.get_branching_indices(tree.root)
            value = values[i]
            if !Bonobo.is_approx_feasible(tree, value)
                distance_to_feasible = Bonobo.get_distance_to_feasible(tree, value)
                if distance_to_feasible > max_distance_to_feasible
                    best_idx = i
                    max_distance_to_feasible = distance_to_feasible
                end
            end
        end
        if best_idx != -1
            pseudo_weight_update!(tree, node, best_idx, values, pseudos, branch_tracker, branching)
        end
        return best_idx
    else
        println("Pseudos are stable")
        # Pseudocosts have stabilized
        call_tracker +=1
        println("pseudocosts have stabilized ", call_tracker)

        branching_candidates = Bonobo.get_branching_indices(tree.root)
        best_idx = argmax(map(idx-> maximum(pseudos[idx]), branching_candidates))# argmax randomly chosen and to be replaced later
        return best_idx
    end
end

function update_avg(new_val::Float64, avg::Float64, N::Int)
    # N is the number of values used to compute the current avg
    # avg is the current average
    # new_val is the value that the current average has to be updated with
    if N > 1
        return 1/(N+1) * (N * avg + new_val)        
    else
        return new_val
    end
end
#pseudos = Dict{Int,Array{Float64}}(i=>zeros(2) for idx in get_integer_variables(blmo))

#branch_tracker = Dict{Int, Int}(idx-> 0 for idx in get_integer_variables(blmo))