using SparseArrays

struct PSEUDO_COST{BLMO<:BoundedLinearMinimizationOracle} <: Bonobo.AbstractBranchStrategy
    iterations_until_stable::Int
    stable::Bool
    bounded_lmo::BLMO
    μ::Float64
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
    pseudos::SparseMatrixCSC{Float64, Int64},
    branch_tracker::SparseMatrixCSC{Int64, Int64},
) where BLMO <: BoundedLinearMinimizationOracle
    #the following should create the arrays and vectors only on first function call and then use existing one
    # local pseudos = sparse(
    #     repeat(Boscia.get_integer_variables(branching.bounded_lmo), 2),
    #     vcat(ones(length(Boscia.get_integer_variables(branching.bounded_lmo))), 2*ones(length(Boscia.get_integer_variables(branching.bounded_lmo)))), 
    #     ones(2 * length(Boscia.get_integer_variables(branching.bounded_lmo)))
    #     )
    # local branch_tracker = sparse(
    #     repeat(Boscia.get_integer_variables(branching.bounded_lmo), 2),
    #     vcat(ones(length(Boscia.get_integer_variables(branching.bounded_lmo))), 2*ones(length(Boscia.get_integer_variables(branching.bounded_lmo)))), 
    #     ones(Int64, 2 * length(Boscia.get_integer_variables(branching.bounded_lmo)))
    #     )

    #local call_tracker = 0
    strategy_switch = branching.iterations_until_stable + 1
    best_idx = -1
    all_stable = true
    branching_candidates = Int[]# this shall contain the indices of the potential branching variables
    values = Bonobo.get_relaxed_values(tree, node)
    branch_counter = []
    for idx in Bonobo.get_branching_indices(tree.root)
        value = values[idx]
        if !Bonobo.is_approx_feasible(tree, value)
            push!(branching_candidates, idx)
            push!(branch_counter, (branch_tracker[idx, 1], branch_tracker[idx, 2]))
            if branch_tracker[idx, 1] < strategy_switch || branch_tracker[idx, 2] < strategy_switch# check if pseudocost is stable for this idx 
                all_stable = false
            end
        end
    end
    # if rand() > 0.95# for debugging on if strategy behaves as intended
    #     println("\n branch_counter")
    #     println(branch_counter)
    # end
    if node.parent_lower_bound_base != Inf# if this node is a result of branching on some variable then update pseudocost of corresponding branching variable
        #println("if clause of update")
        idx = node.branched_on
        update = (tree.root.problem.f(values) - node.dual_gap) - node.parent_lower_bound_base
        if node.branched_right
            update = update / (ceil(values[idx]) - values[idx])  
            pseudos[idx, 1] = update_avg(update, pseudos[idx, 1], branch_tracker[idx, 1])
            branch_tracker[idx, 1] += 1

        else
            update = update / (values[idx] - floor(values[idx]))  
            pseudos[idx, 2] = update_avg(update, pseudos[idx, 2], branch_tracker[idx, 2])
            branch_tracker[idx, 2] += 1

        end
        # println("pseudos")
        # display(pseudos)
        # if rand() > 0.99# for debugging on if strategy behaves as intended
        #     println("\n branch_tracker")
        #     println(sum(branch_tracker))
        # end
    end

    length(branching_candidates) == 0 && return best_idx
    length(branching_candidates) == 1 && return branching_candidates[1]
    if !all_stable# THEN Use Most Infeasible
        max_distance_to_feasible = 0.0
        for i in branching_candidates
            value = values[i]
            distance_to_feasible = Bonobo.get_distance_to_feasible(tree, value)
            if distance_to_feasible > max_distance_to_feasible
                best_idx = i
                max_distance_to_feasible = distance_to_feasible
            end
        end
        #println(best_idx)
        return best_idx
    else
        # Pseudocosts have stabilized
        println("pseudocosts decision is made")
        #best_idx = argmax(map(idx-> maximum(pseudos[idx]), branching_candidates))# argmax randomly chosen and to be replaced later
        #inner = pseudos[idx, 1] * (values[idx] - floor(values[idx])), pseudos[idx, 2] * (ceil(values[idx]) - values[idx])

        branching_scores = map(idx-> ((1 - branching.μ) * min(pseudos[idx, 1] * (values[idx] - floor(values[idx])), pseudos[idx, 2] * (ceil(values[idx]) - values[idx])) + branching.μ * max(pseudos[idx, 1] * (values[idx] - floor(values[idx])), pseudos[idx, 2] * (ceil(values[idx]) - values[idx]))),
                            branching_candidates)
        branching_scores = sparsevec(branching_candidates, branching_scores)
        best_idx = argmax(branching_scores)      
        return best_idx 
    end
end

function update_avg(new_val::Float64, avg::Float64, N::Int)
    # N is the number of values used to compute the current avg
    # avg is the current average shifted by 1
    # new_val is the value that the current average has to be updated with
    if N > 1
        return 1/(N+1) * (N * (avg - 1) + new_val) + 1  #        
    else
        return new_val+ avg # note that we initialize the pseudo costs with 1 as such we need to shift pseudocosts correspondingly
    end
end
#pseudos = Dict{Int,Array{Float64}}(i=>zeros(2) for idx in get_integer_variables(blmo))

#branch_tracker = Dict{Int, Int}(idx-> 0 for idx in get_integer_variables(blmo))