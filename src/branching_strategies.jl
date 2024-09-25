using SparseArrays

struct PSEUDO_COST{BLMO<:BoundedLinearMinimizationOracle} <: Bonobo.AbstractBranchStrategy
    iterations_until_stable::Int
    stable::Bool
    bounded_lmo::BLMO
    μ::Float64
    decision_function::String
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
    node::Bonobo.AbstractNode,
    pseudos::SparseMatrixCSC,
    branch_tracker::SparseMatrixCSC
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

    strategy_switch = branching.iterations_until_stable + 1
    best_idx = -1
    all_stable = true
    branching_candidates = Int[]# this shall contain the indices of the potential branching variables
    values = Bonobo.get_relaxed_values(tree, node)
    for idx in tree.branching_indices
        value = values[idx]
        if !Bonobo.is_approx_feasible(tree, value)
            push!(branching_candidates, idx)
            if branch_tracker[idx, 1] < strategy_switch || branch_tracker[idx, 2] < strategy_switch# check if pseudocost is stable for this idx 
                all_stable = false
            end
        end
    end
    # if this node is a result of branching on some variable then update pseudocost of corresponding branching variable
    if !isinf(node.parent_lower_bound_base)
        idx = node.branched_on
        update = (tree.root.problem.f(values) - node.dual_gap) - node.parent_lower_bound_base
        update = update / node.distance_to_int
        if isinf(update)
            @debug "update is $(Inf)"
        end
        if node.branched_right
            pseudos[idx, 1] = update_avg(update, pseudos[idx, 1], branch_tracker[idx, 1])
            branch_tracker[idx, 1] += 1
        else
            pseudos[idx, 2] = update_avg(update, pseudos[idx, 2], branch_tracker[idx, 2])
            branch_tracker[idx, 2] += 1

        end
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
        return best_idx
    else
        if branching.decision_function == "weighted_sum"
            branching_scores = map(idx-> ((1 - branching.μ) * min((pseudos[idx, 1] - 1) * (values[idx] - floor(values[idx])), (pseudos[idx, 2] - 1) * (ceil(values[idx]) - values[idx])) + branching.μ * max((pseudos[idx, 1] - 1) * (values[idx] - floor(values[idx])), (pseudos[idx, 2] - 1) * (ceil(values[idx]) - values[idx]))),
                                branching_candidates)
        
        elseif branching.decision_function == "product"
            branching_scores = map(idx-> max((pseudos[idx, 1] - 1) * (values[idx] - floor(values[idx])), branching.μ) * max((pseudos[idx, 2] - 1) * (ceil(values[idx]) - values[idx]), branching.μ), 
                                branching_candidates)
        end
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
        # note that we initialize the pseudo costs with 1 as such we need to shift pseudocosts correspondingly
        return new_val + avg 
    end
end


struct HIERARCHY_PSEUDO_COST{BLMO<:BoundedLinearMinimizationOracle} <: Bonobo.AbstractBranchStrategy
    iterations_until_stable::Int
    gradient_influence::Bool
    bounded_lmo::BLMO
    μ::Float64
    decision_function::String
end


"""
    get_branching_variable(
    tree::Bonobo.BnBTree, 
    branching::HIERARCHY_PSEUDO_COST,
    node::Bonobo.AbstractNode,
    pseudos::SparseMatrixCSC,
    branch_tracker::SparseMatrixCSC
)
This strategy first chooses the branching varaibles which have most often led to infeasiblity.
If there are multiple such candidates then among these candidates another strategy is used 
    if candidate pseudocosts are stable then a pseudocost decision is made 
    else
        if gradient_influence=false
            decision is made based on MOST_INFEASIBLE strategy
        else 
            decision is made on LARGEST_MOST_INFEASIBLE_GRADIENT  

"""
function Bonobo.get_branching_variable(
    tree::Bonobo.BnBTree, 
    branching::HIERARCHY_PSEUDO_COST{BLMO},
    node::Bonobo.AbstractNode,
    pseudos::SparseMatrixCSC{Float64, Int64},
    branch_tracker::SparseMatrixCSC{Int64, Int64},
    infeas_tracker::SparseMatrixCSC{Int64, Int64},
) where BLMO <: BoundedLinearMinimizationOracle

    strategy_switch = branching.iterations_until_stable + 1
    best_idx = -1
    all_stable = true
    # the following shall contain the indices of the potential branching variables
    branching_candidates = Int[]
    values = Bonobo.get_relaxed_values(tree, node)
    for idx in tree.branching_indices
        value = values[idx]
        if !Bonobo.is_approx_feasible(tree, value)
            push!(branching_candidates, idx)
        end
    end
    # if this node is a result of branching on some variable then update pseudocost of corresponding branching variable
    if !isinf(node.parent_lower_bound_base)
        idx = node.branched_on
        update = (tree.root.problem.f(values) - node.dual_gap) - node.parent_lower_bound_base
        update = update / node.distance_to_int
        if isinf(update)
            @debug "update is $(Inf)"
        end
        if node.branched_right
            pseudos[idx, 1] = update_avg(update, pseudos[idx, 1], branch_tracker[idx, 1])
            branch_tracker[idx, 1] += 1
        else
            pseudos[idx, 2] = update_avg(update, pseudos[idx, 2], branch_tracker[idx, 2])
            branch_tracker[idx, 2] += 1

        end
    end
    # compute score of how (often) branching on a variable resulted in infeasiblity
    best_score = max_infeas_score(branching_candidates, infeas_tracker)
    branching_candidates = Int[idx for idx in branching_candidates if infeas_score(idx, infeas_tracker) >= best_score]
    if length(branching_candidates) == 0 
        return best_idx
    elseif length(branching_candidates) == 1
        best_idx = branching_candidates[1]
        infeas_tracker[best_idx, 2] += 1
        return best_idx
    end

    for idx in branching_candidates
        # check if pseudocost is stable for this idx
        if branch_tracker[idx, 1] < strategy_switch || branch_tracker[idx, 2] < strategy_switch 
            all_stable = false
        end
    end
    
    if !all_stable# THEN Use Most Infeasible
        if branching.gradient_influence
            # score function as product of gradient at variable and largest distance to int
            nabla = similar(values)
            x_new = copy(values)
            gradient_at_values = tree.root.problem.g(nabla,x_new)
            max_score = 0.0
            for i in branching_candidates
                value = values[i] * abs(gradient_at_values[i])
                if value > max_score
                    best_idx = i
                    max_score = value
                end
            end
            return best_idx
        else
            max_distance_to_feasible = 0.0
            for i in branching_candidates
                value = values[i]
                distance_to_feasible = Bonobo.get_distance_to_feasible(tree, value)
                if distance_to_feasible > max_distance_to_feasible
                    best_idx = i
                    max_distance_to_feasible = distance_to_feasible
                end
            end
            infeas_tracker[best_idx, 2] += 1
            return best_idx
        end
    else
        if branching.decision_function == "weighted_sum"
            branching_scores = map(idx-> ((1 - branching.μ) * min((pseudos[idx, 1] - 1) * (values[idx] - floor(values[idx])), (pseudos[idx, 2] - 1) * (ceil(values[idx]) - values[idx])) + branching.μ * max((pseudos[idx, 1] - 1) * (values[idx] - floor(values[idx])), (pseudos[idx, 2] - 1) * (ceil(values[idx]) - values[idx]))),
                                branching_candidates)
        
        elseif branching.decision_function == "product"
            branching_scores = map(idx-> max((pseudos[idx, 1] - 1) * (values[idx] - floor(values[idx])), branching.μ) * max((pseudos[idx, 2] - 1) * (ceil(values[idx]) - values[idx]), branching.μ), 
                                branching_candidates)
        end
        branching_scores = sparsevec(branching_candidates, branching_scores)
        best_idx = argmax(branching_scores)
        infeas_tracker[best_idx, 2] += 1
        return best_idx
    end
end


function infeas_score(idx::Int, infeas_tracker::SparseMatrixCSC{Int64, Int64})
    infeas_tracker[idx, 1] == 1 && return 0
    # ratio of how often branching on variable idx leads to node infeasiblity of children
    return  (infeas_tracker[idx, 1]) / (infeas_tracker[idx, 2])
end

function max_infeas_score(idxs::Vector{Int64}, infeas_tracker::SparseMatrixCSC{Int64, Int64})
    max_score = 0
    for idx in idxs
        if infeas_tracker[idx, 1] == 1
            continue
        else
            max_score = max(max_score, (infeas_tracker[idx, 1]) / (infeas_tracker[idx, 2]))
        end
    end

    return max_score
end




"""
    LargestGradient <: AbstractBranchStrategy

The `LargestGradient` strategy always picks the variable which has the largest absolute value entry in the current gradient and can be branched on.
"""
struct LargestGradient <: Bonobo.AbstractBranchStrategy end


"""
    get_branching_variable(
    tree::Bonobo.BnBTree, 
    node::Bonobo.AbstractNode
)

Get branching variable which has the largest absolute value in the gradient

"""
function Bonobo.get_branching_variable(
    tree::Bonobo.BnBTree, 
    branching::LargestGradient,
    node::Bonobo.AbstractNode,
) 
    values = Bonobo.get_relaxed_values(tree, node)
    nabla = similar(values)
    x_new = copy(values)
    gradient_at_values = tree.root.problem.g(nabla, x_new)
    best_idx = -1
    max_gradient = 0.0
    for i in tree.branching_indices
        value = values[i]
        # check if variable is branching candidate
        if !Bonobo.is_approx_feasible(tree, value)
            if abs(gradient_at_values[i]) > max_gradient
                best_idx = i
            end
        end
    end
    return best_idx
end



"""
    LargestMostInfeasibleGradient <: AbstractBranchStrategy

The `LargestMostInfeasibleGradient` strategy always picks the variable which has the largest absolute value 
entry in the current gradient multiplied by the maximum distance to being fixed.
"""

struct LargestMostInfeasibleGradient <: Bonobo.AbstractBranchStrategy end


"""
    get_branching_variable(
    tree::Bonobo.BnBTree, 
    node::Bonobo.AbstractNode
)

Get branching variable using LARGEST_MOST_INFEASIBLE_GRADIENT branching 

"""
function Bonobo.get_branching_variable(
    tree::Bonobo.BnBTree, 
    branching::LargestMostInfeasibleGradient,
    node::Bonobo.AbstractNode,
)   
    values = Bonobo.get_relaxed_values(tree, node)
    best_idx = -1
    nabla = similar(values)
    x_new = copy(values)
    gradient_at_values = tree.root.problem.g(nabla,x_new)
    max_score = 0.0
    for i in tree.branching_indices
        value = values[i]
        if !Bonobo.is_approx_feasible(tree, value)
            value *= abs(gradient_at_values[i])
            if value > max_score
                best_idx = i
                max_score = value
            end
        end
    end
    return best_idx
end








