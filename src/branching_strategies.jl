using SparseArrays

mutable struct PSEUDO_COST <: Bonobo.AbstractBranchStrategy
    iterations_until_stable::Int
    alternative::String
    μ::Float64
    decision_function::String
    pseudos::SparseMatrixCSC{Float64, Int64}
    branch_tracker::SparseMatrixCSC{Int64, Int64}
    function PSEUDO_COST(
        iterations_until_stable,
        bounded_lmo,
        μ,
        decision_function
        ) 
        int_vars = Boscia.get_integer_variables(bounded_lmo)
        int_var_number = length(int_vars)
        # create sparse array for pseudocosts
        pseudos = sparse(
            repeat(int_vars, 2),
            vcat(ones(int_var_number), 2*ones(int_var_number)), 
            ones(2 * int_var_number)
            )
        # create sparse array for keeping track of how often a pseudocost has been updated
        branch_tracker = sparse(
            repeat(int_vars, 2),
            vcat(ones(int_var_number), 2*ones(int_var_number)), 
            ones(Int64, 2 * int_var_number)
            )
        new(
            iterations_until_stable,
            alternative,
            μ,
            decision_function,
            pseudos,
            branch_tracker)
    end
end


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
    branching::PSEUDO_COST,
    node::Bonobo.AbstractNode,
)
    strategy_switch = branching.iterations_until_stable + 1
    best_idx = -1
    all_stable = true
    # this shall contain the indices of the potential branching variables
    branching_candidates = Int[]
    values = Bonobo.get_relaxed_values(tree, node)
    for idx in tree.branching_indices
        value = values[idx]
        if !Bonobo.is_approx_feasible(tree, value)
            push!(branching_candidates, idx)
            if branching.branch_tracker[idx, 1] < strategy_switch || branching.branch_tracker[idx, 2] < strategy_switch# check if pseudocost is stable for this idx 
                all_stable = false
            end
        end
    end
    update_pseudocost!(tree, node, branching, values)
    length(branching_candidates) == 0 && return best_idx
    length(branching_candidates) == 1 && return branching_candidates[1]
    if !all_stable# THEN Use alternative
        if branching.alternative == "largest_most_infeasible_gradient"
            best_idx = largest_most_infeasible_gradient_decision(tree, branching_candidates, values)
        elseif branching.alternative == "largest_gradient"
            best_idx = largest_gradient_decision(tree, branching_candidates, values)
        else
            best_idx = most_infeasible_decision(tree, branching_candidates, values)
        end
        return best_idx
    else
        # All candidates pseudo stable thus make pseudocost decision
        return pseudocost_decision(branching, branching_candidates, values)
    end
end

"""
    update_avg(
    new_val::Float64, 
    avg::Float64, 
    N::Int)
    N is the number of values used to compute the current avg
    avg is the current average shifted by 1
    new_val is the value that the current average has to be updated with
"""
    function update_avg(new_val::Float64, avg::Float64, N::Int)

    if N > 1
        return 1/(N+1) * (N * (avg - 1) + new_val) + 1  #        
    else
        # note that we initialize the pseudo costs with 1 as such we need to shift pseudocosts correspondingly
        return new_val + avg 
    end
end


mutable struct HIERARCHY_PSEUDO_COST <: Bonobo.AbstractBranchStrategy
    iterations_until_stable::Int
    alternative::String
    μ::Float64
    decision_function::String
    pseudos::SparseMatrixCSC{Float64, Int64}
    branch_tracker::SparseMatrixCSC{Int64, Int64}
    infeas_tracker::SparseMatrixCSC{Int64, Int64}

    function HIERARCHY_PSEUDO_COST(
        iterations_until_stable,
        alternative,
        bounded_lmo,
        μ,
        decision_function
        ) 
        int_vars = Boscia.get_integer_variables(bounded_lmo)
        int_var_number = length(int_vars)
        # create sparse array for pseudocosts
        pseudos = sparse(
            repeat(int_vars, 2),
            vcat(ones(int_var_number), 2*ones(int_var_number)), 
            ones(2 * int_var_number)
            )
        # create sparse array for keeping track of how often a pseudocost has been updated
        branch_tracker = sparse(
            repeat(int_vars, 2),
            vcat(ones(int_var_number), 2*ones(int_var_number)), 
            ones(Int64, 2 * int_var_number)
            )
        # create sparse array for keeping track of how often a branching has resulted in infeas.
        # of resulting node
        infeas_tracker = deepcopy(branch_tracker)
        new(
            iterations_until_stable,
            alternative,
            μ,
            decision_function,
            pseudos,
            branch_tracker,
            infeas_tracker)
    end
end

"""
    get_branching_variable(
    tree::Bonobo.BnBTree, 
    branching::HIERARCHY_PSEUDO_COST,
    node::Bonobo.AbstractNode,
    pseudos::SparseMatrixCSC,
    branch_tracker::SparseMatrixCSC
)
Description:
This strategy first chooses the branching variables which have most often led to infeasiblity.
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
    branching::HIERARCHY_PSEUDO_COST,
    node::Bonobo.AbstractNode,
) 
    # indices of branching candidates
    values = Bonobo.get_relaxed_values(tree, node)
    branching_candidates = get_branching_candidates(tree, node, values)
    update_pseudocost!(tree, node, branching, values)
    if isempty(branching_candidates)
        #branching is no longer possible 
        return -1
    end
    # Top level Selection criteria: Choose what leads most often to infeasible child nodes
    best_score = max_infeas_score(branching_candidates, branching.infeas_tracker)
    # keep candidates with highest score only
    branching_candidates = Int[idx for idx in branching_candidates if infeas_score(idx, branching.infeas_tracker) >= best_score]

    if length(branching_candidates) == 1
        # if only one candidate exists all strategies will choose it
        best_idx = branching_candidates[1]
    elseif !candidates_pseudo_stable(branching, branching_candidates)
        if branching.alternative == "largest_most_infeasible_gradient"
            best_idx = largest_most_infeasible_gradient_decision(tree, branching_candidates, values)
        elseif branching.alternative == "largest_gradient"
            best_idx = largest_gradient_decision(tree, branching_candidates, values)
        else
            best_idx = most_infeasible_decision(tree, branching_candidates, values)
        end
    else
        best_idx = pseudocost_decision(branching, branching_candidates, values)
    end
    # update number of times the candidate has been branched on
    branching.infeas_tracker[best_idx, 2] += 1
    return best_idx
end


"""
largest_most_infeasible_gradient_decision(
    tree::Bonobo.BnBTree,
    branching_candidates::Vector{Int},
    values::Vector{Float64}
)
\nDescription:
\n choose by LARGEST_MOST_INFEASIBLE_GRADIENT 
among candidates in branching_candidates

"""
function largest_most_infeasible_gradient_decision(
    tree::Bonobo.BnBTree,
    branching_candidates::Vector{Int},
    values::Vector{Float64}
)
    nabla = similar(values)
    x_new = copy(values)
    gradient_at_values = tree.root.problem.g(nabla,x_new)
    max_score = 0.0
    best_idx = -1
    for i in branching_candidates
        value = values[i] 
        value = abs(value - round(value)) * abs(gradient_at_values[i])
        if value >= max_score
            best_idx = i
            max_score = value
        end
    end
    return best_idx
end


"""
most_infeasible_decision(
    tree::Bonobo.BnBTree,
    branching_candidates::Vector{Int},
    values::Vector{Float64}
)   
\nDescription:
\n Choose by largest distance to next integer feasible solution
among the candidates in branching_candidates 
"""
function most_infeasible_decision(
    tree::Bonobo.BnBTree,
    branching_candidates::Vector{Int},
    values::Vector{Float64}
)   
    best_idx = -1
    max_distance_to_feasible = 0.0
    for idx in branching_candidates
        value = values[idx]
        distance_to_feasible = Bonobo.get_distance_to_feasible(tree, value)
        if distance_to_feasible > max_distance_to_feasible
            best_idx = idx
            max_distance_to_feasible = distance_to_feasible
        end
    end
    return best_idx
end

"""
largest_gradient_decision(
    tree::Bonobo.BnBTree,
    branching_candidates::Vector{Int},
    values::Vector{Float64}
)
\nDescription:
\n Decision is made based on highest abs value entry in gradient for branching candidates
"""
function largest_gradient_decision(
    tree::Bonobo.BnBTree,
    branching_candidates::Vector{Int},
    values::Vector{Float64}
)   
    nabla = similar(values)
    gradient_at_values = tree.root.problem.g(nabla, nabla)
    best_idx = -1
    max_gradient = 0.0
    for idx in branching_candidates
        if abs(gradient_at_values[idx]) >= max_gradient
            best_idx = idx
        end
    end
    return best_idx
end


"""
pseudocost_decision(
    branching::Bonobo.AbstractBranchStrategy,
    branching_candidates::Vector{Int},
    values::Vector{Float64}
)
\nDescription: 
\nPerform a pseudocost decision based 
on the choice of decision function defined in 
branching.decision_function
Returns: index of chosen candidate variable
"""
function pseudocost_decision(
    branching::Bonobo.AbstractBranchStrategy,
    branching_candidates::Vector{Int},
    values::Vector{Float64}
)
    if branching.decision_function == "weighted_sum"
        return weighted_sum_decision(
            branching_candidates,
            branching.μ, 
            branching.pseudos, 
            values
        )  
    elseif branching.decision_function == "product"
        return product_decision(
            branching_candidates,
            branching.μ, 
            branching.pseudos, 
            values
        )
    end
end
"""
update_pseudocost!(
    tree::Bonobo.BnBTree,
    node::FrankWolfeNode,
    branching::Bonobo.AbstractBranchStrategy,
    values::Vector{Float64}
)  
\nDescription: 
\n-Updates the pseudocost of the index that
the current node resulted from.\n 
-Distinction being made if it was a left or right branch
and if node is a result of branching at all.
"""
function update_pseudocost!(
    tree::Bonobo.BnBTree,
    node::FrankWolfeNode,
    branching::Bonobo.AbstractBranchStrategy,
    values::Vector{Float64}
)
    if !isinf(node.parent_lower_bound_base)
        idx = node.branched_on
        update = (tree.root.problem.f(values) - node.dual_gap) - node.parent_lower_bound_base
        update = update / node.distance_to_int
        if isinf(update)
            @debug "update is $(Inf)"
        end
        if node.branched_right
            branching.pseudos[idx, 1] = update_avg(update, branching.pseudos[idx, 1], branching.branch_tracker[idx, 1])
            branching.branch_tracker[idx, 1] += 1
        else
            branching.pseudos[idx, 2] = update_avg(update, branching.pseudos[idx, 2], branching.branch_tracker[idx, 2])
            branching.branch_tracker[idx, 2] += 1
        end
    end
end


"""
candidates_pseudo_stable(
    branching::Bonobo.AbstractBranchStrategy,
    branching_candidates::Vector{Int}
) 
\nDescription:
\n Checks if all branching candidates have stable pseudocosts.
\n Returns: true or false
"""
function candidates_pseudo_stable(
    branching::Bonobo.AbstractBranchStrategy,
    branching_candidates::Vector{Int}
)   
    strategy_switch = branching.iterations_until_stable + 1
    all_stable = true
    for idx in branching_candidates
        # check if pseudocost is stable for this candidate
        if branching.branch_tracker[idx, 1] < strategy_switch || branching.branch_tracker[idx, 2] < strategy_switch 
            all_stable = false
            break
        end
    end
    return all_stable
end

"""
get_branching_candidates(
    tree::Bonobo.BnBTree, 
    node::FrankWolfeNode,
    values::Vector{Float64}
) 
\nDescription:
\n finds all possible branching candidates at the current node
"""
function get_branching_candidates(
    tree::Bonobo.BnBTree, 
    node::FrankWolfeNode,
    values::Vector{Float64}
)   
    branching_candidates = Int[]
    for idx in tree.branching_indices
        value = values[idx]
        if !Bonobo.is_approx_feasible(tree, value)
            push!(branching_candidates, idx)
        end
    end
    return branching_candidates
end
"""
weighted_sum_decision(
    idx::Int, 
    branching_candidates::Vector{Int},
    branching::Bonobo.AbstractBranchStrategy, 
    pseudos::SparseMatrixCSC{Float64, Int64}, 
    values::Vector{Float64}
)
\nDescription: 
\nThis method first scales pseudocosts of candidate variables 
by distance to integer solution. Then it calculates a score as a convex combination 
of each pseudocost  pair using the branching.μ parameter. Lastly it chooses the 
candidate which has the highest score, i.e. whose convex combination value is largest.
"""
function weighted_sum_decision(
    branching_candidates::Vector{Int},
    μ::Float64, 
    pseudos::SparseMatrixCSC{Float64, Int64}, 
    values::Vector{Float64}
)   
    best_idx = branching_candidates[1]
    best_score = 0
    for idx in branching_candidates
        unit_cost_pseudos = unit_cost_pseudo_tuple(
            pseudos[idx, 2], 
            pseudos[idx, 1], 
            values[idx]
            )
        score = pseudocost_convex_combination(unit_cost_pseudos, μ)
        if score > best_score
            best_idx = idx
            best_score = best_score
        end
    end
    # calc convex comb. of each unit cost scaled pseudo pair
    branching_scores = map(
        idx-> pseudocost_convex_combination(
            unit_cost_pseudo_tuple(
                pseudos[idx, 2], 
                pseudos[idx, 1], 
                values[idx]
            ),
            μ
        ),
        branching_candidates)
    # return candidate with highest value
    return argmax(sparsevec(branching_candidates, branching_scores))
end

"""
function product_decision(
    branching_candidates::Vector{Int},
    μ::Float64, 
    pseudos::SparseMatrixCSC{Float64, Int64}, 
    values::Vector{Float64}
)   
\nDescription: 
\nMakes decision based on highest μ_product of pseudocosts
for branching candidate
"""
function product_decision(
    branching_candidates::Vector{Int},
    μ::Float64, 
    pseudos::SparseMatrixCSC{Float64, Int64}, 
    values::Vector{Float64}
)   
# calc μ_product of each unit cost scaled pseudo pair
    branching_scores = map(
        idx-> μ_product(
            unit_cost_pseudo_tuple(
                pseudos[idx, 2], 
                pseudos[idx, 1], 
                values[idx]
            ),
            μ
        ),
        branching_candidates)
    # return candidate with highest μ_product 
    return argmax(sparsevec(branching_candidates, branching_scores))
end


"""
pseudocost_convex_combination(
    pseudocost_tuple::Tuple{Float64,Float64},  
    μ::Float64
    )
Description: Calculates a convex combination of elements contained in the tuple pseudocost_tuple
"""
function pseudocost_convex_combination(
    pseudocost_tuple::Tuple{Float64,Float64},  
    μ::Float64
    )
    return (1 - μ) * minimum(pseudocost_tuple) + μ * maximum(pseudocost_tuple)
end

"""
    unit_cost_pseudo_tuple(
    pd::Float64, 
    pu::Float64, 
    value::Float64
)
\nDescription: 
\nMultiply up and down branch pseudocost of branching candidate 
by the distance of the current solution (value) to the respective 
next integer solution. 
\nThereby performing unit cost scaling of pseudocosts. 
The -1 is a result of pseudocosts being intitalized as 1.
\nReturns: tuple containing the two unit cost scaled pseudocosts.
"""
function unit_cost_pseudo_tuple(
    left_pseudo::Float64, 
    right_pseudo::Float64, 
    value::Float64
)
    return ((left_pseudo - 1) * (value - floor(value)), 
            (right_pseudo - 1) * (ceil(value) - value))
end


"""
μ_product(
    pseudocost_tuple::Tuple{Float64, Float64},
    μ::Float64
)
\nDescription: 
\nMultiplies elements of tuple while replacing elements 
smaller than μ by μ in the product  
"""
function μ_product(
    pseudocost_tuple::Tuple{Float64, Float64},
    μ::Float64
)
    return max(pseudocost_tuple[1], μ) * max(pseudocost_tuple[2], μ) 
end


"""
infeas_score(
    idx::Int, 
    infeas_tracker::SparseMatrixCSC{Int64, Int64}
    )
Description: calculates ratio of how often branching on candidate idx
             has led to infeasibilty of a resulting node.
"""
function infeas_score(
    idx::Int, 
    infeas_tracker::SparseMatrixCSC{Int64, Int64}
    )
    infeas_tracker[idx, 1] == 1 && return 0
    # ratio of how often branching on variable idx leads to node infeasiblity of children
    return  (infeas_tracker[idx, 1]) / (infeas_tracker[idx, 2])
end

"""
max_infeas_score(
    idxs::Vector{Int64}, 
    infeas_tracker::SparseMatrixCSC{Int64, Int64}
    )
\nDescription: 
\nCalculates maximum infeasibilty score over all branching candidates 
\n Infeasibility score as ratio of how (often) branching on a variable resulted in infeasiblity of resulting node
            
"""
function max_infeas_score(
    idxs::Vector{Int64}, 
    infeas_tracker::SparseMatrixCSC{Int64, Int64}
    )
    max_score = 0
    for idx in idxs
        max_score = max(max_score, infeas_score(idx, infeas_tracker))
    end
    return max_score
end




"""
    LargestGradient <: AbstractBranchStrategy

The `LargestGradient` strategy always picks the variable which 
has the largest absolute value entry in the current gradient 
and can be branched on.
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
    gradient_at_values = tree.root.problem.g(nabla, nabla)
    best_idx = -1
    max_gradient = 0.0
    for i in tree.branching_indices
        value = values[i]
        # check if variable is branching candidate
        if !Bonobo.is_approx_feasible(tree, value)
            if abs(gradient_at_values[i]) >= max_gradient
                best_idx = i
            end
        end
    end
    return best_idx
end


"""
    LargestMostInfeasibleGradient <: AbstractBranchStrategy

The `LargestMostInfeasibleGradient` strategy always picks the variable which 
has the largest absolute value entry in the current gradient multiplied
by the maximum distance to being fixed.
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
    gradient_at_values = tree.root.problem.g(nabla,nabla)
    max_score = 0.0
    for i in tree.branching_indices
        value = values[i]
        if !Bonobo.is_approx_feasible(tree, value)
            value = abs(value - round(value))
            value *= abs(gradient_at_values[i])
            if value >= max_score
                best_idx = i
                max_score = value
            end
        end
    end
    return best_idx
end








