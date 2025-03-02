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
    x_new = copy(values)
    tree.root.problem.g(nabla,x_new)
    best_idx = -1
    max_gradient = 0.0
    for idx in tree.branching_indices
        value = values[idx]
        # check if variable is branching candidate
        if !Bonobo.is_approx_feasible(tree, value)
            if abs(nabla[idx]) >= max_gradient 
                best_idx = idx
                max_gradient = abs(nabla[idx])
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
    x_new = copy(values)
    tree.root.problem.g(nabla,x_new)
    max_score = 0.0
    for i in tree.branching_indices
        value = values[i]
        if !Bonobo.is_approx_feasible(tree, value)
            value = abs(value - round(value))
            value *= abs(nabla[i])
            if value >= max_score
                best_idx = i
                max_score = value
            end
        end
    end
    return best_idx
end

"""
    LargestIndex <: AbstractBranchStrategy

Always returns the largest index
"""
struct LargestIndex	 <: Bonobo.AbstractBranchStrategy end

function Bonobo.get_branching_variable(
    tree::Bonobo.BnBTree, 
    branching::LargestIndex,
    node::Bonobo.AbstractNode,
) 
    values = Bonobo.get_relaxed_values(tree, node)
    best_idx = -1
    # tree.branching_indices is sorted 
    for idx in tree.branching_indices
        value = values[idx]
        # check if variable is branching candidate
        if !Bonobo.is_approx_feasible(tree, value)
            best_idx = idx
        end
    end
    return best_idx
end


"""
    RandomBranching <: AbstractBranchStrategy

Return a random index
"""
struct RandomBranching <: Bonobo.AbstractBranchStrategy end

function Bonobo.get_branching_variable(
    tree::Bonobo.BnBTree, 
    branching::RandomBranching,
    node::Bonobo.AbstractNode,
) 
    values = Bonobo.get_relaxed_values(tree, node)
    # tree.branching_indices is sorted 
    branching_candidates = Int64[]
    for idx in tree.branching_indices
        value = values[idx]
        # check if variable is branching candidate
        if !Bonobo.is_approx_feasible(tree, value)
            push!(branching_candidates, idx)
        end
    end
    isempty(branching_candidates) && return -1

    return rand(branching_candidates)
end

##############################################################################################
#           FUNCTIONS and STRUCTURES NEEDED WITHIN Hierarchy Branching   
##############################################################################################



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
    tree.root.problem.g(nabla,x_new)
    max_score = 0.0
    best_idx = -1
    for i in branching_candidates
        value = values[i] 
        value = abs(value - round(value)) * abs(nabla[i])
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
    x_new = copy(values)
    tree.root.problem.g(nabla,x_new)
    best_idx = -1
    max_gradient = 0.0
    for idx in branching_candidates
        if abs(nabla[idx]) >= max_gradient
            best_idx = idx
            max_gradient = abs(nabla[idx])
        end
    end
    return best_idx
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
`CutoffFunctionGenerator` generates functions that calculate a cutoff value based on a vector of scores.
The cutoff is calculated as: `max(cutoff_type * maximum(scores) + (1 - cutoff_type) * mean(scores), min_cutoff)`
"""
struct CutoffFunctionGenerator
    cutoff_type::Float64
    min_cutoff::Float64
end

"""
Calculates the cutoff value for a vector of scores.
"""
function (gen::CutoffFunctionGenerator)(scores::Vector{Float64})
    if gen.cutoff_type == 1.0
        return max(maximum(scores), gen.min_cutoff)
    elseif gen.cutoff_type == 0.0
        return  max(mean(scores), gen.min_cutoff)
    else
        return max(gen.cutoff_type * maximum(scores) + (1 - gen.cutoff_type) * mean(scores), gen.min_cutoff)
    end
end

"""
`SelectionGenerator` generates functions that calculate a selection based on name and cutoff_f
"""
struct SelectionGenerator
    name::String
    cutoff_f::Union{Function, CutoffFunctionGenerator}
    alt_name::Union{String, Missing}
    alt_cutoff_f::Union{Function, CutoffFunctionGenerator, Missing}
    decision_function::Union{String, Missing}
    iterations_until_stable::Int64
    μ:: Float64
    comparison_type::String
    alt_final_flag::Bool
    SelectionGenerator(
        name::String,
        cutoff_f::Union{Function, CutoffFunctionGenerator};
        alt_name::Union{String, Missing} = missing,
        alt_cutoff_f::Union{Function, CutoffFunctionGenerator, Missing} = missing,
        decision_function::Union{String, Missing} = missing,
        iterations_until_stable::Int64 = 1,
        μ::Float64 = 1e-6,
        comparison_type::String = ">",
        alt_final_flag::Bool = false,
    ) = new(name, cutoff_f, alt_name, alt_cutoff_f, decision_function, iterations_until_stable + 1, μ, comparison_type, alt_final_flag)
end
"""
Calculates the Candidate Selection based on the cutoff function and name
"""
function (gen::SelectionGenerator)(
    tree::Bonobo.BnBTree, 
    branching::Bonobo.AbstractBranchStrategy, 
    values::Vector{Float64}, 
    candidates::Vector{Int64}
    )
    if gen.name == "largest_gradient"
        nabla = similar(values)
        x_new = copy(values)
        tree.root.problem.g(nabla,x_new)
        scores = Float64[abs(nabla[idx]) for idx in candidates]
        cutoff = gen.cutoff_f(scores)
    elseif gen.name == "largest_most_infeasible_gradient"
        nabla = similar(values)
        x_new = copy(values)
        tree.root.problem.g(nabla,x_new)
        scores = Float64[Bonobo.get_distance_to_feasible(tree, values[idx]) * abs(nabla[idx]) for idx in candidates]
        cutoff = gen.cutoff_f(scores)
    elseif gen.name == "most_infeasible"
        scores = Float64[Bonobo.get_distance_to_feasible(tree, values[idx]) for idx in candidates]
        cutoff = gen.cutoff_f(scores)
    elseif gen.name == "pseudocost"
        all_stable = true
        for idx in candidates
            # check if pseudocost is stable for this candidate
            if branching.branch_tracker[idx, 1] < gen.iterations_until_stable || branching.branch_tracker[idx, 2] < gen.iterations_until_stable 
                all_stable = false
                break
            end
        end
        if all_stable
            if gen.decision_function == "product"
                scores = map(
                    idx-> μ_product(
                        unit_cost_pseudo_tuple(
                            branching.pseudos[idx, 2], 
                            branching.pseudos[idx, 1], 
                            values[idx]
                        ),
                        gen.μ
                    ),
                    candidates)
            elseif gen.decision_function == "weighted_sum"
                scores = map(
                    idx-> pseudocost_convex_combination(
                        unit_cost_pseudo_tuple(
                            branching.pseudos[idx, 2], 
                            branching.pseudos[idx, 1], 
                            values[idx]
                        ),
                        gen.μ
                    ),
                    candidates)
            elseif gen.decision_function == "minimum"
                scores = map(
                    idx-> minimum(
                        unit_cost_pseudo_tuple(
                            branching.pseudos[idx, 2], 
                            branching.pseudos[idx, 1], 
                            values[idx]
                        )
                    ),
                    candidates)
            end
            # Compute cutoff based on pseudocost scores
            cutoff = gen.cutoff_f(scores)
        else
            if gen.alt_final_flag
                if gen.alt_name == "largest_gradient"
                    return Int64[largest_gradient_decision(tree, candidates, values)]
                elseif gen.alt_name == "largest_most_infeasible_gradient"
                    return Int64[largest_most_infeasible_gradient_decision(tree, candidates, values)]
                elseif gen.alt_name == "most_infeasible"
                    return Int64[most_infeasible_decision(tree, candidates, values)]
                end
            else 
                if gen.alt_name == "largest_gradient"
                    nabla = similar(values)
                    x_new = copy(values)
                    tree.root.problem.g(nabla,x_new)
                    scores = Float64[abs(nabla[idx]) for idx in candidates]
            
                elseif gen.alt_name == "largest_most_infeasible_gradient"
                    nabla = similar(values)
                    x_new = copy(values)
                    tree.root.problem.g(nabla,x_new)
                    scores = Float64[Bonobo.get_distance_to_feasible(tree, values[idx]) * abs(nabla[idx]) for idx in candidates]
                    
                elseif gen.alt_name == "most_infeasible"
                    scores = Float64[Bonobo.get_distance_to_feasible(tree, values[idx]) for idx in candidates]
                end
                # Compute cutoff based on alternative criteria scores
                cutoff = gen.alt_cutoff_f(scores)
            end

        end
    end
    # Calculate which candidates are good enough as per the cutoff 
    if gen.comparison_type == ">"
        return  [idx for (i, idx) in enumerate(candidates) if scores[i] > cutoff]
    else 
        return [idx for (i, idx) in enumerate(candidates) if scores[i] >= cutoff]
    end
end


"""
mutable struct Stage
    name::String
    selection_f::function
    decision_counter::Int64 = 0
    min_cutoff_counter::Int64 = 0
end   
This is used to allow a flexible choice of criteria in hierarchy branching. 
The last two parameters provide information on how the criterium contributed to decisions.
"""
mutable struct Stage
    name::String
    selection_criterion::Union{Function, SelectionGenerator}
    decision_counter::Int64
    min_cutoff_counter::Int64
    function Stage(
        name, 
        selection_f
    )
        new(name, selection_f, 0, 0)
    end 
end 
########################################################################################
#                Hierarchy Branching 
########################################################################################

mutable struct Hierarchy <: Bonobo.AbstractBranchStrategy
    pseudos::SparseMatrixCSC{Float64, Int64}
    branch_tracker::SparseMatrixCSC{Int64, Int64}
    stages::Vector{Stage}
    function Hierarchy(
        bounded_lmo;
        stages = []
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

        # If Stages == [] we set the stages by default 
        # stages determine order of criteria 
        if isempty(stages)
            stages = default_hierarchy_strategies()
        end
        new(
            pseudos,
            branch_tracker,
            stages)
    end
end


function Bonobo.get_branching_variable(
    tree::Bonobo.BnBTree, 
    branching::Hierarchy,
    node::Bonobo.AbstractNode,
) 
    # indices of branching candidates
    values = Bonobo.get_relaxed_values(tree, node)
    branching_candidates = get_branching_candidates(tree, node, values)
    update_pseudocost!(tree, node, branching, values)
    if isempty(branching_candidates)
        #branching is not possible 
        return -1
    end
    for (i, stage) in enumerate(branching.stages)
        
        # As per criterium defined in the decision function of the stage
        # find the candidates which are "good" enough
        remaining_candidates = stage.selection_criterion(tree, branching, values, branching_candidates)
        if isempty(remaining_candidates)
            # If a minimum cutoff value > 0 is set then no candidate might be left 
            # We in this case select a random candidate
            stage.min_cutoff_counter += 1
            return rand(branching_candidates)   

        elseif length(remaining_candidates) == 1
            # Final candidate was chosen at this stage 
            stage.decision_counter += 1
            return remaining_candidates[1]
        end
        # Pass the remaining candidates to next stage
        branching_candidates = remaining_candidates
    end
    # just in case that after the last criterium there is more than one candidate left
    # we choose at random from the final selection 
    return rand(branching_candidates) 
end
  
############################################################################ ###################################### ######################################
#                                      Default Stages/Setting for Hierarchy Branching                                                                        #
############################################################################ ###################################### ######################################
function default_hierarchy_strategies(
    name::String = "most_infeasible",# first stage criterium
    alt_name::String = "most_infeasible",# second stage pseudocost with alternative defined by alt_name
    iterations_until_stable::Int64 = 1,
    decision_function::String = "product",
) 
    if name == "most_infeasible"
        # cutoffs for different stages 
        
        cutoff_1 = CutoffFunctionGenerator(0.5, 1e-3)
        cutoff_2 = CutoffFunctionGenerator(0.75, 0.0)
        cutoff_alt = CutoffFunctionGenerator(1.0, 0.0)

        # Selection criteria for different stages 
        func_1 = SelectionGenerator(
            "most_infeasible", 
            cutoff_1
        )
        if decision_function == "weighted_sum"
            μ = 0.5
        else 
            μ = 1e-6
        end
        func_2 = SelectionGenerator(
            "pseudocost", # 
            cutoff_2; # stable cutoff 
            alt_name = alt_name,# alternative decision function
            alt_cutoff_f = cutoff_alt, # cutoff for alternative
            decision_function = decision_function, # stable decision function for pseu#
            iterations_until_stable = iterations_until_stable,# number of iterations until a variable is deemed stable
            comparison_type = ">=",
            alt_final_flag=true,
            μ = μ
        )
        func_3 = SelectionGenerator(
            "largest_most_infeasible_gradient", 
            cutoff_alt,
            comparison_type=">="
        )
        # Actual stages 
        stage1 = Stage("most_infeasible", func_1)
        stage2 = Stage("pseudocost", func_2)
        stage3 = Stage("largest_most_infeasible_gradient", func_3)

        return [stage1, stage2, stage3]
    end
end

"""
function create_binary_stage(
    bounded_lmo
)
   Creates a Stage for Hierarchy Branching where non binary variables are 
    filtered out when binary variables exist.
"""
function create_binary_stage(
    bounded_lmo
)
    binary_vars = Set{Int64}(getproperty.(get_binary_variables(bounded_lmo), :value))
    function select_binary_vars(
        tree::Bonobo.BnBTree, 
        branching::Bonobo.AbstractBranchStrategy, 
        values::Vector{Float64}, 
        candidates::Vector{Int64}
        )
        remaining_candidates = Int64[]
        for idx in candidates
            if idx in binary_vars
                push!(remaining_candidates, idx)
            end
        end
        if !isempty(remaining_candidates)
            return remaining_candidates
        else
            return candidates
        end
    end

    return Boscia.Stage("binary", select_binary_vars)
end

###############################################################################################################################################################################
### Reworked PseudocostBranching
###############################################################################################################################################################################


"""
StableChecker Structure for checking stabilization of pseudocosts
"""
struct StableChecker
    iterations_until_stable::Int64
    StableChecker(
    iterations_until_stable = 1
    ) = new(iterations_until_stable)
end

"""
Calculates if pseudocosts  are stable.
"""
function (gen::StableChecker)(
    branch_tracker::SparseMatrixCSC{Int64, Int64}, 
    branching_candidates::Vector{Int64}
    )
    all_stable = true
    for idx in branching_candidates
        if branch_tracker[idx, 1] < gen.iterations_until_stable || branch_tracker[idx, 2] < gen.iterations_until_stable 
            all_stable = false
            break
        end
    end
    return all_stable
end
"""
`PseudocostStableSelectionGenerator` generates decision function for stable pseudocost branching 
"""
struct PseudocostStableSelectionGenerator
    decision_function::Union{String, Missing}
    μ:: Float64
    PseudocostStableSelectionGenerator(
        decision_function = "product",
        μ::Float64 = 1e-6
    ) = new(decision_function, μ)
end

"""
Calculates the Candidate Selection based on pseudocost decision function
"""
function (gen::PseudocostStableSelectionGenerator)(
    tree::Bonobo.BnBTree, 
    branching::Bonobo.AbstractBranchStrategy, 
    values::Vector{Float64}, 
    candidates::Vector{Int64}
    )
    best_idx = -1
    best_score = 0
    for idx in candidates
        if gen.decision_function == "product"
            score = μ_product(
                    unit_cost_pseudo_tuple(
                        branching.pseudos[idx, 2], 
                        branching.pseudos[idx, 1], 
                        values[idx]
                    ),
                    gen.μ
                ) 
        elseif gen.decision_function == "weighted_sum"
            score = pseudocost_convex_combination(
                    unit_cost_pseudo_tuple(
                        branching.pseudos[idx, 2], 
                        branching.pseudos[idx, 1], 
                        values[idx]
                    ),
                    gen.μ
                )
        elseif gen.decision_function == "minimum"
            score = minimum(
                    unit_cost_pseudo_tuple(
                        branching.pseudos[idx, 2], 
                        branching.pseudos[idx, 1], 
                        values[idx]
                    )
                )
        end
        if score >= best_score
            best_score = score
            best_idx = idx
        end
    end
    return best_idx
end





mutable struct PseudocostBranching <: Bonobo.AbstractBranchStrategy
    pseudos::SparseMatrixCSC{Float64, Int64}
    branch_tracker::SparseMatrixCSC{Int64, Int64}
    alt_f::Function
    stable_f:: Union{Function, PseudocostStableSelectionGenerator}
    stable_checker::StableChecker
    alt_decision_number::Int64
    stable_decision_number::Int64
    function PseudocostBranching(
        bounded_lmo;
        alt_f = most_infeasible_decision,
        stable_f = PseudocostStableSelectionGenerator("product", 1e-6),
        iterations_until_stable = 1
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
        # get function that checks if pseudocosts are stable. +1 because of init of branch_tracker
        stable_checker = StableChecker(iterations_until_stable + 1)

        new(
            pseudos,
            branch_tracker,
            alt_f,
            stable_f,
            stable_checker,
            0,
            0)
    end
end


function Bonobo.get_branching_variable(
    tree::Bonobo.BnBTree, 
    branching::PseudocostBranching,
    node::Bonobo.AbstractNode,
) 

    values = Bonobo.get_relaxed_values(tree, node)
    #indices of branching candidates
    branching_candidates = get_branching_candidates(tree, node, values)
    update_pseudocost!(tree, node, branching, values)
    if isempty(branching_candidates)
        #branching is not possible 
        return -1
    end
    # check if pseudocosts are stable
    if branching.stable_checker(branching.branch_tracker, branching_candidates)
        branching.stable_decision_number += 1
        return branching.stable_f(tree, branching, values, branching_candidates)

    else 
        branching.alt_decision_number += 1
        return branching.alt_f(tree, branching_candidates, values)
    end
end
  

