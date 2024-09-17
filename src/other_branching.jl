"""
    Largest_Gradient <: AbstractBranchStrategy

The `LARGEST GRADIENT` strategy always picks the variable which has the largest absolute value entry in the current gradient and can be branched on.
"""
struct LARGEST_GRADIENT <: Bonobo.AbstractBranchStrategy end


"""
    get_branching_variable(
    tree::Bonobo.BnBTree, 
    node::Bonobo.AbstractNode
)

Get branching variable using Largest_Gradient branching 

"""
function Bonobo.get_branching_variable(
    tree::Bonobo.BnBTree, 
    branching::LARGEST_GRADIENT,
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
        if !Bonobo.is_approx_feasible(tree, value)# check if variable is branching candidate
            if abs(gradient_at_values[i]) > max_gradient
                best_idx = i
            end
        end
    end
    return best_idx
end



"""
    LARGEST_MOST_INFEASIBLE_GRADIENT <: AbstractBranchStrategy

The `LARGEST_MOST_INFEASIBLE_GRADIENT` strategy always picks the variable which has the largest absolute value 
entry in the current gradient multiplied by the maximum distance to being fixed.
"""

struct LARGEST_MOST_INFEASIBLE_GRADIENT <: Bonobo.AbstractBranchStrategy end


"""
    get_branching_variable(
    tree::Bonobo.BnBTree, 
    node::Bonobo.AbstractNode
)

Get branching variable using LARGEST_MOST_INFEASIBLE_GRADIENT branching 

"""
function Bonobo.get_branching_variable(
    tree::Bonobo.BnBTree, 
    branching::LARGEST_MOST_INFEASIBLE_GRADIENT,
    node::Bonobo.AbstractNode,
)   
    values = Bonobo.get_relaxed_values(tree, node)
    best_idx = -1
    nabla = similar(values)
    x_new = copy(values)
    gradient_at_values = tree.root.problem.g(nabla,x_new)# is this information already computed elsewhere?
    max_score = 0.0
    for i in branching_candidates
        if !Bonobo.is_approx_feasible(tree, value)
            value = values[i] * abs(gradient_at_values[i])
            if value > max_score
                best_idx = i
                max_score = value
            end
        end
    end
    return best_idx
end








