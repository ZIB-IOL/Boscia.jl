
struct PartialStrongBranching{O} <: Bonobo.AbstractBranchStrategy
    max_iteration::Int
    solving_epsilon::Float64
    optimizer::O
end

function Bonobo.get_branching_variable(tree::Bonobo.BnBTree, branching::PartialStrongBranching, node::Bonobo.AbstractNode)
    xrel = Bonobo.get_relaxed_values(tree, node)
    max_lowerbound = -Inf
    max_idx = -1
    # copy problem and remove integer constraints
    filtered_src = MOI.Utilities.ModelFilter(tree.root.problem.lmo.lmo.o) do item
        if item isa Tuple
            (_, S) = item
            if S <: Union{MOI.Indicator, MOI.Integer, MOI.ZeroOne}
                return false
            end
        end
        return !(item isa MOI.ConstraintIndex{<:Any, <:Union{MOI.ZeroOne, MOI.Integer, MOI.Indicator}})
    end
    index_map = MOI.copy_to(branching.optimizer, filtered_src)
    # sanity check, otherwise the functions need permuted indices
    for (v1, v2) in index_map
        if v1 isa MOI.VariableIndex
            @assert v1 == v2
        end
    end
    relaxed_lmo = FrankWolfe.MathOptLMO(branching.optimizer)
    active_set = copy(node.active_set)
    empty!(active_set)
    num_frac = 0
    for idx in Bonobo.get_branching_indices(tree.root)
        if !isapprox(xrel[idx], round(xrel[idx]), atol=tree.options.atol, rtol=tree.options.rtol)
            # left node: x_i <=  floor(̂x_i)
            fxi = floor(xrel[idx])
            # create LMO
            boundsLeft = copy(node.local_bounds)
            if haskey(boundsLeft.upper_bounds, idx)
                delete!(boundsLeft.upper_bounds, idx)
            end
            push!(boundsLeft.upper_bounds, (idx => MOI.LessThan(fxi)))
            build_LMO(relaxed_lmo, tree.root.problem.integer_variable_bounds, boundsLeft, Bonobo.get_branching_indices(tree.root))
            MOI.optimize!(relaxed_lmo.o)
            if MOI.get(relaxed_lmo.o, MOI.TerminationStatus()) == MOI.OPTIMAL
                empty!(active_set)
                for (λ, v) in node.active_set
                    if v[idx] <= fxi
                        push!(active_set, ((λ, v)))
                    end
                end
                @assert !isempty(active_set)
                FrankWolfe.active_set_renormalize!(active_set)
                _, _, primal_relaxed, dual_gap_relaxed, _ = FrankWolfe.blended_pairwise_conditional_gradient(tree.root.problem.f, tree.root.problem.g, relaxed_lmo, active_set, epsilon=branching.solving_epsilon, max_iteration=branching.max_iteration)
                left_relaxed = primal_relaxed - dual_gap_relaxed
            else
                @debug "Left non-optimal status $(MOI.get(relaxed_lmo.o, MOI.TerminationStatus()))"
                left_relaxed = Inf
            end
            #right node: x_i >=  floor(̂x_i)
            cxi = ceil(xrel[idx])
            boundsRight = copy(node.local_bounds)
            if haskey(boundsRight.lower_bounds, idx)
                delete!(boundsRight.lower_bounds, idx)
            end
            push!(boundsRight.lower_bounds, (idx => MOI.GreaterThan(cxi)))
            build_LMO(relaxed_lmo, tree.root.problem.integer_variable_bounds, boundsRight, Bonobo.get_branching_indices(tree.root))
            MOI.optimize!(relaxed_lmo.o)
            if MOI.get(relaxed_lmo.o, MOI.TerminationStatus()) == MOI.OPTIMAL
                empty!(active_set)
                for (λ, v) in node.active_set
                    if v[idx] >= cxi
                        push!(active_set, ((λ, v)))
                    end
                end
                @assert !isempty(active_set)
                FrankWolfe.active_set_renormalize!(active_set)
                _, _, primal_relaxed, dual_gap_relaxed, _ = FrankWolfe.blended_pairwise_conditional_gradient(tree.root.problem.f, tree.root.problem.g, relaxed_lmo, active_set, epsilon=branching.solving_epsilon, max_iteration=branching.max_iteration)
                right_relaxed = primal_relaxed - dual_gap_relaxed
            else
                @debug "Right non-optimal status $(MOI.get(relaxed_lmo.o, MOI.TerminationStatus()))"
                right_relaxed = Inf
            end
            # lowest lower bound on the two branches
            lowerbound_increase = min(left_relaxed, right_relaxed)
            if lowerbound_increase > max_lowerbound
                max_lowerbound = lowerbound_increase
                max_idx = idx
            end
            num_frac += 1
        end
    end
    @debug "strong branching: index $max_idx, lower bound $max_lowerbound"
    if max_idx <= 0 && num_frac != 0
        error("Infeasible node! Please check constraints! node lb: $(node.lb)")
        max_idx = -1
    end
    if max_idx <= 0
        @info "Integral, node lb: $(node.lb)"
        max_idx = -1
    end
    return max_idx
end

using Bonobo

"""
Hybrid between partial strong branching and another strategy.
`perform_strong_branch(tree, node) -> Bool` decides whether to perform strong branching or not.
"""
struct HybridStrongBranching{O, F <: Function, B <: Bonobo.AbstractBranchStrategy} <: Bonobo.AbstractBranchStrategy
    pstrong::PartialStrongBranching{O}
    perform_strong_branch::F
    alternative_branching::B
end

function HybridStrongBranching(max_iteration::Int, solving_epsilon::Float64, optimizer::O, perform_strong_branch::Function, alternative = Bonobo.MOST_INFEASIBLE()) where {O}
    return HybridStrongBranching(PartialStrongBranching(max_iteration, solving_epsilon, optimizer), perform_strong_branch, alternative)
end

function Bonobo.get_branching_variable(tree::Bonobo.BnBTree, branching::HybridStrongBranching, node::Bonobo.AbstractNode)
    do_strong_branch = branching.perform_strong_branch(tree, node)
    return if do_strong_branch
        Bonobo.get_branching_variable(tree, branching.pstrong, node)
    else
        Bonobo.get_branching_variable(tree, branching.alternative_branching, node)
    end
end

"""
strong_up_to_depth performs strong branching on nodes up to a predetermined depth, and the falls back to another rule
"""
function strong_up_to_depth(max_iteration::Int, solving_epsilon::Float64, optimizer::O, max_depth::Int, alternative = Bonobo.MOST_INFEASIBLE()) where {O}
    perform_strong_while_depth(_, node) = node.level <= max_depth
    return HybridStrongBranching(PartialStrongBranching(max_iteration, solving_epsilon, optimizer), perform_strong_while_depth, alternative)
end
