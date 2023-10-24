
struct PartialStrongBranching{BLMO<:BoundedLinearMinimizationOracle} <: Bonobo.AbstractBranchStrategy
    max_iteration::Int
    solving_epsilon::Float64
    bounded_lmo::BLMO
end

"""
Get branching variable using strong branching.
Create all possible subproblems, solve them and pick the one with the most progress.
"""
function Bonobo.get_branching_variable(
    tree::Bonobo.BnBTree,
    branching::PartialStrongBranching{BoundedLinearMinimizationOracle},
    node::Bonobo.AbstractNode,
)
    xrel = Bonobo.get_relaxed_values(tree, node)
    max_lowerbound = -Inf
    max_idx = -1
    @assert !isempty(node.active_set)
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
            push!(boundsLeft.upper_bounds, (idx => fxi))
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
                    if v[idx] <= xrel[idx]
                        push!(active_set, ((λ, v)))
                    end
                end
                @assert !isempty(active_set)
                FrankWolfe.active_set_renormalize!(active_set)
                _, _, primal_relaxed, dual_gap_relaxed, _ =
                    FrankWolfe.blended_pairwise_conditional_gradient(
                        tree.root.problem.f,
                        tree.root.problem.g,
                        branching.bounded_lmo,
                        active_set,
                        verbose=false,
                        epsilon=branching.solving_epsilon,
                        max_iteration=branching.max_iteration,
                    )
                left_relaxed = primal_relaxed - dual_gap_relaxed
            else
                @debug "Left non-optimal status $(status)"
                left_relaxed = Inf
            end

            #right node: x_i >=  floor(̂x_i)
            cxi = ceil(xrel[idx])
            boundsRight = copy(node.local_bounds)
            if haskey(boundsRight.lower_bounds, idx)
                delete!(boundsRight.lower_bounds, idx)
            end
            push!(boundsRight.lower_bounds, (idx => cxi))
            build_LMO(
                branching.bounded_lmo,
                tree.root.problem.integer_variable_bounds,
                boundsRight,
                Bonobo.get_branching_indices(tree.root),
            )
            status = check_feasibility(branching.bounded_lmo)
            if status == OPTIMALS
                empty!(active_set)
                for (λ, v) in node.active_set
                    if v[idx] >= xrel[idx]
                        push!(active_set, (λ, v))
                    end
                end
                if isempty(active_set)
                    @show xrel[idx]
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
                        max_iteration=branching.max_iteration,
                    )
                right_relaxed = primal_relaxed - dual_gap_relaxed
            else
                @debug "Right non-optimal status $(status)"
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
        max_idx = -1
    end
    return max_idx
end

"""
Hybrid between partial strong branching and another strategy.
`perform_strong_branch(tree, node) -> Bool` decides whether to perform strong branching or not.
"""
struct HybridStrongBranching{BLMO<:BoundedLinearMinimizationOracle,F<:Function,B<:Bonobo.AbstractBranchStrategy} <:
       Bonobo.AbstractBranchStrategy
    pstrong::PartialStrongBranching{BLMO}
    perform_strong_branch::F
    alternative_branching::B
end

function HybridStrongBranching(
    max_iteration::Int,
    solving_epsilon::Float64,
    bounded_lmo::BoundedLinearMinimizationOracle,
    perform_strong_branch::Function,
    alternative=Bonobo.MOST_INFEASIBLE(),
) 
    return HybridStrongBranching(
        PartialStrongBranching(max_iteration, solving_epsilon, bounded_lmo),
        perform_strong_branch,
        alternative,
    )
end

function Bonobo.get_branching_variable(
    tree::Bonobo.BnBTree,
    branching::HybridStrongBranching,
    node::Bonobo.AbstractNode,
)
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
function strong_up_to_depth(
    max_iteration::Int,
    solving_epsilon::Float64,
    bounded_lmo::BoundedLinearMinimizationOracle,
    max_depth::Int,
    alternative=Bonobo.MOST_INFEASIBLE(),
) 
    perform_strong_while_depth(_, node) = node.level <= max_depth
    return HybridStrongBranching(
        PartialStrongBranching(max_iteration, solving_epsilon, bounded_lmo),
        perform_strong_while_depth,
        alternative,
    )
end
