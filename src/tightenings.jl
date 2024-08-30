"""
Tightening of the bounds at node level. Children node inherit the updated bounds.
"""
function dual_tightening(tree, node, x, dual_gap)
    if tree.root.options[:dual_tightening] && isfinite(tree.incumbent)
        grad = similar(x)
        tree.root.problem.g(grad, x)
        num_tightenings = 0
        num_potential_tightenings = 0
        μ = tree.root.options[:strong_convexity]
        for j in tree.root.problem.integer_variables
            lb_global = get(tree.root.problem.integer_variable_bounds, (j, :greaterthan), -Inf)
            ub_global = get(tree.root.problem.integer_variable_bounds, (j, :lessthan), Inf)
            lb = get(node.local_bounds.lower_bounds, j, lb_global)
            ub = get(node.local_bounds.upper_bounds, j, ub_global)
            @assert lb >= lb_global
            @assert ub <= ub_global
            if lb ≈ ub
                # variable already fixed
                continue
            end
            gj = grad[j]
            safety_tolerance = 2.0
            rhs =
                tree.incumbent - tree.root.problem.f(x) +
                safety_tolerance * dual_gap +
                sqrt(eps(tree.incumbent))
            if ≈(x[j], lb, atol=tree.options.atol, rtol=tree.options.rtol)
                if !isapprox(gj, 0, atol=1e-5)
                    num_potential_tightenings += 1
                end
                if gj > 0
                    Mlb = 0
                    bound_tightened = true
                    @debug "starting tightening ub $(rhs)"
                    while 0.99 * (Mlb * gj + μ / 2 * Mlb^2) <= rhs
                        Mlb += 1
                        if lb + Mlb - 1 == ub
                            bound_tightened = false
                            break
                        end
                    end
                    if bound_tightened
                        new_bound = lb + Mlb - 1
                        @debug "found UB tightening $ub -> $new_bound"
                        node.local_bounds[j, :lessthan] = new_bound
                        num_tightenings += 1
                        if haskey(tree.root.problem.integer_variable_bounds, (j, :lessthan))
                            @assert node.local_bounds[j, :lessthan] <=
                                    tree.root.problem.integer_variable_bounds[j, :lessthan]
                        end
                    end
                end
            elseif ≈(x[j], ub, atol=tree.options.atol, rtol=tree.options.rtol)
                if !isapprox(gj, 0, atol=1e-5)
                    num_potential_tightenings += 1
                end
                if gj < 0
                    Mub = 0
                    bound_tightened = true
                    @debug "starting tightening lb $(rhs)"
                    while -0.99 * (Mub * gj + μ / 2 * Mub^2) <= rhs
                        Mub += 1
                        if ub - Mub + 1 == lb
                            bound_tightened = false
                            break
                        end
                    end
                    if bound_tightened
                        new_bound = ub - Mub + 1
                        @debug "found LB tightening $lb -> $new_bound"
                        node.local_bounds[j, :greaterthan] = new_bound
                        num_tightenings += 1
                        if haskey(tree.root.problem.integer_variable_bounds, (j, :greaterthan))
                            @assert node.local_bounds[j, :greaterthan] >=
                                    tree.root.problem.integer_variable_bounds[j, :greaterthan]
                        end
                    end
                end
            end
        end
        @debug "# tightenings $num_tightenings"
        node.local_tightenings = num_tightenings
        node.local_potential_tightenings = num_potential_tightenings
    end
end

"""
Save the gradient of the root solution (i.e. the relaxed solution) and the 
corresponding lower and upper bounds.
"""
function store_data_global_tightening(tree, node, x, dual_gap)
    if tree.root.options[:global_dual_tightening] && node.std.id == 1
        @debug "storing root node info for tightening"
        grad = similar(x)
        tree.root.problem.g(grad, x)
        safety_tolerance = 2.0
        tree.root.global_tightening_rhs[] = -tree.root.problem.f(x) + safety_tolerance * dual_gap
        for j in tree.root.problem.integer_variables
            if haskey(tree.root.problem.integer_variable_bounds.upper_bounds, j)
                ub = tree.root.problem.integer_variable_bounds[j, :lessthan]
                if ≈(x[j], ub, atol=tree.options.atol, rtol=tree.options.rtol) && grad[j] < 0
                    tree.root.global_tightening_root_info.upper_bounds[j] = (grad[j], ub)
                end
            end
            if haskey(tree.root.problem.integer_variable_bounds.lower_bounds, j)
                lb = tree.root.problem.integer_variable_bounds[j, :greaterthan]
                if ≈(x[j], lb, atol=tree.options.atol, rtol=tree.options.rtol) && grad[j] > 0
                    tree.root.global_tightening_root_info.lower_bounds[j] = (grad[j], lb)
                end
            end
        end
    end
end

"""
Use the gradient of the root node to tighten the global bounds.
"""
function global_tightening(tree, node)
    # new incumbent: check global fixings
    if tree.root.options[:global_dual_tightening] && tree.root.updated_incumbent[]
        num_tightenings = 0
        rhs = tree.incumbent + tree.root.global_tightening_rhs[]
        @assert isfinite(rhs)
        for (j, (gj, lb)) in tree.root.global_tightening_root_info.lower_bounds
            ub = get(tree.root.problem.integer_variable_bounds.upper_bounds, j, Inf)
            ub_new = get(tree.root.global_tightening_root_info.upper_bounds, j, Inf)
            ub = min(ub, ub_new)
            Mlb = 0
            bound_tightened = true
            lb = lb
            while Mlb * gj <= rhs
                Mlb += 1
                if lb + Mlb - 1 == ub
                    bound_tightened = false
                    break
                end
            end
            if bound_tightened
                new_bound = lb + Mlb - 1
                @debug "found global UB tightening $ub -> $new_bound"
                if haskey(tree.root.global_tightenings.upper_bounds, j)
                    if tree.root.global_tightenings.upper_bounds[j] != new_bound
                        num_tightenings += 1
                    end
                else
                    num_tightenings += 1
                end
                tree.root.global_tightenings.upper_bounds[j] = new_bound
            end
        end
        for (j, (gj, ub)) in tree.root.global_tightening_root_info.upper_bounds
            lb = get(tree.root.problem.integer_variable_bounds.lower_bounds, j, -Inf)
            lb_new = get(tree.root.global_tightening_root_info.lower_bounds, j, -Inf)
            lb = max(lb, lb_new)
            Mub = 0
            bound_tightened = true
            ub = ub
            while -Mub * gj <= rhs
                Mub += 1
                if ub - Mub + 1 == lb
                    bound_tightened = false
                    break
                end
            end
            if bound_tightened
                new_bound = ub - Mub + 1
                @debug "found global LB tightening $lb -> $new_bound"
                if haskey(tree.root.global_tightenings.lower_bounds, j)
                    if tree.root.global_tightenings.lower_bounds[j] != new_bound
                        num_tightenings += 1
                    end
                else
                    num_tightenings += 1
                end
                tree.root.global_tightenings.lower_bounds[j] = new_bound
            end
        end
        node.global_tightenings = num_tightenings
    end
end

"""
Tighten the lower bound using strong convexity of the objective.
"""
function tightening_strong_convexity(tree, x, lower_bound)
    μ = tree.root.options[:strong_convexity]
    if μ > 0
        @debug "Using strong convexity $μ"
        strong_convexity_bound = lower_bound
        num_fractional = 0
        for j in tree.root.problem.integer_variables
            if x[j] > floor(x[j]) + 1e-6 && x[j] < ceil(x[j]) - 1e-6
                num_fractional += 1
                new_left_increment = μ / 2 * (x[j] - floor(x[j]))^2
                new_right_increment = μ / 2 * (ceil(x[j]) - x[j])^2
                new_increment = min(new_left_increment, new_right_increment)
                strong_convexity_bound += new_increment
            end
        end
        @debug "Strong convexity: $lower_bound -> $strong_convexity_bound"
        @assert num_fractional == 0 || strong_convexity_bound > lower_bound
        lower_bound = strong_convexity_bound
    end
    return lower_bound
end

"""
Use strong convexity to potentially remove one of the children nodes 
"""
function prune_children(tree, node, lower_bound_base, x, vidx)
    prune_left = false
    prune_right = false

    μ = tree.root.options[:strong_convexity]
    if μ > 0
        @debug "Using strong convexity $μ"
        for j in tree.root.problem.integer_variables
            if vidx == j
                continue
            end
            lower_bound_base += μ / 2 * min((x[j] - floor(x[j]))^2, (ceil(x[j]) - x[j])^2)
        end
        new_bound_left = lower_bound_base + μ / 2 * (x[vidx] - floor(x[vidx]))^2
        new_bound_right = lower_bound_base + μ / 2 * (ceil(x[vidx]) - x[vidx])^2
        if new_bound_left > tree.incumbent
            @debug "prune left, from $(node.lb) -> $new_bound_left, ub $(tree.incumbent), lb $(node.lb)"
            prune_left = true
        end
        if new_bound_right > tree.incumbent
            @debug "prune right, from $(node.lb) -> $new_bound_right, ub $(tree.incumbent), lb $(node.lb)"
            prune_right = true
        end
        @assert !(
            (new_bound_left > tree.incumbent + tree.root.options[:dual_gap]) &&
            (new_bound_right > tree.incumbent + tree.root.options[:dual_gap])
        ) "both sides should not be pruned"
    end

    # If both nodes are pruned, when one of them has to be equal to the incumbent.
    # Thus, we have proof of optimality by strong convexity.
    if prune_left && prune_right
        tree.lb = min(new_bound_left, new_bound_right)
    end

    return prune_left, prune_right
end
