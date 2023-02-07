# FW callback
function build_FW_callback(tree, min_number_lower, check_rounding_value::Bool, fw_iterations, min_fw_iterations)
    vars = [MOI.VariableIndex(var) for var in 1:tree.root.problem.nvars]
    # variable to only fetch heuristics when the counter increases
    ncalls = -1
    return function fw_callback(state, active_set, args...)
        @assert isapprox(sum(active_set.weights), 1.0)
        @assert sum(active_set.weights .< 0) == 0
        if !is_linear_feasible(tree.root.problem.lmo, state.v)
            @info "$v"
        end
        @assert is_linear_feasible(tree.root.problem.lmo, state.v)
        @assert is_linear_feasible(tree.root.problem.lmo, state.x)

        push!(fw_iterations, state.t)
        if ncalls != state.lmo.ncalls
            ncalls = state.lmo.ncalls
            (best_v, best_val) =
                find_best_solution(tree.root.problem.f, tree.root.problem.lmo.lmo.o, vars)
            if best_val < tree.incumbent
                tree.root.updated_incumbent[] = true
                node = tree.nodes[tree.root.current_node_id[]]
                sol = FrankWolfeSolution(best_val, best_v, node, :SCIP)
                push!(tree.solutions, sol)
                if tree.incumbent_solution === nothing ||
                   sol.objective < tree.incumbent_solution.objective
                    tree.incumbent_solution = sol
                end
                tree.incumbent = best_val
                Bonobo.bound!(tree, node.id)
            end
        end

        if (state.primal - state.dual_gap > tree.incumbent + 1e-2) && tree.num_nodes != 1 && state.t > min_fw_iterations
            return false
        end

        val = tree.root.problem.f(state.v)
        if val < tree.incumbent
            tree.root.updated_incumbent[] = true
            #TODO: update solution without adding node
            node = tree.nodes[tree.root.current_node_id[]]
            sol = FrankWolfeSolution(val, copy(state.v), node, :vertex)
            push!(tree.solutions, sol)
            if tree.incumbent_solution === nothing ||
               sol.objective < tree.incumbent_solution.objective
                tree.incumbent_solution = sol
            end
            tree.incumbent = val
            Bonobo.bound!(tree, node.id)
        end

        node = tree.nodes[tree.root.current_node_id[]]
        if length(node.active_set) > 1 &&
           !isempty(tree.nodes) &&
           min_number_lower <= length(values(tree.nodes))
            counter = 0
            for n in values(tree.nodes)
                if n.lb < val
                    counter += 1
                end
                if counter > min_number_lower
                    return false
                end
            end
        end

        if check_rounding_value && state.tt == FrankWolfe.pp
            # round values
            x_rounded = copy(state.x)
            for idx in tree.branching_indices
                x_rounded[idx] = round(state.x[idx])
            end
            # check linear feasibility
            if is_linear_feasible(tree.root.problem.lmo, x_rounded) && is_integer_feasible(tree, x_rounded)
                # evaluate f(rounded)
                val = tree.root.problem.f(x_rounded)
                if val < tree.incumbent
                    tree.root.updated_incumbent[] = true
                    node = tree.nodes[tree.root.current_node_id[]]
                    sol = FrankWolfeSolution(val, x_rounded, node, :rounded)
                    push!(tree.solutions, sol)
                    if tree.incumbent_solution === nothing ||
                       sol.objective < tree.incumbent_solution.objective
                        tree.incumbent_solution = sol
                    end
                    tree.incumbent = val
                    Bonobo.bound!(tree, node.id)
                end
            end
        end

        return true
    end
end
