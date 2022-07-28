# FW callback
function build_FW_callback(tree, min_number_lower, check_rounding_value::Bool, fw_iterations)
    vars = [MOI.VariableIndex(var) for var in 1:tree.root.problem.nvars]
    # variable to only fetch heuristics when the counter increases
    ncalls = -1
    return function fw_callback(state, active_set)
        push!(fw_iterations, state.t)
        if ncalls != state.lmo.ncalls
            ncalls = state.lmo.ncalls
            (best_v, best_val) = find_best_solution(tree.root.problem.f, tree.root.problem.lmo.lmo.o, vars)
            if best_val < tree.incumbent
                node = tree.nodes[tree.root.current_node_id[]]
                sol = Bonobo.DefaultSolution(best_val, best_v, node)
                if isempty(tree.solutions)
                    push!(tree.solutions, sol)
                else
                    tree.solutions[1] = sol
                end
                tree.incumbent = best_val
                Bonobo.bound!(tree, node.id)
            end
        end

        if (state.primal - state.dual_gap > tree.incumbent)
            return false
        end

        val = tree.root.problem.f(state.v)
        if val < tree.incumbent
            #TODO: update solution without adding node
            node = tree.nodes[tree.root.current_node_id[]]
            sol = Bonobo.DefaultSolution(val, copy(state.v), node)
            if isempty(tree.solutions)
                push!(tree.solutions, sol)
            else
                tree.solutions[1] = sol
            end
            tree.incumbent = val
            Bonobo.bound!(tree, node.id)
        end

        if !isempty(tree.nodes) && min_number_lower <= length(values(tree.nodes))
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

        if check_rounding_value && state.tt == FrankWolfe.last
            # round values
            x_rounded = copy(state.x)
            for idx in tree.branching_indices
                x_rounded[idx] = round(state.x[idx])
            end
            # check linear feasibility
            if is_linear_feasible(tree.root.problem.lmo, x_rounded)
                 # evaluate f(rounded)
                val = tree.root.problem.f(x_rounded)
                if val < tree.incumbent
                    node = tree.nodes[tree.root.current_node_id[]]
                    sol = Bonobo.DefaultSolution(val, x_rounded, node)
                    if isempty(tree.solutions)
                        push!(tree.solutions, sol)
                    else
                        tree.solutions[1] = sol
                    end
                    tree.incumbent = val
                    Bonobo.bound!(tree, node.id)
                end
            end
        end

        return true
    end
end