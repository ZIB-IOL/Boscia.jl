# FW callback
function build_FW_callback(
    tree,
    min_number_lower,
    check_rounding_value::Bool,
    fw_iterations,
    min_fw_iterations,
    time_ref,
    time_limit,
)
    vars = get_variables_pointers(tree.root.problem.tlmo.blmo, tree)
    # variable to only fetch heuristics when the counter increases
    ncalls = -1
    return function fw_callback(state, active_set, kwargs...)
        @assert isapprox(sum(active_set.weights), 1.0)
        @assert sum(active_set.weights .< 0) == 0
        # TODO deal with vertices becoming infeasible with conflicts
        @debug begin
            if !is_linear_feasible(tree.root.problem.tlmo, state.v)
                @info "$(state.v)"
                check_infeasible_vertex(tree.root.problem.tlmo.blmo, tree)
                @assert is_linear_feasible(tree.root.problem.tlmo, state.v)
            end  
            if state.step_type != FrankWolfe.ST_SIMPLEXDESCENT && !is_integer_feasible(tree, state.v)
                @info "Vertex not integer feasible! Here are the integer variables: $(state.v[tree.root.problem.integer_variables])"
                @assert is_integer_feasible(tree, state.v)
            end    
        end
        push!(fw_iterations, state.t)

        if state.lmo !== nothing  # can happen with using Blended Conditional Gradient
            if ncalls != state.lmo.ncalls
                ncalls = state.lmo.ncalls
                (best_v, best_val) = find_best_solution(
                    tree.root.problem.f,
                    tree.root.problem.tlmo.blmo,
                    vars,
                    tree.root.options[:domain_oracle],
                )
                if best_val < tree.incumbent
                    node = tree.nodes[tree.root.current_node_id[]]
                    add_new_solution!(tree, node, best_val, best_v, :Solver)
                    Bonobo.bound!(tree, node.id)
                end
            end
        end

        if (state.primal - state.dual_gap > tree.incumbent + 1e-2) &&
           tree.num_nodes != 1 &&
           state.t > min_fw_iterations
            return false
        end

        if tree.root.options[:domain_oracle](state.v) && state.step_type != FrankWolfe.ST_SIMPLEXDESCENT
            val = tree.root.problem.f(state.v)
            if val < tree.incumbent
                #TODO: update solution without adding node
                node = tree.nodes[tree.root.current_node_id[]]
                add_new_solution!(tree, node, val, copy(state.v), :vertex)
                Bonobo.bound!(tree, node.id)
            end
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

        # check for time limit
        if isfinite(time_limit) && Dates.now() >= time_ref + Dates.Second(time_limit)
            return false
        end

        return true
    end
end
