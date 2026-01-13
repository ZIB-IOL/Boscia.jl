"""
Frank-Wolfe Callback.

Is called in every Frank-Wolfe iteration.
Node evaluation can be dynamically stopped here.
Time limit is checked.
If the vertex is providing a better incumbent, it is added as solution.
"""
function build_FW_callback(
    tree,
    min_number_lower,
    check_rounding_value::Bool,
    fw_iterations,
    min_fw_iterations,
    time_ref,
    time_limit;
    use_DICG=false,
)
    vars = get_variables_pointers(tree.root.problem.tlmo.lmo, tree)
    # variable to only fetch heuristics when the counter increases
    ncalls = -1
    if !use_DICG
        return function (state, active_set, kwargs...)
            return process_FW_callback_logic(
                tree,
                state,
                vars,
                fw_iterations,
                ncalls,
                min_fw_iterations,
                min_number_lower,
                time_ref,
                time_limit,
                use_DICG;
                active_set=active_set,
                kwargs,
            )
        end
    else
        return function (state, pre_computed_set, kwargs...)
            return process_FW_callback_logic(
                tree,
                state,
                vars,
                fw_iterations,
                ncalls,
                min_fw_iterations,
                min_number_lower,
                time_ref,
                time_limit,
                use_DICG;
                pre_computed_set=pre_computed_set,
                kwargs,
            )
        end
    end
end

function process_FW_callback_logic(
    tree,
    state,
    vars,
    fw_iterations,
    ncalls,
    min_fw_iterations,
    min_number_lower,
    time_ref,
    time_limit,
    use_DICG;
    active_set=nothing,
    pre_computed_set=nothing,
    kwargs...,
)

    if !use_DICG
        @assert isapprox(sum(active_set.weights), 1.0) "sum(active_set.weights) = $(sum(active_set.weights))"
        @assert sum(active_set.weights .< 0) == 0
    end

    # TODO deal with vertices becoming infeasible with conflicts
    @debug begin
        if !is_linear_feasible(tree.root.problem.tlmo, state.v)
            @info "$(state.v)"
            check_infeasible_vertex(tree.root.problem.tlmo.lmo, tree)
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
                tree,
                tree.root.problem.f,
                tree.root.problem.tlmo.lmo,
                vars,
                tree.root.options[:domain_oracle],
            )
            if best_val < tree.incumbent && !tree.root.options[:add_all_solutions]
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
        if val < tree.incumbent || tree.root.options[:add_all_solutions]
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

    # push FW vertex into pre_computed_set for DICG
    if pre_computed_set != nothing
        if state.step_type != FrankWolfe.last || state.step_type != FrankWolfe.pp
            idx = findfirst(x -> x == state.v, pre_computed_set)
            if idx == nothing
                if length(pre_computed_set) > (length(state.v) + 1)
                    deleteat!(pre_computed_set, 1)
                end
                push!(pre_computed_set, state.v)
            end
        end
    end

    return true

end

"""
Branch-and-Bound Callback.
Collects statistics and prints logs if verbose is turned on.

Output of Boscia:
    iter :          current iteration of Boscia
    node id :       current node id
    lower bound :   tree_lb(tree)
    incumbent :     tree.incumbent
    gap :           tree.incumbent-tree_lb(tree)
    rel. gap :      dual_gap/tree.incumbent
    time :          total time of Boscia
    time/nodes :    average time per node
    FW time :       time spent in FW 
    LMO time :      time used by LMO
    LMO calls :     number of compute_extreme_point calls in FW
    FW iterations : number of iterations in FW
"""
function build_bnb_callback(
    tree,
    time_ref,
    list_lb_cb,
    list_ub_cb,
    list_time_cb,
    list_num_nodes_cb,
    list_lmo_calls_cb,
    verbose,
    fw_iterations,
    list_active_set_size_cb,
    list_discarded_set_size_cb,
    result,
    lmo_calls_per_layer,
    active_set_size_per_layer,
    discarded_set_size_per_layer,
    node_level,
    baseline_callback,
    local_tightenings,
    global_tightenings,
    local_potential_tightenings,
    num_bin,
    num_int,
)
    iteration = 0

    headers = [
        " ",
        "Iter",
        "Open",
        "Bound",
        "Incumbent",
        "Gap (abs)",
        "Gap (rel)",
        "Time (s)",
        "Nodes/sec",
        "FW (ms)",
        "LMO (ms)",
        "LMO (calls c)",
        "FW (its)",
        "#activeset",
        "#shadow",
    ]
    format_string = "%1s %5i %5i %14e %14e %14e %14e %14e %14e %12i %10i %14i %10i %8i %8i\n"
    print_iter = get(tree.root.options, :print_iter, 100)

    if verbose
        FrankWolfe.print_callback(headers, format_string, print_header=true)
    end
    return function callback(
        tree,
        node;
        worse_than_incumbent=false,
        node_infeasible=false,
        lb_update=false,
    )
        if baseline_callback !== nothing
            baseline_callback(
                tree,
                node,
                worse_than_incumbent=worse_than_incumbent,
                node_infeasible=node_infeasible,
                lb_update=lb_update,
            )
        end
        if !node_infeasible
            #update lower bound
            if lb_update == true
                tree.node_queue[node.id] = (node.lb, node.id)
                _, prio = peek(tree.node_queue)
                @assert tree.lb <= prio[1]
                tree.lb = min(minimum([prio[2][1] for prio in tree.node_queue]), tree.incumbent)
            end
            push!(list_ub_cb, tree.incumbent)
            push!(list_num_nodes_cb, tree.num_nodes)
            push!(node_level, node.std.depth)
            iteration += 1
            if tree.lb == -Inf && isempty(tree.nodes)
                tree.lb = node.lb
            end

            time = float(Dates.value(Dates.now() - time_ref))
            push!(list_time_cb, time)

            if tree.root.options[:time_limit] < Inf
                if time / 1000.0 ≥ tree.root.options[:time_limit]
                    @assert tree.root.problem.solving_stage == SOLVING
                    tree.root.problem.solving_stage = TIME_LIMIT_REACHED
                end
            end

            if iteration > tree.root.options[:node_limit]
                tree.root.problem.solving_stage = NODE_LIMIT_REACHED
            end

            fw_time = Dates.value(node.fw_time)
            fw_iter = if !isempty(fw_iterations)
                fw_iterations[end]
            else
                0
            end
            if !isempty(tree.root.problem.tlmo.optimizing_times)
                LMO_time = sum(1000 * tree.root.problem.tlmo.optimizing_times)
                empty!(tree.root.problem.tlmo.optimizing_times)
            else
                LMO_time = 0
            end
            LMO_calls_c = tree.root.problem.tlmo.ncalls
            push!(list_lmo_calls_cb, copy(LMO_calls_c))

            if !isempty(tree.node_queue)
                p_lb = tree.lb
                tree.lb = min(minimum([prio[2][1] for prio in tree.node_queue]), tree.incumbent)
                @assert p_lb <= tree.lb + tree.root.options[:dual_gap] "p_lb <= tree.lb + tree.root.options[:dual_gap] $(p_lb) <= $(tree.lb + tree.root.options[:dual_gap])"
            end
            # correct lower bound if necessary
            tree.lb = tree_lb(tree)
            dual_gap = tree.incumbent - tree_lb(tree)
            push!(list_lb_cb, tree_lb(tree))
            active_set_size = node.active_set_size
            discarded_set_size = node.discarded_set_size
            push!(list_active_set_size_cb, active_set_size)
            push!(list_discarded_set_size_cb, discarded_set_size)
            nodes_left = length(tree.nodes)
            if tree.root.updated_incumbent[]
                incumbent_updated = "*"
            else
                incumbent_updated = " "
            end
            if verbose && (
                mod(iteration, print_iter) == 0 ||
                iteration == 1 ||
                Bonobo.terminated(tree) ||
                tree.root.updated_incumbent[]
            )
                if (mod(iteration, print_iter * 40) == 0)
                    FrankWolfe.print_callback(headers, format_string, print_header=true)
                end
                FrankWolfe.print_callback(
                    (
                        incumbent_updated,
                        iteration,
                        nodes_left,
                        tree_lb(tree),
                        tree.incumbent,
                        dual_gap,
                        relative_gap(tree.incumbent, tree_lb(tree)),
                        time / 1000.0,
                        tree.num_nodes / time * 1000.0,
                        fw_time,
                        LMO_time,
                        tree.root.problem.tlmo.ncalls,
                        fw_iter,
                        active_set_size,
                        discarded_set_size,
                    ),
                    format_string,
                    print_header=false,
                )
                tree.root.updated_incumbent[] = false
            end
            # lmo calls per layer
            if length(list_lmo_calls_cb) > 1
                LMO_calls = list_lmo_calls_cb[end] - list_lmo_calls_cb[end-1]
            else
                LMO_calls = list_lmo_calls_cb[end]
            end
            if length(lmo_calls_per_layer) < node.std.depth
                push!(lmo_calls_per_layer, [LMO_calls])
                push!(active_set_size_per_layer, [active_set_size])
                push!(discarded_set_size_per_layer, [discarded_set_size])
            else
                push!(lmo_calls_per_layer[node.std.depth], LMO_calls)
                push!(active_set_size_per_layer[node.std.depth], active_set_size)
                push!(discarded_set_size_per_layer[node.std.depth], discarded_set_size)
            end

            # add tightenings
            push!(global_tightenings, node.global_tightenings)
            push!(local_tightenings, node.local_tightenings)
            push!(local_potential_tightenings, node.local_potential_tightenings)
            @assert node.local_potential_tightenings <= num_bin + num_int
            @assert node.local_tightenings <= num_bin + num_int
            @assert node.global_tightenings <= num_bin + num_int
        end
        # update current_node_id
        if !Bonobo.terminated(tree)
            tree.root.current_node_id[] =
                Bonobo.get_next_node(tree, tree.options.traverse_strategy).id
        end

        if Bonobo.terminated(tree)
            Bonobo.sort_solutions!(tree.solutions) 
            x = Bonobo.get_solution(tree)
            # x can be nothing if the user supplied a custom domain oracle and the time limit is reached
            if x === nothing
                @assert tree.root.problem.solving_stage == TIME_LIMIT_REACHED
            end
            primal_value = x !== nothing ? tree.root.problem.f(x) : Inf
            # deactivate postsolve if there is no solution
            tree.root.options[:use_postsolve] =
                x === nothing ? false : tree.root.options[:use_postsolve]

            # TODO: here we need to calculate the actual state

            # If the tree is empty, incumbent and solution should be the same!
            if !tree.root.options[:no_pruning] && isempty(tree.nodes)
                @assert isapprox(tree.incumbent, primal_value)
            end

            result[:number_nodes] = tree.num_nodes
            result[:lmo_calls] = tree.root.problem.tlmo.ncalls
            result[:heu_lmo_calls] = tree.root.options[:heu_ncalls]
            result[:list_num_nodes] = list_num_nodes_cb
            result[:list_lmo_calls_acc] = list_lmo_calls_cb
            result[:list_active_set_size] = list_active_set_size_cb
            result[:list_discarded_set_size] = list_discarded_set_size_cb
            result[:list_lb] = list_lb_cb
            result[:list_ub] = list_ub_cb
            result[:list_time] = list_time_cb
            result[:lmo_calls_per_layer] = lmo_calls_per_layer
            result[:active_set_size_per_layer] = active_set_size_per_layer
            result[:discarded_set_size_per_layer] = discarded_set_size_per_layer
            result[:node_level] = node_level
            result[:global_tightenings] = global_tightenings
            result[:local_tightenings] = local_tightenings
            result[:local_potential_tightenings] = local_potential_tightenings
            result[:tree_solutions] = tree.solutions

            if verbose
                FrankWolfe.print_callback(headers, format_string, print_footer=true)
                println()
            end
        end
    end
end
