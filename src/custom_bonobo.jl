# Customized optimize! and branch function
# Eventually we would like to return to the original Bonobo one and delete this.
# Changes in Bonobo are being made but are midterm changes.

function Bonobo.optimize!(tree::Bonobo.BnBTree; min_number_lower=20, percentage_dual_gap=0.7, callback=(args...; kwargs...)->(),)
    time_ref = Dates.now()
    list_lb = [] 
    list_ub = []
    FW_iterations = []
    fw_callback = build_FW_callback(tree, min_number_lower, true, FW_iterations)
    callback = build_bnb_callback(tree)
    while !Bonobo.terminated(tree)
        node = Bonobo.get_next_node(tree, tree.options.traverse_strategy)
        tree.root.current_node_id[] = node.id
        lb, ub, FW_time, LMO_time = Bonobo.evaluate_node!(tree, node, fw_callback) 
        # if the problem was infeasible we simply close the node and continue
        if isnan(lb) && isnan(ub)
            Bonobo.close_node!(tree, node)
            list_lb, list_ub = callback(tree, node; FW_time=FW_time, LMO_time=LMO_time, FW_iterations=FW_iterations, node_infeasible=true)
            continue
        end
        Bonobo.set_node_bound!(tree.sense, node, lb, ub)
        # if the evaluated lower bound is worse than the best incumbent -> close and continue
        if node.lb >= tree.incumbent
            Bonobo.close_node!(tree, node)
            list_lb, list_ub = callback(tree, node; FW_time=FW_time, LMO_time=LMO_time, FW_iterations=FW_iterations, worse_than_incumbent=true)
            continue
        end
        updated = Bonobo.update_best_solution!(tree, node)
        updated && Bonobo.bound!(tree, node.id)

        Bonobo.close_node!(tree, node)
        #println("branch node")
        Bonobo.branch!(tree, node; percentage_dual_gap=percentage_dual_gap)
        list_lb, list_ub = callback(tree, node; FW_time=FW_time, LMO_time=LMO_time, FW_iterations=FW_iterations,)
    end
     if get(tree.root.options, :verbose, -1)
        print_callback = FrankWolfe.print_callback
        headers = ["Iteration", "Open", "Bound", "Incumbent", "Gap (abs)", "Gap (%)", "Time (s)", "Nodes/Sec", "FW (ms)", "LMO (ms)", "LMO (calls)", "FW (iters)", "Active Set", "Discarded"]   
        format_string = "%10i %10i %14e %14e %14e %14e %14e %14e %14i %14i %14i %10i %10i %10i\n"
        print_callback(headers, format_string, print_footer=true)
        println()

        x = Bonobo.get_solution(tree)
        println("Solution Statistics.")
        primal_value = tree.root.problem.f(x)

        # TODO: here we need to calculate the actual state

        status_string = "FIX ME" # should report "feasible", "optimal", "infeasible", "gap tolerance met"
        if isempty(tree.nodes)
            status_string = "Optimal (tree empty)"
        else
            status_string = "Optimal (tolerance reached)"
        end

        println("\t Solution Status: ", status_string)
        println("\t Primal Objective: ", primal_value)
        println("\t Dual Bound (absolute): ", tree.lb)
        println("\t Dual Bound (relative in %): $(relative_gap(primal_value,tree.lb) * 100.0)\n")
        println("Search Statistics.")
        println("\t Total number of nodes processed: ", tree.num_nodes)
        println("\t Total number of lmo calls: ", tree.root.problem.lmo.ncalls)
        total_time_in_sec = (Dates.value(Dates.now()-time_ref))/1000.0
        println("\t Total time (s): ", total_time_in_sec)
        println("\t LMO calls / sec: ", tree.root.problem.lmo.ncalls / total_time_in_sec)        
        println("\t Nodes / sec: ", tree.num_nodes / total_time_in_sec)
        println("\t LMO calls / node: $(tree.root.problem.lmo.ncalls / tree.num_nodes)\n")

        append!(list_ub, copy(tree.incumbent))
        append!(list_lb, copy(tree.lb))
    end
    return list_lb, list_ub
end

function Bonobo.branch!(tree, node; percentage_dual_gap)
    variable_idx = Bonobo.get_branching_variable(tree, tree.options.branch_strategy, node)
    # no branching variable selected => return
    variable_idx == -1 && return 
    nodes_info = Bonobo.get_branching_nodes_info(tree, node, variable_idx; percentage_dual_gap=percentage_dual_gap)
    for node_info in nodes_info
        Bonobo.add_node!(tree, node, node_info)
    end
end
