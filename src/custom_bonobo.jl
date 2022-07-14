# Customized optimuze! and branch function
# Eventually we would like to return to the original Bonobo one and delete this.
# Changes in Bonobo are being made but are midterm changes.

function Bonobo.optimize!(tree::Bonobo.BnBTree; min_number_lower=20, percentage_dual_gap=0.7, callback=(args...; kwargs...)->(),)
    println("OWN OPTIMIZE FUNCTION USED")
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
        println("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        x = Bonobo.get_solution(tree)
        println("objective: ", tree.root.problem.f(x))
        println("number of nodes: $(tree.num_nodes)")
        println("number of lmo calls: ", tree.root.problem.lmo.ncalls)
        println("time in seconds: ", (Dates.value(Dates.now()-time_ref))/1000)
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
