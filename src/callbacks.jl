# FW callback
function build_FW_callback(tree, min_number_lower, check_rounding_value::Bool, FW_iterations)
    return function fw_callback(state, active_set)
        push!(FW_iterations, copy(state.t))

        vars = [MOI.VariableIndex(var) for var in 1:tree.root.problem.nvars]
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

        if !isempty(tree.nodes)
            if count(n.lb < val for n in values(tree.nodes)) > min_number_lower
                return false
            end
        end

        if check_rounding_value
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

"""
    Output of BranchWolfe

        iter :          current iteration of BranchWolfe
        node id :       current node id
        lower bound :   tree.lb
        incumbent :     tree.incumbent
        gap :           tree.incumbent-tree.lb
        rel. gap :      dual_gap/tree.incumbent
        time :          total time of BranchWolfe
        time/nodes :    average time per node
        FW time :       time spent in FW 
        LMO time :      time used by LMO
        LMO calls :     number of compute_extreme_point calls in FW
        FW iterations : number of iterations in FW
"""
function build_bnb_callback(tree)
    time_ref = Dates.now()
    list_ub = []
    list_lb = []
    iteration = 0
    println("Starting BranchWolfe")
    verbose = get(tree.root.options, :verbose, -1)

    headers = ["Iteration", "Node id", "Left", "Bound", "Incumbent", "Gap (abs)", "Gap (%)", "Total Time", "Nodes/Sec", "FW (ms)", "LMO (ms)", "LMO (calls)", "FW (iters)", "Active Set", "Discarded"]   
    format_string = "%10i %10i %10i %14e %14e %14e %14e %14e %14e %14i %14i %14i %10i %10i %10i\n"
    print_callback = FrankWolfe.print_callback

    if verbose
        print_callback(headers, format_string, print_header=true)
    end
    return function callback(tree, node; FW_time=NaN, LMO_time=NaN, FW_iterations=FW_iterations, worse_than_incumbent=false, node_infeasible=false)
        if node_infeasible==false
            # update lower bound
            append!(list_ub, copy(tree.incumbent))
            append!(list_lb, copy(tree.lb))
            iteration = iteration + 1

            if !isempty(tree.nodes)
                ids = [n[2].id for n in tree.nodes]
                lower_bounds = [n[2].lb for n in tree.nodes]
                if tree.lb>minimum(lower_bounds)
                end
                tree.lb = minimum(lower_bounds)
            end

            dual_gap = tree.incumbent-tree.lb
            time = Dates.value(Dates.now()-time_ref)
            FW_time = Dates.value(FW_time)

            if !isempty(FW_iterations)
                FW_iter = FW_iterations[end]
            else 
                FW_iter = 0
            end

            active_set_size = length(node.active_set)
            discarded_set_size = length(node.discarded_vertices.storage)
            nodes_left= length(tree.nodes)
            
            print_iter = 100 # this should not be hard-coded but a parameter the user can pass as well
            
            if verbose && (mod(iteration, print_iter) == 0 || iteration == 1 || isempty(tree.nodes)) # TODO: need to output the very last iteration also if we skip some inbetween
                if (mod(iteration, print_iter*40) == 0)
                    print_callback(headers, format_string, print_header=true)
                end
                print_callback((iteration, node.id, nodes_left, tree.lb, tree.incumbent, dual_gap, abs(dual_gap)/abs(tree.incumbent) * 100.0, time, tree.num_nodes/time * 1000.0, FW_time, LMO_time, tree.root.problem.lmo.ncalls, FW_iter, active_set_size, discarded_set_size), format_string, print_header=false)
            end
            FW_iter = []
            return list_lb, list_ub
        end
        append!(list_ub, copy(tree.incumbent))
        append!(list_lb, copy(tree.lb))
        iteration = iteration + 1

        if !isempty(tree.nodes)
            ids = [n[2].id for n in tree.nodes]
            lower_bounds = [n[2].lb for n in tree.nodes]
            if tree.lb>minimum(lower_bounds)
            end
            tree.lb = minimum(lower_bounds)
        end

        dual_gap = tree.incumbent-tree.lb
        time = Dates.value(Dates.now()-time_ref)
        if verbose
            print_callback((iteration, node.id, tree.lb, tree.incumbent, dual_gap, abs(dual_gap)/abs(tree.incumbent) * 100.0, time, tree.num_nodes/time * 1000.0, 0, 0, 0, 0), format_string, print_header=false)
        end
        return list_lb, list_ub
    end
end