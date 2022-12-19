using Boscia
using FrankWolfe
using Ipopt, JuMP
import MathOptInterface
const MOI = MathOptInterface
using Bonobo
const BB = Bonobo

""" Bonobo Tree structure for Ipopt """

# The Problem at the root 
mutable struct IpoptOptimizationProblem
    int_vars::Vector{Int64}
    m
    solving_stage::Boscia.Solve_Stage
    time_limit
    lbs::Vector{Float64}
    ubs::Vector{Float64}
end

# The node 
mutable struct MIPNode <: AbstractNode
    std::BnBNodeInfo
    lbs::Vector{Float64}
    ubs::Vector{Float64}
    status::MOI.TerminationStatus
end

# get the branching indices 
function BB.get_branching_indices(model::IpoptOptimizationProblem)
    return model.int_vars
end

# get relaxed values
function BB.get_relaxed_values(tree::BnBTree{MIPNode, IpoptOptimizationProblem}, node)
    vids = MOI.get(tree.root.m, MOI.ListOfVariableIndices())
    vars = VariableRef.(tree.root.m, vids)
    return JuMP.value.(vars)
end

# get branching node info 
function BB.get_branching_nodes_info(tree::BnBTree{MIPNode, IpoptOptimizationProblem}, node::MIPNode, vidx::Int)
    model = tree.root
    node_info = NamedTuple[]

    var = VariableRef(model.m, MOI.VariableIndex(vidx))

    # copy parent node bounds
    lbs = copy(node.lbs)
    ubs = copy(node.ubs)

    val = JuMP.value(var)

    # set left child
    ubs[vidx] = floor(Int, val)

    push!(node_info, (
        lbs = copy(node.lbs),
        ubs = ubs,
        status = MOI.OPTIMIZE_NOT_CALLED,
    ))

    # set right child
    lbs[vidx] = ceil(Int, val)

    push!(node_info, (
        lbs = lbs,
        ubs = copy(node.ubs),
        status = MOI.OPTIMIZE_NOT_CALLED,
    ))
    return node_info
end

# evaluate the problem at the node 
function BB.evaluate_node!(tree::BnBTree{MIPNode, IpoptOptimizationProblem}, node::MIPNode)
    build_node_model(tree, node)
    model = tree.root
    vids = MOI.get(model.m, MOI.ListOfVariableIndices())
    vars = VariableRef.(model.m, vids)

    optimize!(model.m)
    status = termination_status(model.m)
    node.status = status

    if status == MOI.LOCALLY_SOLVED || status == MOI.OPTIMAL
        obj_val = objective_value(model.m)
        if all(BB.is_approx_feasible.(tree, value.(vars[model.int_vars])))
            node.ub = obj_val 
            return obj_val, obj_val
        end
        return obj_val, NaN 
    end

    return NaN, NaN 
end

function build_node_model(tree::BnBTree{MIPNode, IpoptOptimizationProblem}, node::MIPNode)
    model = tree.root

    vids = MOI.get(model.m, MOI.ListOfVariableIndices())
    vars = VariableRef.(model.m, vids)
    for vidx in eachindex(vars)
        if vidx in model.int_vars
            if isfinite(node.lbs[vidx])
                JuMP.set_lower_bound(vars[vidx], node.lbs[vidx])
            elseif node.lbs[vidx] == -Inf 
                JuMP.set_lower_bound(vars[vidx], model.lbs[vidx])
            elseif node.lbs[vidx] == Inf
                error("Invalid lower bound for variable $vidx: $(node.lbs[vidx])")
            end

            if isfinite(node.ubs[vidx])
                JuMP.set_upper_bound(vars[vidx], node.ubs[vidx])
            elseif node.ubs[vidx] == Inf 
                JuMP.set_upper_bound(vars[vidx], model.ubs[vidx])
            elseif node.ubs[vidx] == -Inf
                error("Invalid upper bound for variable $vidx: $(node.ubs[vidx])")
            end
        end
    end
end

# Check if the Branch and Bound process can be stopped
function BB.terminated(tree::BnBTree{MIPNode, IpoptOptimizationProblem})
    if tree.root.solving_stage == Boscia.TIME_LIMIT_REACHED
        if isempty(tree.solutions)
            n = length(MOI.get(tree.root.m, MOI.ListOfVariableIndices()))
            node = MIPNode(BB.BnBNodeInfo(1, 0.0, 0.0), Vector(), Vector(), MOI.OPTIMAL)
            sol = BB.DefaultSolution(Inf, zeros(n), node)
            push!(tree.solutions, sol)
        end
        return true
    end
    absgap = tree.incumbent - tree.lb
    if absgap ≤ tree.options.abs_gap_limit
        return true
    end
    dual_gap = if signbit(tree.incumbent) != signbit(tree.lb)
        Inf
    elseif tree.incumbent == tree.lb 
        0.0
    else
        absgap / min(abs(tree.incumbent), abs(tree.lb))
    end
    return isempty(tree.nodes) || dual_gap ≤ tree.options.dual_gap_limit
end 

# BnB Callback
function build_callback(list_lb, list_ub, list_time, list_number_nodes)
    iteration = 0
    time_ref = Dates.now()
    return function callback(tree, node; kwargs...)
        time = float(Dates.value(Dates.now()-time_ref))
        iteration += 1
        push!(list_lb, tree.lb)
        push!(list_ub, tree.incumbent)
        push!(list_time, time)
        push!(list_number_nodes, tree.num_nodes)
        if mod(iteration, 100) == 0
            @show iteration, tree.num_nodes, tree.lb, tree.incumbent, time/1000
        end
        if tree.root.time_limit != Inf && time/1000.0 > tree.root.time_limit
            tree.root.solving_stage = Boscia.TIME_LIMIT_REACHED
        end
    end
end

# build tree 