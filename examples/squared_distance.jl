using BranchWolfe
using FrankWolfe
using Random
using Debugger
using SCIP
import Bonobo
import GLPK
import MathOptInterface
const MOI = MathOptInterface
using Dates
using Printf

Random.seed!(1)

const n = 25#25
const diff = Random.rand(Bool,n)*0.6.+0.3

o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
MOI.empty!(o)
x = MOI.add_variables(o, n)
for xi in x
    MOI.add_constraint(o, xi, MOI.ZeroOne())
    MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
    MOI.add_constraint(o, xi, MOI.LessThan(1.0))
end
lmo = FrankWolfe.MathOptLMO(o)
integer_variable_bounds = BranchWolfe.IntegerBounds()
for i = 1:n
    push!(integer_variable_bounds, (i, MOI.GreaterThan(0.0)))
    push!(integer_variable_bounds, (i, MOI.LessThan(1.0)))
end

# Define the root of the tree
# we fix the direction so we can actually find a veriable to split on later!
direction = Vector{Float64}(undef,n)
Random.rand!(direction)
v = compute_extreme_point(lmo, direction)

function f(x)
    return sum(0.5*(x.-diff).^2)
end
function grad!(storage, x)
    @. storage = x-diff
end

# build tree
active_set = FrankWolfe.ActiveSet([(1.0, v)])
vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)
tlmo = BranchWolfe.TimeTrackingLMO(lmo)
m = BranchWolfe.SimpleOptimizationProblem(f, grad!, n, collect(1:n), tlmo, integer_variable_bounds)

nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, -1, 1e-3, Millisecond(0))

tree = Bonobo.initialize(; 
    traverse_strategy = Bonobo.BFS(),
    Node = typeof(nodeEx),
    root = (problem=m, current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => 1e-5, :verbose => true, :percentage_dual_gap => 0.7)),
)
Bonobo.set_root!(tree, 
    (active_set = active_set, 
    discarded_vertices = vertex_storage,
    local_bounds = BranchWolfe.IntegerBounds(),
    level = 1,
    sidx = -1,
    fw_dual_gap_limit = 1e-3,
    FW_time = Millisecond(0))
)

function build_bnb_callback(tree, list_lb_cb, list_ub_cb, list_time_cb, list_num_nodes_cb, list_lmo_calls_cb)
    time_ref = Dates.now()
    iteration = 0
    println("Starting BranchWolfe")
    verbose = get(tree.root.options, :verbose, -1)
    if verbose
        println("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        @printf("| iter \t| node id | lower bound | incumbent | gap \t| rel. gap | total time   | time/nodes \t| FW time    | LMO time   | total LMO calls | FW iterations | active set size | discarded set size |\n")
        println("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    end
    return function bnb_callback(tree, node; worse_than_incumbent=false, node_infeasible=false) # FW_iterations
        # update lower bound
        iteration = iteration +1
        append!(list_ub_cb, copy(tree.incumbent)) # new cb structure
        append!(list_lb_cb, copy(tree.lb))
        append!(list_num_nodes_cb, copy(tree.num_nodes))
        dual_gap = tree.incumbent-tree.lb
        time = float(Dates.value(Dates.now()-time_ref))
        append!(list_time_cb, time)
        FW_time = Dates.value(node.FW_time)
        FW_iter = FW_iterations[end]
        if !isempty(tree.root.problem.lmo.optimizing_times)
            LMO_time = sum(1000*tree.root.problem.lmo.optimizing_times)
            empty!(tree.root.problem.lmo.optimizing_times)
        end 
        LMO_calls = tree.root.problem.lmo.ncalls
        append!(list_lmo_calls_cb, copy(LMO_calls))

        if !isempty(tree.nodes)
            lower_bounds = [n[2].lb for n in tree.nodes]
            if tree.lb>minimum(lower_bounds)
            end
            tree.lb = minimum(lower_bounds)
        end

        if !isempty(FW_iterations)
            FW_iter = FW_iterations[end]
        else 
            FW_iter = 0
        end

        active_set_size = length(node.active_set)
        discarded_set_size = length(node.discarded_vertices.storage)

        if verbose & !worse_than_incumbent & !node_infeasible
            @printf("|   %4i|     %4i| \t% 06.5f|    %.5f|    %.5f|     %.3f|     %6i ms|      %4i ms|   %6i ms|   %6i ms|            %5i|          %5i|            %5i|               %5i|\n", iteration, node.id, tree.lb, tree.incumbent, dual_gap, dual_gap/tree.incumbent, time, round(time/tree.num_nodes), FW_time, LMO_time, tree.root.problem.lmo.ncalls, FW_iter, active_set_size, discarded_set_size)
        end
        #FW_iter = []

        # update current_node_id
        if !Bonobo.terminated(tree)
            tree.root.current_node_id[] = Bonobo.get_next_node(tree, tree.options.traverse_strategy).id
        end
        
        return 
    end
end

# build callbacks
list_ub_cb = []
list_lb_cb = []
list_time_cb = [] 
list_num_nodes_cb = [] 
list_lmo_calls_cb = []
bnb_callback = build_bnb_callback(tree, list_lb_cb, list_ub_cb, list_time_cb, list_num_nodes_cb, list_lmo_calls_cb)

FW_iterations = []
min_number_lower = Inf
fw_callback = BranchWolfe.build_FW_callback(tree, min_number_lower, true, FW_iterations)

tree.root.options[:callback] = fw_callback
tree.root.current_node_id[] = Bonobo.get_next_node(tree, tree.options.traverse_strategy).id

time_ref = Dates.now()
Bonobo.optimize!(tree; callback=bnb_callback) # min_number_lower, bnb_callback)
#list_lb, list_ub = Bonobo.optimize!(tree; min_number_lower=Inf, callback=callback)

println("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
x = Bonobo.get_solution(tree)
println("objective: ", tree.root.problem.f(x))
println("number of nodes: $(tree.num_nodes)")
println("number of lmo calls: ", tree.root.problem.lmo.ncalls)
println("time in seconds: ", (Dates.value(Dates.now()-time_ref))/1000)
# lb>ub????