using Statistics
using Random
using Distributions
using LinearAlgebra
import HiGHS
using SCIP
using MathOptInterface
MOI = MathOptInterface
using FrankWolfe
using BranchWolfe
using Bonobo
using Dates
using Printf

# Sparse Poisson regression
# min_{w, b, z} ∑_i exp(w x_i + b) - y_i (w x_i + b) + α norm(w)^2
# s.t. -N z_i <= w_i <= N z_i
# b ∈ [-N, N]
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p

Random.seed!(4)

n0=30
p = n0

# underlying true weights
const w0 = 2 * rand(Float64, p) .- 1 
# set 50 entries to 0
for _ in 1:20
    w0[rand(1:p)] = 0
end
const b0 = 2 * rand(Float64) - 1
const X0 = 2 * rand(Float64, n0, p) .- 1 
const y0 = map(1:n0) do idx
    a = dot(X0[idx,:], w0) + b0
    rand(Distributions.Poisson(exp(a)))
end
N = 5.0

# "Poisson sparse regression" 
k = 5
o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
w = MOI.add_variables(o, p)
z = MOI.add_variables(o, p)
b = MOI.add_variable(o)
for i in 1:p
    MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
    MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
    MOI.add_constraint(o, z[i], MOI.ZeroOne())
end
for i in 1:p
    MOI.add_constraint(o, -N * z[i]- w[i], MOI.LessThan(0.0))
    MOI.add_constraint(o, N * z[i]- w[i], MOI.GreaterThan(0.0))
    # Indicator: z[i] = 1 => -N <= w[i] <= N
    gl = MOI.VectorAffineFunction(
        [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
            MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),],
        [0.0, 0.0], )
    gg = MOI.VectorAffineFunction(
        [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
            MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),],
        [0.0, 0.0], )
    MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(N)))
    MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-N))) 
end
MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
MOI.add_constraint(o, b, MOI.LessThan(N))
MOI.add_constraint(o, b, MOI.GreaterThan(-N))
lmo = FrankWolfe.MathOptLMO(o)
global_bounds = BranchWolfe.IntegerBounds()
for i in 1:p
    push!(global_bounds, (p+i, MOI.GreaterThan(0.0)))
    push!(global_bounds, (p+i, MOI.LessThan(1.0)))
    push!(global_bounds, (i, MOI.GreaterThan(-N)))
    push!(global_bounds, (i, MOI.LessThan(N)))
end
push!(global_bounds, (2p+1, MOI.GreaterThan(-N)))
push!(global_bounds, (2p+1, MOI.LessThan(N)))

direction = Vector{Float64}(undef,2p + 1)
Random.rand!(direction)
v = compute_extreme_point(lmo, direction)
vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)
α = 1.3
function f(θ)
    w = @view(θ[1:p])
    b = θ[end]
    s = sum(1:n0) do i
        a = dot(w, X0[:,i]) + b
        1/n0 * (exp(a) - y0[i] * a)
    end
    s + α * norm(w)^2
end
function grad!(storage, θ)
    w = @view(θ[1:p])
    b = θ[end]
    storage[1:p] .= 2α .* w
    storage[p+1:2p] .= 0
    storage[end] = 0
    for i in 1:n0
        xi = @view(X0[:,i])
        a = dot(w, xi) + b
        storage[1:p] .+= 1/n0 * xi * exp(a)
        storage[1:p] .-= 1/n0 * y0[i] * xi
        storage[end] += 1/n0 * (exp(a) - y0[i])
    end
    return storage
end
time_lmo = BranchWolfe.TimeTrackingLMO(lmo)
active_set = FrankWolfe.ActiveSet([(1.0, v)]) 
m = BranchWolfe.SimpleOptimizationProblem(f, grad!, 2p+1, collect(p+1:2p), time_lmo, global_bounds) 
nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, -1, 1e-3, Millisecond(0))

# create tree
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
        FW_iter = 0 #FW_iterations[end]
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

println("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
x = Bonobo.get_solution(tree)
println("objective: ", tree.root.problem.f(x))
println("number of nodes: $(tree.num_nodes)")
println("number of lmo calls: ", tree.root.problem.lmo.ncalls)
println("time in seconds: ", (Dates.value(Dates.now()-time_ref))/1000)

# "Hybrid branching poisson sparse regression"
k = 5
o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
w = MOI.add_variables(o, p)
z = MOI.add_variables(o, p)
b = MOI.add_variable(o)
for i in 1:p
    MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
    MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
    MOI.add_constraint(o, z[i], MOI.ZeroOne())
end
for i in 1:p
    MOI.add_constraint(o, -N * z[i]- w[i], MOI.LessThan(0.0))
    MOI.add_constraint(o, N * z[i]- w[i], MOI.GreaterThan(0.0))
    # Indicator: z[i] = 1 => -N <= w[i] <= N
    gl = MOI.VectorAffineFunction(
        [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
            MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),],
        [0.0, 0.0], )
    gg = MOI.VectorAffineFunction(
        [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
            MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),],
        [0.0, 0.0], )
    MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(N)))
    MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-N))) 
end
MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
MOI.add_constraint(o, b, MOI.LessThan(N))
MOI.add_constraint(o, b, MOI.GreaterThan(-N))
lmo = FrankWolfe.MathOptLMO(o)
global_bounds = BranchWolfe.IntegerBounds()
for i in 1:p
    push!(global_bounds, (p+i, MOI.GreaterThan(0.0)))
    push!(global_bounds, (p+i, MOI.LessThan(1.0)))
    push!(global_bounds, (i, MOI.GreaterThan(-N)))
    push!(global_bounds, (i, MOI.LessThan(N)))
end
push!(global_bounds, (2p+1, MOI.GreaterThan(-N)))
push!(global_bounds, (2p+1, MOI.LessThan(N)))

direction = Vector{Float64}(undef,2p + 1)
Random.rand!(direction)
v = compute_extreme_point(lmo, direction)
vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)
α = 1.3
function f(θ)
    w = @view(θ[1:p])
    b = θ[end]
    s = sum(1:n0) do i
        a = dot(w, X0[:,i]) + b
        1/n0 * (exp(a) - y0[i] * a)
    end
    s + α * norm(w)^2
end
function grad!(storage, θ)
    w = @view(θ[1:p])
    b = θ[end]
    storage[1:p] .= 2α .* w
    storage[p+1:2p] .= 0
    storage[end] = 0
    for i in 1:n0
        xi = @view(X0[:,i])
        a = dot(w, xi) + b
        storage[1:p] .+= 1/n0 * xi * exp(a)
        storage[1:p] .-= 1/n0 * y0[i] * xi
        storage[end] += 1/n0 * (exp(a) - y0[i])
    end
    return storage
end
time_lmo = BranchWolfe.TimeTrackingLMO(lmo)
active_set = FrankWolfe.ActiveSet([(1.0, v)]) 
m = BranchWolfe.SimpleOptimizationProblem(f, grad!, 2p+1, collect(p+1:2p), time_lmo, global_bounds) 
nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, -1, 1e-3, Millisecond(0))

# create tree
function perform_strong_branch(tree, node)
    return node.level <= length(tree.root.problem.integer_variables)/3
end
branching_strategy = BranchWolfe.HybridStrongBranching(10, 1e-3, HiGHS.Optimizer(), perform_strong_branch)
MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)
tree = Bonobo.initialize(;
    traverse_strategy = Bonobo.BFS(),
    Node = typeof(nodeEx),
    root = (problem=m, current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => 1e-5, :verbose => true, :percentage_dual_gap => 0.7)),
    branch_strategy = branching_strategy, #() ->
)
Bonobo.set_root!(tree, 
(active_set = active_set, 
discarded_vertices = vertex_storage,
local_bounds = BranchWolfe.IntegerBounds(),
level = 1, 
sidx = -1,
fw_dual_gap_limit= 1e-3,
FW_time = Millisecond(0))
)


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


n0g=20
pg = n0g

# underlying true weights
const w0g = 2 * rand(Float64, pg) .- 1
# set 50 entries to 0
for _ in 1:15
w0g[rand(1:pg)] = 0
end
const b0g = 2 * rand(Float64) - 1
const X0g = 2 * rand(Float64, n0g, pg) .- 1 
const y0g = map(1:n0g) do idx
a = dot(X0g[idx,:], w0g) + b0g
rand(Distributions.Poisson(exp(a)))
end
Ng = 5.0

k = 10
group_size = convert(Int64, floor(pg/k))
groups= []
for i in 1:(k-1)
push!(groups, ((i-1)*group_size+1):(i*group_size))
end
push!(groups,((k-1)*group_size+1):pg)


# "Sparse Group Poisson"
o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
w = MOI.add_variables(o, pg)
z = MOI.add_variables(o, pg)
b = MOI.add_variable(o)
for i in 1:pg
    MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
    MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
    MOI.add_constraint(o, z[i], MOI.ZeroOne())
end
for i in 1:pg
    MOI.add_constraint(o, -Ng * z[i]- w[i], MOI.LessThan(0.0))
    MOI.add_constraint(o, Ng * z[i]- w[i], MOI.GreaterThan(0.0))
    # Indicator: z[i] = 1 => -Ng <= w[i] <= Ng
    gl = MOI.VectorAffineFunction(
        [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
            MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),],
        [0.0, 0.0], )
    gg = MOI.VectorAffineFunction(
        [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
            MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),],
        [0.0, 0.0], )
    MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(Ng)))
    MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-Ng)))
end
MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
MOI.add_constraint(o, b, MOI.LessThan(Ng))
MOI.add_constraint(o, b, MOI.GreaterThan(-Ng))
for i in 1:k
    #MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(group_size),x[groups[i]]), 0.0), MOI.GreaterThan(1.0))
    MOI.add_constraint(o, sum(z[groups[i]], init=0.0), MOI.GreaterThan(1.0))
end
lmo = FrankWolfe.MathOptLMO(o)
global_bounds = BranchWolfe.IntegerBounds()
for i in 1:pg
    push!(global_bounds, (pg+i, MOI.GreaterThan(0.0)))
    push!(global_bounds, (pg+i, MOI.LessThan(1.0)))
    push!(global_bounds, (i, MOI.GreaterThan(-Ng)))
    push!(global_bounds, (i, MOI.LessThan(Ng)))
end
push!(global_bounds, (2pg+1, MOI.GreaterThan(-Ng)))
push!(global_bounds, (2pg+1, MOI.LessThan(Ng)))

direction = Vector{Float64}(undef,2pg + 1)
Random.rand!(direction)
v = compute_extreme_point(lmo, direction)
vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)
α = 1.3
function f(θ)
    w = @view(θ[1:pg])
    b = θ[end]
    s = sum(1:n0g) do i
        a = dot(w, X0g[:,i]) + b
        1/n0g * (exp(a) - y0g[i] * a)
    end
    s + α * norm(w)^2
end
function grad!(storage, θ)
    w = @view(θ[1:pg])
    b = θ[end]
    storage[1:pg] .= 2α .* w
    storage[pg+1:2pg] .= 0
    storage[end] = 0
    for i in 1:n0g
        xi = @view(X0g[:,i])
        a = dot(w, xi) + b
        storage[1:pg] .+= 1/n0g * xi * exp(a)
        storage[1:pg] .-= 1/n0g * y0g[i] * xi
        storage[end] += 1/n0g * (exp(a) - y0g[i])
    end
    return storage
end
time_lmo = BranchWolfe.TimeTrackingLMO(lmo)
active_set = FrankWolfe.ActiveSet([(1.0, v)]) 
m = BranchWolfe.SimpleOptimizationProblem(f, grad!, 2pg+1, collect(pg+1:2pg), time_lmo, global_bounds) 
nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, -1, 1e-3, Millisecond(0))

# create tree
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
fw_dual_gap_limit= 1e-3,
FW_time = Millisecond(0)))

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


# "Strong branching sparse group poisson"
o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
w = MOI.add_variables(o, pg)
z = MOI.add_variables(o, pg)
b = MOI.add_variable(o)
for i in 1:pg
    MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
    MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
    MOI.add_constraint(o, z[i], MOI.ZeroOne())
end
for i in 1:pg
    MOI.add_constraint(o, -Ng * z[i]- w[i], MOI.LessThan(0.0))
    MOI.add_constraint(o, Ng * z[i]- w[i], MOI.GreaterThan(0.0))
    # Indicator: z[i] = 1 => -Ng <= w[i] <= Ng
    gl = MOI.VectorAffineFunction(
        [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
            MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),],
        [0.0, 0.0], )
    gg = MOI.VectorAffineFunction(
        [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
            MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),],
        [0.0, 0.0], )
    MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(Ng)))
    MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-Ng)))
end
MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
MOI.add_constraint(o, b, MOI.LessThan(Ng))
MOI.add_constraint(o, b, MOI.GreaterThan(-Ng))
for i in 1:k
    #MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(group_size),x[groups[i]]), 0.0), MOI.GreaterThan(1.0))
    MOI.add_constraint(o, sum(z[groups[i]], init=0.0), MOI.GreaterThan(1.0))
end
lmo = FrankWolfe.MathOptLMO(o)
global_bounds = BranchWolfe.IntegerBounds()
for i in 1:pg
    push!(global_bounds, (pg+i, MOI.GreaterThan(0.0)))
    push!(global_bounds, (pg+i, MOI.LessThan(1.0)))
    push!(global_bounds, (i, MOI.GreaterThan(-Ng)))
    push!(global_bounds, (i, MOI.LessThan(Ng)))
end
push!(global_bounds, (2pg+1, MOI.GreaterThan(-Ng)))
push!(global_bounds, (2pg+1, MOI.LessThan(Ng)))

direction = Vector{Float64}(undef,2pg + 1)
Random.rand!(direction)
v = compute_extreme_point(lmo, direction)
vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)
α = 1.3
function f(θ)
    w = @view(θ[1:pg])
    b = θ[end]
    s = sum(1:n0g) do i
        a = dot(w, X0g[:,i]) + b
        1/n0g * (exp(a) - y0g[i] * a)
    end
    s + α * norm(w)^2
end
function grad!(storage, θ)
    w = @view(θ[1:pg])
    b = θ[end]
    storage[1:pg] .= 2α .* w
    storage[pg+1:2pg] .= 0
    storage[end] = 0
    for i in 1:n0g
        xi = @view(X0g[:,i])
        a = dot(w, xi) + b
        storage[1:pg] .+= 1/n0g * xi * exp(a)
        storage[1:pg] .-= 1/n0g * y0g[i] * xi
        storage[end] += 1/n0g * (exp(a) - y0g[i])
    end
    return storage
end
time_lmo = BranchWolfe.TimeTrackingLMO(lmo)
active_set = FrankWolfe.ActiveSet([(1.0, v)]) 
m = BranchWolfe.SimpleOptimizationProblem(f, grad!, 2pg+1, collect(pg+1:2pg), time_lmo, global_bounds) 
nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, -1, 1e-3, Millisecond(0))

# create tree
function perform_strong_branch(tree, node)
    return node.level <= length(tree.root.problem.integer_variables)/3
end
branching_strategy = BranchWolfe.HybridStrongBranching(10, 1e-3, HiGHS.Optimizer(), perform_strong_branch)
MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)
tree = Bonobo.initialize(;
    traverse_strategy = Bonobo.BFS(),
    Node = typeof(nodeEx),
    root = (problem=m, current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => 1e-5, :verbose => true, :percentage_dual_gap => 0.7)),
    branch_strategy = branching_strategy, #() ->
)
Bonobo.set_root!(tree, 
(active_set = active_set, 
discarded_vertices = vertex_storage,
local_bounds = BranchWolfe.IntegerBounds(),
level = 1, 
sidx = -1,
fw_dual_gap_limit= 1e-3,
FW_time = Millisecond(0)))

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

x = Bonobo.get_solution(tree)
