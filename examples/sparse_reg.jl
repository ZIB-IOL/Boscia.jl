using Statistics
using BranchWolfe
using FrankWolfe
using Random
using SCIP
import Bonobo
import MathOptInterface
const MOI = MathOptInterface
using PyPlot

# Constant parameters for the sparse regression
# min norm(y-A β)² + λ_0 ∑ z_i + λ_2 ||β||²
# s.t. -Mz_i<=β_i <= Mz_i
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p 
Random.seed!(42)
n0=15; p = 5*n0; k = ceil(n0/5);
const lambda_0 = rand(Float64); const lambda_2 = 10.0*rand(Float64);
const A = rand(Float64, n0, p)
const y = rand(Float64, n0)
const M = 2*var(A)

# "Sparse Regression" 
o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
MOI.empty!(o)
x = MOI.add_variables(o,2p)
for i in p+1:2p
    MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
    MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
    MOI.add_constraint(o, x[i], MOI.ZeroOne()) # or MOI.Integer()
end 
for i in 1:p
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,M], [x[i], x[i+p]]), 0.0), MOI.GreaterThan(0.0))
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-M], [x[i], x[i+p]]), 0.0), MOI.LessThan(0.0))
end
MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p),x[p+1:2p]), 0.0), MOI.LessThan(k))
lmo = FrankWolfe.MathOptLMO(o)
global_bounds = BranchWolfe.IntegerBounds()
for i = p+1:2p
    push!(global_bounds, (i, MOI.GreaterThan(0.0)))
    push!(global_bounds, (i, MOI.LessThan(1.0)))
end
time_lmo= BranchWolfe.TimeTrackingLMO(lmo)

# Define the root of the tree
# we fix the direction so we can actually find a veriable to split on later!
direction = Vector{Float64}(undef,2p)
Random.rand!(direction)
v = compute_extreme_point(time_lmo, direction)

function f(x)
    return sum((y-A*x[1:p]).^2) + lambda_0*sum(x[p+1:2p]) + lambda_2*FrankWolfe.norm(x[1:p])^2
end
function grad!(storage, x)
    storage.=vcat(2*(transpose(A)*A*x[1:p] - transpose(A)*y + lambda_2*x[1:p]), lambda_0*ones(p))
    return storage
end
active_set = FrankWolfe.ActiveSet([(1.0, v)]) 
vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

m = BranchWolfe.SimpleOptimizationProblem(f, grad!, 2p, collect(p+1:2p), time_lmo, global_bounds) 

# TO DO: how to do this elegantly
nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage,BranchWolfe.IntegerBounds(), 1, -1, 1e-3)

# create tree
tree = Bonobo.initialize(; 
traverse_strategy = Bonobo.BFS(),
Node = typeof(nodeEx),
root = (problem=m, current_node_id = current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => 1e-5, :verbose => true, :dual_gap => 1e-6))
)
Bonobo.set_root!(tree, 
    (active_set = active_set, 
    discarded_vertices = vertex_storage,
    local_bounds = BranchWolfe.IntegerBounds(),
    level = 1,
    sidx = -1,
    fw_dual_gap_limit= 1e-3)
)

list_lb, list_ub = Bonobo.optimize!(tree, min_number_lower=3)

# @show tree.root.problem.lmo.optimizing_times
# @show tree.root.problem.lmo.optimizing_nodes
# @show tree.root.problem.lmo.simplex_iterations

# plot(1:length(list_ub), list_ub, label="upper bound")
# plot(1:length(list_lb), list_lb, label="lower bound")
# title("Example : Sparse Regression", loc="left")
# title("Dimension : 15", loc="right")
# legend()
# grid("on")
# ylabel("objective value")
# xlabel("number of iterations")
# # xticks([])
# # yticks([])
# # axis("off")
# # tick_params(bottom=false)
# # tick_params(left=false)
# savefig("test/dual_gap_sparsereg.pdf")