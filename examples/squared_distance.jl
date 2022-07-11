using BranchAndBound
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
using PyPlot

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
integer_variable_bounds = BranchAndBound.IntegerBounds()
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
active_set = FrankWolfe.ActiveSet([(1.0, v)])
vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)
tlmo = BranchAndBound.TimeTrackingLMO(lmo)
m = BranchAndBound.SimpleOptimizationProblem(f, grad!, n, collect(1:n), tlmo, integer_variable_bounds)

nodeEx = BranchAndBound.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchAndBound.IntegerBounds(), 1, -1, 1e-3)
println(nodeEx.std)
# create tree
tree = Bonobo.initialize(; 
    traverse_strategy = Bonobo.BFS(),
    Node = typeof(nodeEx),
    root = (problem=m, current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => 1e-5, :verbose => true)),
)
Bonobo.set_root!(tree, 
    (active_set = active_set, 
    discarded_vertices = vertex_storage,
    local_bounds = BranchAndBound.IntegerBounds(),
    level = 1,
    sidx = -1,
    fw_dual_gap_limit = 1e-3)
)

list_lb, list_ub = Bonobo.optimize!(tree; min_number_lower=Inf)

# @show tree.root.problem.lmo.optimizing_times
# @show tree.root.problem.lmo.optimizing_nodes
# @show tree.root.problem.lmo.simplex_iterations

# plot(1:length(list_ub), list_ub, label="upper bound")
# plot(1:length(list_lb), list_lb, label="lower bound")
# title("Example : Squared Distance", loc="left")
# title("Dimension : 20", loc="right")
# legend()
# grid("on")
# ylabel("objective value")
# xlabel("number of iterations")
# # xticks([])
# # yticks([])
# # axis("off")
# # tick_params(bottom=false)
# # tick_params(left=false)
# savefig("test/dual_gap_linear.png")
