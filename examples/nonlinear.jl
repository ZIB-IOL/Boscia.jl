using SCIP
using FrankWolfe
using LinearAlgebra
import MathOptInterface
using Random
using BranchWolfe
import Bonobo
using Printf
using Dates
using PyPlot

const MOI = MathOptInterface

n=5
seed=10

Random.seed!(seed)

const o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
MOI.empty!(o)
x = MOI.add_variables(o, n)
for xi in x
    MOI.add_constraint(o, xi, MOI.ZeroOne())
    MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
    MOI.add_constraint(o, xi, MOI.LessThan(1.0))
end

# print(o.variable_info)

const lmo = FrankWolfe.MathOptLMO(o)

#const A = LinearAlgebra.Symmetric(randn(n,n), :U)
const A = let
    A = randn(n,n)
    A'*A
end

@assert isposdef(A) == true
# add value on the diagonal 

const y = Random.rand(Bool,n)*0.6.+0.3

function f(x)
    d = x-y
    return dot(d, A, d)
end

function grad!(storage, x)
    # storage = Ax
    mul!(storage, A, x)
    # storage = 2Ax - 2Ay
    mul!(storage, A, y, -2, 2)
end

integer_variable_bounds = BranchWolfe.IntegerBounds()
for i = 1:n
    push!(integer_variable_bounds, (i, MOI.GreaterThan(0.0)))
    push!(integer_variable_bounds, (i, MOI.LessThan(1.0)))
end

const direction = randn(n)
v = compute_extreme_point(lmo, direction)

active_set = FrankWolfe.ActiveSet([(1.0, v)]) 
vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)
const tlmo = BranchWolfe.TimeTrackingLMO(lmo)

m = BranchWolfe.SimpleOptimizationProblem(f, grad!, n, collect(1:n), tlmo, integer_variable_bounds) 

nodeEx = BranchWolfe.FrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), active_set, vertex_storage, BranchWolfe.IntegerBounds(), 1, -1, 1e-3)

# create tree
tree = Bonobo.initialize(; 
    traverse_strategy = Bonobo.BFS(),
    Node = typeof(nodeEx),
    root = (problem=m, current_node_id = current_node_id = Ref{Int}(0), options= Dict{Symbol, Any}(:FW_tol => 1e-5, (:verbose => true))),
)
Bonobo.set_root!(tree, 
    (active_set = active_set, 
    discarded_vertices = vertex_storage,
    local_bounds = BranchWolfe.IntegerBounds(),
    level = 1,
    sidx = -1,
    fw_dual_gap_limit= 1e-3)
)


list_lb, list_ub = Bonobo.optimize!(tree; min_number_lower=3, epsilon=0.7)
# @show tree.root.problem.lmo.optimizing_times
# @show tree.root.problem.lmo.optimizing_nodes
# @show tree.root.problem.lmo.simplex_iterations

# plot(1:length(list_ub), list_ub, label="upper bound")
# plot(1:length(list_lb), list_lb, label="lower bound")
# title("Example : (x-y)' A (x-y)", loc="left")
# title("Dimension : 15", loc="right")
# legend()
# grid("on")
# ylabel("objective value")
# # ylabel("objective value")
# xlabel("number of iterations")
# savefig("test/dual_gap_nonlinear.pdf")
