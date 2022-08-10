using Boscia
using Test
using FrankWolfe
using Random
using SCIP
using Statistics
import Bonobo
import GLPK
import MathOptInterface

const a = 10
const diff_a = Random.rand(Bool,a)*0.6.+0.3
Random.seed!(1)

Random.seed!(1)

Random.seed!(1)

@testset "Norm over the hyperbox - infeasible" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, a)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
        MOI.add_constraint(o, xi, MOI.ZeroOne())
    end
    lmo = FrankWolfe.MathOptLMO(o)
    global_bounds = Boscia.IntegerBounds()
    for i = 1:a
        push!(global_bounds, (i, MOI.GreaterThan(0.0)))
        push!(global_bounds, (i, MOI.LessThan(1.0)))
    end
    time_lmo=Boscia.TimeTrackingLMO(lmo)

    # Define the root of the tree
    # we fix the direction so we can actually find a veriable to split on later!
    direction = Vector{Float64}(undef,a)
    Random.rand!(direction)
    v = compute_extreme_point(time_lmo, direction)

    function f(x)
        return sum(0.5*(x.-diff_a).^2)
    end
    function grad!(storage, x)
        @. storage = x-diff_a
    end

    active_set = FrankWolfe.ActiveSet([(1.0, v)])
    m = Boscia.SimpleOptimizationProblemInfeasible(f, grad!, a, collect(1:a), time_lmo, global_bounds, active_set)

    # TO DO: how to do this elegantly
    nodeEx = Boscia.InfeasibleFrankWolfeNode(Bonobo.BnBNodeInfo(1, 0.0,0.0), Bool[], Boscia.IntegerBounds())

    # create tree
    tree = Bonobo.initialize(; 
        traverse_strategy = Bonobo.BFS(),
        Node = typeof(nodeEx),
        root = (problem=m, current_node_id = current_node_id = Ref{Int}(0)),
    )
    Bonobo.set_root!(tree, 
        (valid_active = Bool[], 
        local_bounds= Boscia.IntegerBounds())
    )
    # Profile.init()
    # ProfileView.@profview Bonobo.optimize!(tree)
    @time Bonobo.optimize!(tree; min_number_lower=200)
    x = convert.(Int64,Bonobo.get_solution(tree))
    println("Number of tree nodes $(tree.num_nodes)")
    @test x == round.(diff_a)
end
