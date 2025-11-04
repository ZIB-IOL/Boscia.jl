# Network Design Problem Example
#
# This example demonstrates solving a network design problem using Boscia.jl with two approaches:
# 1. MOI-based LMO: Using MathOptInterface to model the feasible region
# 2. Custom LMO: Using a customized Linear Minimization Oracle based on shortest path algorithms
#
# ## Problem Description
# We solve a transportation network design problem where:
# - Some edges have been removed from the network
# - We decide which edges to restore (binary decision y[e])
# - Traffic flows are routed to minimize total travel time (with BPR congestion)
# - Flow conservation constraints must be satisfied
# - Linking constraints: x[e] <= M * y[e] (flow on removed edges only if restored)
#
# ## Key Difference Between Approaches
#
# **IMPORTANT**: The two approaches solve DIFFERENT formulations of the same problem!
#
# ### MOI-based LMO
# - Enforces linking constraints x[e] <= M*y[e] as HARD constraints in the MOI model
# - Objective: minimize BPR_cost(x) + restoration_cost(y)
# - Finds feasible solution satisfying all constraints exactly
#
# ### Custom LMO  
# - Cannot encode linking constraints x[e] <= M*y[e] in the shortest-path oracle
# - Instead, adds PENALTY TERMS to the objective function
# - Objective: minimize BPR_cost(x) + restoration_cost(y) + ρ·∑max(0, x[e] - M*y[e])²
# - Uses penalty method to discourage constraint violations
#
# Because they optimize different objective functions, they may find different solutions!
# The penalty weight ρ controls the tradeoff between optimality and feasibility.
using Boscia
using FrankWolfe
using Graphs
using SparseArrays
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
using HiGHS 

println("\nDocumentation Example 01: Network Design Problem")

# ## Data Structure
# Simple graph structure with edge weights and demands
mutable struct NetworkData
    num_nodes::Int
    num_edges::Int  
    init_nodes::Vector{Int}
    term_nodes::Vector{Int}
    free_flow_time::Vector{Float64}
    capacity::Vector{Float64}
    b::Vector{Float64}  # BPR function parameter
    power::Vector{Float64}  # BPR function exponent
    travel_demand::Matrix{Float64}
    num_zones::Int
end

# ## Load Network
# Transportation network matching the purchasable edge diagram
# 2 sources (S1, S2) → 5 intermediate nodes → 1 destination (D)
# Total: 8 nodes
#
# IMPORTANT: Node numbering is constrained by the code structure
# The code requires zones 1..num_zones to BE nodes 1..num_zones
# So: Node 1=S1 (zone 1), Node 2=S2 (zone 2), Node 3=D (zone 3)
# Nodes 4-8 are the 5 intermediate nodes from the diagram
#
# Mapping from diagram to code:
# Diagram -> Code numbering
# S1 -> 1, S2 -> 2, D -> 3
# intermediate nodes 1,2,3,4,5 -> 4,5,6,7,8
#
# Network topology from your description:
# S1(1) → node_1(4)
# S2(2) → node_3(6)  
# node_1(4) → node_3(6)
# node_2(5) → node_1(4), D(3)
# node_3(6) → node_1(4), node_4(7)
# node_4(7) → node_3(6), node_5(8)
# node_5(8) → node_4(7), D(3)
# Optional edge: node_1(4) → node_2(5) [the purchasable dashed edge]

function load_braess_network()
    init_nodes = [1, 2, 4, 5, 5, 6, 6, 7, 7, 8, 8, 4]
    term_nodes = [4, 6, 6, 4, 3, 4, 7, 6, 8, 7, 3, 5]
    free_flow_time = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    capacity = [10.0, 10.0, 10.0, 10.0, 1.5, 10.0, 10.0, 10.0, 10.0, 10.0, 1.5, 10.0]
    b = [0.1, 0.1, 0.1, 0.1, 3.0, 0.1, 0.1, 0.1, 0.1, 0.1, 3.0, 0.1]
    power = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    travel_demand = [0.0 0.0 1.0; 0.0 0.0 1.0; 0.0 0.0 0.0]
    return NetworkData(8, length(init_nodes), init_nodes, term_nodes, free_flow_time,
                      capacity, b, power, travel_demand, 3)
end

# ## Shortest Path Dijkstra Implementation  

# Custom Dijkstra implementation for traffic assignment
function traffic_dijkstra(graph, travel_time, origin, link_dic)
    state = Graphs.dijkstra_shortest_paths(graph, origin)
    return state
end

# Add demand to flow vector following shortest path
function add_demand_to_path!(x, demand, state, origin, destination, link_dic, edge_list, num_zones)
    current = destination
    parent = -1
    edge_count = length(edge_list)
    agg_start = edge_count * num_zones
    
    while parent != origin && origin != destination && current != 0
        parent = state.parents[current]
        if parent != 0
            link_idx = link_dic[parent, current]
            if link_idx != 0
                x[(destination - 1) * edge_count + link_idx] += demand
                x[agg_start + link_idx] += demand
            end
        end
        current = parent
    end
end

# All-or-nothing assignment: route all flow on shortest paths
function all_or_nothing_assignment(travel_time_vector, net_data, graph, link_dic, edge_list)
    num_zones = net_data.num_zones
    edge_count = net_data.num_edges
    travel_time = travel_time_vector[num_zones * edge_count + 1 : (num_zones + 1) * edge_count]
    x = zeros(length(travel_time_vector))
    
    for origin in 1:num_zones
        state = Graphs.dijkstra_shortest_paths(graph, origin)
        
        for destination in 1:num_zones
            demand = net_data.travel_demand[origin, destination]
            if demand > 0
                add_demand_to_path!(x, demand, state, origin, destination, 
                                  link_dic, edge_list, num_zones)
            end
        end
    end
    
    return x
end

# ## Custom LMO using Shortest Path

# Custom Linear Minimization Oracle using shortest path computations
struct ShortestPathLMO <: FrankWolfe.LinearMinimizationOracle
    graph::Graphs.SimpleDiGraph{Int}
    net_data::NetworkData
    link_dic::SparseMatrixCSC{Int, Int}
    edge_list::Vector{Tuple{Int, Int}}
end

function Boscia.bounded_compute_extreme_point(lmo::ShortestPathLMO, direction, 
                                               lower_bounds, upper_bounds, int_vars)
    x = all_or_nothing_assignment(direction, lmo.net_data, lmo.graph, 
                                  lmo.link_dic, lmo.edge_list)
    for (i, var_idx) in enumerate(int_vars)
        if direction[var_idx] < 0
            x[var_idx] = upper_bounds[i]
        else
            x[var_idx] = lower_bounds[i]
        end
    end
    return x
end

function Boscia.is_simple_linear_feasible(lmo::ShortestPathLMO, x)
    num_zones = lmo.net_data.num_zones
    num_edges = lmo.net_data.num_edges
    return all(x .>= -1e-6)
end

# ## MOI Model Setup

# Build MOI model with flow conservation and network design constraints
# Big-M formulation: x[dest, edge] <= M * y[edge]
# When y=1 (restore), flow can be up to M
# When y=0 (closed), flow must be 0
# Indicator constraint: y[edge] = 0 => x[dest, edge] = 0
# Note: Indicator constraints may not be supported by all solvers
function build_moi_model(net_data, removed_edges, use_big_m=true)
    optimizer = HiGHS.Optimizer()
    MOI.set(optimizer, MOI.Silent(), true)
    num_zones = net_data.num_zones
    num_edges = net_data.num_edges
    num_removed = length(removed_edges)
    num_flow_vars = num_zones * num_edges  # x[dest, edge]
    num_agg_vars = num_edges  # x_agg[edge]
    num_design_vars = num_removed  # y[removed_edge] binary
    total_vars = num_flow_vars + num_agg_vars + num_design_vars
    x = MOI.add_variables(optimizer, num_flow_vars)
    x_agg = MOI.add_variables(optimizer, num_agg_vars)
    y = MOI.add_variables(optimizer, num_design_vars)
    for i in 1:num_flow_vars
        MOI.add_constraint(optimizer, x[i], MOI.GreaterThan(0.0))
    end
    for i in 1:num_agg_vars
        MOI.add_constraint(optimizer, x_agg[i], MOI.GreaterThan(0.0))
    end
    for i in 1:num_design_vars
        MOI.add_constraint(optimizer, y[i], MOI.ZeroOne())
    end
    edge_list = [(net_data.init_nodes[i], net_data.term_nodes[i]) for i in 1:num_edges]
    edge_dict = Dict(edge_list[i] => i for i in eachindex(edge_list))
    incoming = Dict{Int, Vector{Int}}()
    outgoing = Dict{Int, Vector{Int}}()
    
    for (idx, (src, dst)) in enumerate(edge_list)
        if !haskey(outgoing, src)
            outgoing[src] = Int[]
        end
        push!(outgoing[src], idx)
        
        if !haskey(incoming, dst)
            incoming[dst] = Int[]
        end
        push!(incoming[dst], idx)
    end
    for dest in 1:num_zones
        for node in 1:net_data.num_nodes
            terms = MOI.ScalarAffineTerm{Float64}[]
            if haskey(outgoing, node)
                for edge_idx in outgoing[node]
                    push!(terms, MOI.ScalarAffineTerm(1.0, x[(dest-1)*num_edges + edge_idx]))
                end
            end
            if haskey(incoming, node)
                for edge_idx in incoming[node]
                    push!(terms, MOI.ScalarAffineTerm(-1.0, x[(dest-1)*num_edges + edge_idx]))
                end
            end
            if node == dest
                rhs = -sum(net_data.travel_demand[:, dest])
            elseif node <= num_zones
                rhs = net_data.travel_demand[node, dest]
            else
                rhs = 0.0
            end
            MOI.add_constraint(optimizer, 
                             MOI.ScalarAffineFunction(terms, 0.0),
                             MOI.EqualTo(rhs))
        end
    end
    for edge_idx in 1:num_edges
        terms = [MOI.ScalarAffineTerm(1.0, x_agg[edge_idx])]
        for dest in 1:num_zones
            push!(terms, MOI.ScalarAffineTerm(-1.0, x[(dest-1)*num_edges + edge_idx]))
        end
        MOI.add_constraint(optimizer,
                         MOI.ScalarAffineFunction(terms, 0.0),
                         MOI.EqualTo(0.0))
    end
    max_flow = 1.5 * sum(net_data.travel_demand)
    for (y_idx, edge) in enumerate(removed_edges)
        edge_idx = edge_dict[edge]
        for dest in 1:num_zones
            var_idx = (dest - 1) * num_edges + edge_idx
            if use_big_m
                terms = [
                    MOI.ScalarAffineTerm(1.0, x[var_idx]),
                    MOI.ScalarAffineTerm(-max_flow, y[y_idx])
                ]
                MOI.add_constraint(optimizer,
                                 MOI.ScalarAffineFunction(terms, 0.0),
                                 MOI.LessThan(0.0))
            else
                indicator_func = MOI.VectorAffineFunction(
                    [
                        MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, y[y_idx])),
                        MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[var_idx]))
                    ],
                    [0.0, 0.0]
                )
                MOI.add_constraint(optimizer, indicator_func,
                                 MOI.Indicator{MOI.ACTIVATE_ON_ZERO}(MOI.EqualTo(0.0)))
            end
        end
    end
    return optimizer, edge_list
end

# ## Objective Function and Gradient

# BPR (Bureau of Public Roads) travel time function and gradient (for MOI-based LMO)
#
# This function builds the objective function and gradient for the MOI-based approach.
# The objective function computes:
# - BPR travel time: t = t0 * (flow + b * flow^(power+1) / capacity^power / (power+1))
# - Design cost: sum of cost_per_edge[i] * y[i] for each restored edge
# - No penalty terms needed - constraints are enforced by the MOI model
#
# The gradient function computes derivatives of the objective with respect to:
# - Aggregate flows: d/d(flow) of BPR function
# - Design variables: cost_per_edge[i] for each restored edge
#
# Returns: (f, grad!) where f is the objective function and grad! is the gradient function
function build_objective_and_gradient(net_data, removed_edges, cost_per_edge)
    num_zones = net_data.num_zones
    num_edges = net_data.num_edges
    num_removed = length(removed_edges)
    function f(x)
        x = max.(x, 0.0)
        total = 0.0
        agg_start = num_zones * num_edges + 1
        agg_end = num_zones * num_edges + num_edges
        x_agg = @view x[agg_start:agg_end]
        for i in 1:num_edges
            flow = x_agg[i]
            t0 = net_data.free_flow_time[i]
            b = net_data.b[i]
            cap = net_data.capacity[i]
            p = net_data.power[i]
            total += t0 * (flow + b * flow^(p + 1) / cap^p / (p + 1))
        end
        design_start = num_zones * num_edges + num_edges + 1
        for i in 1:num_removed
            total += cost_per_edge[i] * x[design_start + i - 1]
        end
        return total
    end
    function grad!(storage, x)
        x = max.(x, 0.0)
        fill!(storage, 0.0)
        agg_start = num_zones * num_edges + 1
        agg_end = num_zones * num_edges + num_edges
        x_agg = @view x[agg_start:agg_end]
        for i in 1:num_edges
            flow = x_agg[i]
            t0 = net_data.free_flow_time[i]
            b = net_data.b[i]
            cap = net_data.capacity[i]
            p = net_data.power[i]
            storage[agg_start + i - 1] = t0 * (1 + b * flow^p / cap^p)
        end
        for dest in 1:num_zones
            for edge in 1:num_edges
                storage[(dest - 1) * num_edges + edge] = storage[agg_start + edge - 1]
            end
        end
        design_start = num_zones * num_edges + num_edges + 1
        for i in 1:num_removed
            storage[design_start + i - 1] = cost_per_edge[i]
        end
        return storage
    end
    return f, grad!
end

# BPR objective WITH penalty terms for linking constraints (for Custom LMO)
#
# This function builds the objective function and gradient for the Custom LMO approach.
# Since the shortest-path oracle cannot enforce linking constraints x[dest,edge] <= M * y[edge]
# as hard constraints, we add penalty terms to the objective function to discourage violations.
#
# The objective function computes:
# - BPR travel time: t = t0 * (flow + b * flow^(power+1) / capacity^power / (power+1))
# - Design cost: sum of cost_per_edge[i] * y[i] for each restored edge
# - Penalty terms: penalty_weight * sum_i sum_dest max(0, x[dest,removed_edge_i] - M * y[i])^penalty_exponent
#
# The gradient function computes derivatives of the objective with respect to:
# - Aggregate flows: d/d(flow) of BPR function + penalty gradient w.r.t. flows
# - Design variables: cost_per_edge[i] + penalty gradient w.r.t. design variables
#
# Parameters:
# - penalty_weight: Weight of the penalty term (default: 1e6)
# - penalty_exponent: Exponent for the penalty term (default: 2.0)
#   - penalty_exponent = 1: Linear penalty (L1)
#   - penalty_exponent = 2: Quadratic penalty (L2, most common)
#   - penalty_exponent > 2: Higher order penalties (stronger enforcement)
#
# Returns: (f, grad!) where f is the objective function and grad! is the gradient function
function build_objective_and_gradient_with_penalty(net_data, removed_edges, cost_per_edge, 
                                                    penalty_weight=1e6, penalty_exponent=2.0)
    num_zones = net_data.num_zones
    num_edges = net_data.num_edges
    num_removed = length(removed_edges)
    edge_list = [(net_data.init_nodes[i], net_data.term_nodes[i]) for i in 1:num_edges]
    removed_edge_indices = [findfirst(e -> e == removed_edge, edge_list) 
                            for removed_edge in removed_edges]
    max_flow = 1.5 * sum(net_data.travel_demand)
    function f(x)
        x = max.(x, 0.0)
        total = 0.0
        agg_start = num_zones * num_edges + 1
        agg_end = num_zones * num_edges + num_edges
        x_agg = @view x[agg_start:agg_end]
        for i in 1:num_edges
            flow = x_agg[i]
            t0 = net_data.free_flow_time[i]
            b = net_data.b[i]
            cap = net_data.capacity[i]
            p = net_data.power[i]
            total += t0 * (flow + b * flow^(p + 1) / cap^p / (p + 1))
        end
        design_start = num_zones * num_edges + num_edges + 1
        for i in 1:num_removed
            total += cost_per_edge[i] * x[design_start + i - 1]
        end
        for (y_idx, edge_idx) in enumerate(removed_edge_indices)
            if edge_idx !== nothing
                y_val = x[design_start + y_idx - 1]
                for dest in 1:num_zones
                    flow_idx = (dest - 1) * num_edges + edge_idx
                    flow_val = x[flow_idx]
                    violation = max(0.0, flow_val - max_flow * y_val)
                    total += penalty_weight * violation^penalty_exponent
                end
            end
        end
        return total
    end
    function grad!(storage, x)
        x = max.(x, 0.0)
        fill!(storage, 0.0)
        agg_start = num_zones * num_edges + 1
        agg_end = num_zones * num_edges + num_edges
        x_agg = @view x[agg_start:agg_end]
        for i in 1:num_edges
            flow = x_agg[i]
            t0 = net_data.free_flow_time[i]
            b = net_data.b[i]
            cap = net_data.capacity[i]
            p = net_data.power[i]
            storage[agg_start + i - 1] = t0 * (1 + b * flow^p / cap^p)
        end
        for dest in 1:num_zones
            for edge in 1:num_edges
                storage[(dest - 1) * num_edges + edge] = storage[agg_start + edge - 1]
            end
        end
        design_start = num_zones * num_edges + num_edges + 1
        for i in 1:num_removed
            storage[design_start + i - 1] = cost_per_edge[i]
        end
        for (y_idx, edge_idx) in enumerate(removed_edge_indices)
            if edge_idx !== nothing
                y_val = x[design_start + y_idx - 1]
                for dest in 1:num_zones
                    flow_idx = (dest - 1) * num_edges + edge_idx
                    flow_val = x[flow_idx]
                    violation = max(0.0, flow_val - max_flow * y_val)
                    if violation > 1e-10
                        grad_coeff = penalty_weight * penalty_exponent * violation^(penalty_exponent - 1)
                        storage[flow_idx] += grad_coeff
                        storage[design_start + y_idx - 1] += grad_coeff * (-max_flow)
                    end
                end
            end
        end
        return storage
    end
    return f, grad!
end

# ## Helper Functions for Results

function print_solution(x, net_data, removed_edges, edge_list, method_name)
    println("\n" * "-"^70)
    println("Solution using $method_name")
    println("-"^70)
    num_zones = net_data.num_zones
    num_edges = net_data.num_edges
    num_removed = length(removed_edges)
    function node_label(node)
        if node == 1
            return "S1"
        elseif node == 2
            return "S2"
        elseif node == 3
            return "D"
        elseif node == 4
            return "1"
        elseif node == 5
            return "2"
        elseif node == 6
            return "3"
        elseif node == 7
            return "4"
        elseif node == 8
            return "5"
        else
            return string(node)
        end
    end
    design_start = num_zones * num_edges + num_edges + 1
    design_vars = x[design_start:end]
    
    println("\nEdges to restore:")
    for (i, edge) in enumerate(removed_edges)
        status = design_vars[i] > 0.5 ? "RESTORE" : "KEEP CLOSED"
        from_label = node_label(edge[1])
        to_label = node_label(edge[2])
        println("  Edge ($from_label → $to_label): y = $(round(design_vars[i], digits=3)) → $status")
    end
    agg_start = num_zones * num_edges + 1
    println("\nAggregate flows on edges:")
    for i in 1:num_edges
        flow = x[agg_start + i - 1]
        if flow > 1e-6
            from_label = node_label(edge_list[i][1])
            to_label = node_label(edge_list[i][2])
            println("  Edge ($from_label → $to_label): flow = $(round(flow, digits=3))")
        end
    end
end

# ## Example Execution

# Load network
net_data = load_braess_network()
println("\nNetwork: Two-Source Network with Purchasable Edge")
println("  Nodes: $(net_data.num_nodes) (2 sources, 5 intermediate, 1 destination)")
println("  Edges: $(net_data.num_edges)")
println("  Sources: S1 (node 1), S2 (node 2)")
println("  Destination: D (node 3)")
println("  Intermediate nodes: 4, 5, 6, 7, 8")
println("  Demand: 1 unit from each source (2 units total)")

# Define potentially purchasable edges (edges that need design decision)
removed_edges = [(4, 5)]  # Optional edge from node_1 (intermediate node 4) to node_2 (intermediate node 5)
cost_per_edge = [0.5]  # Cost to purchase the edge

println("\nPurchasable edges (need design decision): $removed_edges")
println("Cost to restore: $cost_per_edge")

# Build edge list for display
edge_list = [(net_data.init_nodes[i], net_data.term_nodes[i]) 
             for i in 1:net_data.num_edges]

# ## Solve with MOI-based LMO

println("\n" * "="^70)
println("Solving with MOI-based LMO (MIP solver models feasible region)")
println("="^70)

# Build MOI model
optimizer, _ = build_moi_model(net_data, removed_edges, true)

# Create Boscia LMO from MOI model
lmo_moi = FrankWolfe.MathOptLMO(optimizer)

# Build objective
f_moi, grad_moi! = build_objective_and_gradient(net_data, removed_edges, cost_per_edge)

# Configure settings
settings_moi = Boscia.create_default_settings()
settings_moi.branch_and_bound[:verbose] = true

# Solve with Boscia
x_moi, _, result_moi = Boscia.solve(f_moi, grad_moi!, lmo_moi, settings=settings_moi)

print_solution(x_moi, net_data, removed_edges, edge_list, "MOI-based LMO")

# ## Solve with Custom LMO

println("\n" * "="^70)
println("Solving with Custom LMO (shortest path oracle)")
penalty_weight = 1e3
penalty_exponent = 1.5
println("Penalty weight: $penalty_weight, Penalty exponent: $penalty_exponent")
println("="^70)

# Build graph
graph = Graphs.SimpleDiGraph(net_data.num_nodes)
edge_list_custom = Tuple{Int,Int}[]
for i in 1:net_data.num_edges
    Graphs.add_edge!(graph, net_data.init_nodes[i], net_data.term_nodes[i])
    push!(edge_list_custom, (net_data.init_nodes[i], net_data.term_nodes[i]))
end

# Build sparse link dictionary
link_dic = sparse(net_data.init_nodes, net_data.term_nodes, 
                 collect(1:net_data.num_edges))

# Create custom LMO for continuous variables
custom_lmo = ShortestPathLMO(graph, net_data, link_dic, edge_list_custom)

# Set up integer bounds for Boscia
num_zones = net_data.num_zones
num_edges = net_data.num_edges
num_removed = length(removed_edges)
total_vars = num_zones * num_edges + num_edges + num_removed

# Binary variables for network design (last num_removed variables)
int_vars = collect((num_zones * num_edges + num_edges + 1):total_vars)

# Create bounds vectors for integer variables
lower_bounds = zeros(Float64, num_removed)  # Binary: lower bound = 0
upper_bounds = ones(Float64, num_removed)   # Binary: upper bound = 1

# Wrap LMO with integer handling
bounded_lmo = Boscia.ManagedLMO(custom_lmo, lower_bounds, upper_bounds, int_vars, total_vars)

# Build objective WITH PENALTY TERMS for linking constraints
f_custom, grad_custom! = build_objective_and_gradient_with_penalty(net_data, removed_edges, cost_per_edge, 
                                                      penalty_weight, penalty_exponent)

# Configure settings
settings_custom = Boscia.create_default_settings()
settings_custom.branch_and_bound[:verbose] = true

# Solve with Boscia
x_custom, _, result_custom = Boscia.solve(f_custom, grad_custom!, bounded_lmo, settings=settings_custom)

print_solution(x_custom, net_data, removed_edges, edge_list, "Custom Shortest-Path LMO")

# ## Comparison of Results

println("\n" * "="^70)
println("Comparison of Results")
println("="^70)
println("MOI-based LMO (Hard Constraints):")
println("  Objective: $(result_moi[:primal_objective])")
println("  Time: $(result_moi[:total_time_in_sec]) seconds")
println("  Formulation: min BPR_cost(x) + restoration_cost(y)")
println("               s.t. x[e] <= M*y[e] (hard constraints)")

println("\nCustom LMO (Penalty Method):")
println("  Objective: $(result_custom[:primal_objective])")
println("  Time: $(result_custom[:total_time_in_sec]) seconds")
println("  Formulation: min BPR_cost(x) + restoration_cost(y) + penalties")
println("               (penalties for violating x[e] <= M*y[e])")

println("\n" * "-"^70)
println("IMPORTANT: These are DIFFERENT objective functions!")
println("The Custom LMO uses a penalized formulation because the shortest-path")
println("oracle cannot enforce the linking constraints x[e] <= M*y[e] directly.")
println("Different formulations → different solutions → different objectives.")
println("="^70)

