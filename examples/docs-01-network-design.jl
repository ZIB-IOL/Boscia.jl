# # Network Design Problem
#
# We demonstrate solving a network design problem using Boscia.jl.
# We want to minimize the total travel time over a network:
# ```math
# \begin{aligned}
#     \min_{\mathbf{x}, \mathbf{y}} \quad & r^T \mathbf{y} + c(\mathbf{x})  && \\
#     \text{s.t.} \quad & x_e = \sum_{z \in \mathcal{Z}} x_e^z && \forall e \in \mathcal{E} \\
#     & \mathbf{x}^z \in \mathcal{X}^z =
#     \begin{cases}
#         \sum_{e \in \delta^{+}(i)} x_e^z - \sum_{e \in \delta^{-}(i)} x_e^z = 0, & \forall i \in \mathcal{V} \setminus (\mathcal{O} \cup \mathcal{Z}) \\
#         \sum_{e \in \delta^{+}(i)} x_e^z = d_i^z, & \forall i \in \mathcal{O} \\
#         \sum_{e \in \delta^{-}(z)} x_e^z = \sum_{i \in \mathcal{O}} d_i^z
#     \end{cases} && \forall z \in \mathcal{Z}. \\  
#       & y_e = 0 \Rightarrow x_e \leq 0 && \forall e \in \mathcal{R} \\
#       & \mathbf{y} \in \mathcal{Y} \subset \{0,1\}^{|\mathcal{R}|}
# \end{aligned}
# ```
# where
# ```math
# c(x) = \sum_{e \in E} c_e(x) = α_e + β_e*x_e + γ_e*x_e^{ρ_e}
# ```
# with $α_e$, $β_e$, and $γ_e$ are constants and the exponent $ρ_e > 1$ model the congestion effect.
# Given a set of purchasable/optional edges $\mathcal{R}$, we want to decide which edges to build/restore.
# $\mathcal{E} denotes the set of edges, $\mathcal{S}$ and 
# $\mathcal{O}$ denote the set of source and destination nodes, respectively.
# The design cost is linear and the operating cost of the
# network is modeled as a traffic assignment problem.
# We solve the problem with two approaches based on the formulations in ["Network design for the traffic
# assignment problem with mixed-integer Frank-Wolfe"](https://arxiv.org/abs/2402.00166) by Sharma et al.:
# 1. Using [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl) (MOI) to model the feasible region
# 2. A penalty formulation using a customized Linear Minimization Oracle based on shortest path algorithms

# ## Imports and Setup
#
# We start by generating the network.
using Boscia
using FrankWolfe
using Graphs
using SparseArrays
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
using HiGHS 

println("\nDocumentation Example 01: Network Design Problem")

# The graph structure is shown below.
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

# The example is a small network with 8 nodes.
# Nodes 1 and 2 are the sources, node 3 is the destination, and nodes 4-8 are the intermediate nodes.
# The network is a directed graph with 12 edges.
# The edges are:
# 1. 1 - 4
# 2. 2 - 6
# 3. 4 - 6
# 4. 6 - 7
# 5. 7 - 8
# 6. 8 - 3
# 7. 5 - 3
# 8. 4 - 5 (optional edge)
# Edge 8 will be the purchasable edge, i.e. an edge for which we have to
# decide to restore it or keep it closed.
# Travel demand is 1 unit from each source to the destination.
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

# ## Direct modelling via MathOptInterface
#
# With MOI, we can directly model the feasible region.
# The linking constraints $y_e = 0 \Rightarrow x_e \leq 0 \forall e \in \mathcal{R}$
# can be modelled either as bigM-constraints or indicator constraints (if the chosen MIP solver supports them).
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

# **BPR (Bureau of Public Roads) travel time function and gradient (for MOI-based LMO)**
#
# This function builds the objective function and gradient for the MOI-based approach.
# The objective function computes:
# - BPR travel time: t = t0 * (flow + b * flow^(power+1) / capacity^power / (power+1))
# - Design cost: sum of cost_per_edge[i] * y[i] for each restored edge
#
# The gradient function computes derivatives of the objective with respect to:
# - Aggregate flows: d/d(flow) of BPR function
# - Design variables: cost_per_edge[i] for each restored edge
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

# ## Calling Boscia on the MOI formulation

# Define potentially purchasable edges (edges that need design decision).
removed_edges = [(4, 5)]  # Optional edge from node_1 (intermediate node 4) to node_2 (intermediate node 5)
cost_per_edge = [0.5]  # Cost to purchase the edge

net_data = load_braess_network()
optimizer, _ = build_moi_model(net_data, removed_edges, true)
lmo_moi = FrankWolfe.MathOptLMO(optimizer)

f_moi, grad_moi! = build_objective_and_gradient(net_data, removed_edges, cost_per_edge)

# This problem does not require any particular settings.
# We only enable the logs and run with the default settings.
settings_moi = Boscia.create_default_settings()
settings_moi.branch_and_bound[:verbose] = true

x_moi, _, result_moi = Boscia.solve(f_moi, grad_moi!, lmo_moi, settings=settings_moi)

# ## Penalty formulation and custom LMO
# 
# The LMO of the previous formulation is computationally expensive due 
# to the linking constraints. Also, we cannot really exploit the network
# structure. Thus, Sharma et al. introduce a penalty formulation adding the 
# linking constraints to the objective.
# ```math	
# \mu \sum_{z \in \mathcal{Z}} \sum_{e \in \mathcal{R}} \max(x_e^z - M^z y_e, 0)^p 
# ```
# The advantage of this formulation is that we can separate the LMO call for 
# flow variables $x$ and design variables $y$.
# On the other hand, we have estimate $\mu$ to solve the problem exactly.
# The LMO for the flow variables implements a shortest path algorithm.
# As for the design space $\mathcal{Y}$, we assume it is simply the hypercube.

# We create a custom LMO for the penalty formulation.
# The bound management will be handled by Boscia itself,
# so we only need to implement the `bounded_compute_extreme_point` and `is_simple_linear_feasible` methods.
struct ShortestPathLMO <: FrankWolfe.LinearMinimizationOracle
    graph::Graphs.SimpleDiGraph{Int}
    net_data::NetworkData
    link_dic::SparseMatrixCSC{Int, Int}
    edge_list::Vector{Tuple{Int, Int}}
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


# **BPR objective WITH penalty terms for linking constraints (for Custom LMO)**
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

# ## Calling Boscia on the penalty formulation

penalty_weight = 1e3
penalty_exponent = 1.5

# Generate the graph structure.
graph = Graphs.SimpleDiGraph(net_data.num_nodes)
edge_list_custom = Tuple{Int,Int}[]
for i in 1:net_data.num_edges
    Graphs.add_edge!(graph, net_data.init_nodes[i], net_data.term_nodes[i])
    push!(edge_list_custom, (net_data.init_nodes[i], net_data.term_nodes[i]))
end

link_dic = sparse(net_data.init_nodes, net_data.term_nodes, 
                 collect(1:net_data.num_edges))

custom_lmo = ShortestPathLMO(graph, net_data, link_dic, edge_list_custom)

# Set the bounds for the binary variables.
num_zones = net_data.num_zones
num_edges = net_data.num_edges
num_removed = length(removed_edges)
total_vars = num_zones * num_edges + num_edges + num_removed

int_vars = collect((num_zones * num_edges + num_edges + 1):total_vars) # last num_removed variables
lower_bounds = zeros(Float64, num_removed)  # Binary: lower bound = 0
upper_bounds = ones(Float64, num_removed)   # Binary: upper bound = 1

# To have Boscia handle the bounds, we need to wrap our LMO in an instance of `ManagedLMO`.
bounded_lmo = Boscia.ManagedLMO(custom_lmo, lower_bounds, upper_bounds, int_vars, total_vars)

f_custom, grad_custom! = build_objective_and_gradient_with_penalty(net_data, removed_edges, cost_per_edge, 
                                                      penalty_weight, penalty_exponent)

settings_custom = Boscia.create_default_settings()
settings_custom.branch_and_bound[:verbose] = true

x_custom, _, result_custom = Boscia.solve(f_custom, grad_custom!, bounded_lmo, settings=settings_custom)
