# =============================================================================
# Part of this file is copied/adapted from Bonobo.jl (MIT License)
# Original repository: https://wikunia.github.io/Bonobo.jl
# License: MIT License
# =============================================================================

"""
    AbstractNode

The abstract type for a tree node. Your own type for `Node` given to [`initialize`](@ref) needs to subtype it.
The default if you don't provide your own is [`DefaultNode`](@ref).
"""
abstract type AbstractNode end

"""
    AbstractSolution{Node<:AbstractNode, Value}

The abstract type for a `Solution` object. The default is [`DefaultSolution`](@ref).
It is parameterized by `Node` and `Value` where `Value` is the value which describes the full solution i.e the value for every variable.
"""
abstract type AbstractSolution{Node<:AbstractNode,Value} end

"""
    BnBNodeInfo

Holds the necessary information of every node.
This needs to be added by every `AbstractNode` as `std::BnBNodeInfo`

```julia
id :: Int
lb :: Float64
ub :: Float64
depth :: Int
```
"""
mutable struct BnBNodeInfo
    id::Int
    lb::Float64
    ub::Float64
    depth::Int
end

"""
    AbstractTraverseStrategy

The abstract type for a traverse strategy. 
If you implement a new traverse strategy this must be the supertype. 

If you want to implement your own strategy the [`get_next_node`](@ref) function needs a new method 
which dispatches on the `traverse_strategy` argument. 
"""
abstract type AbstractTraverseStrategy end

"""
    AbstractBranchStrategy

The abstract type for a branching strategy. 
If you implement a new branching strategy, this must be the supertype. 

If you want to implement your own strategy, you must implement a new method for [`get_branching_variable`](@ref)
which dispatches on the `branch_strategy` argument. 
"""
abstract type AbstractBranchStrategy end

"""
    BestFirstSearch <: AbstractTraverseStrategy

The BestFirstSearch traverse strategy always picks the node with the lowest bound first.
If there is a tie then the smallest node id is used as a tie breaker.
"""
struct BestFirstSearch <: AbstractTraverseStrategy end

struct DepthFirstSearch <: AbstractTraverseStrategy end

@deprecate BFS() BestFirstSearch() false

"""
    FIRST <: AbstractBranchStrategy

The `FIRST` strategy always picks the first variable which isn't fixed yet and can be branched on.
"""
struct FIRST <: AbstractBranchStrategy end

"""
    MOST_INFEASIBLE <: AbstractBranchStrategy

The `MOST_INFEASIBLE` strategy always picks the variable which is furthest away from being "fixed" and can be branched on.
"""
struct MOST_INFEASIBLE <: AbstractBranchStrategy end

mutable struct Options
    traverse_strategy::AbstractTraverseStrategy
    branch_strategy::AbstractBranchStrategy
    atol::Float64
    rtol::Float64
    dual_gap_limit::Float64
    abs_gap_limit::Float64
end

"""
    BnBTree{Node<:AbstractNode,Root,Value,Solution<:AbstractSolution{Node,Value}}

Holds all the information of the branch and bound tree. 

```
incumbent::Float64 - The best objective value found so far. Is stores as problem is a minimization problem
incumbent_solution::Solution - The currently best solution object
lb::Float64        - The highest current lower bound 
solutions::Vector{Solution} - A list of solutions
node_queue::PriorityQueue{Int,Tuple{Float64, Int}} - A priority queue with key being the node id and the priority consists of the node lower bound and the node id.
nodes::Dict{Int, Node}  - A dictionary of all nodes with key being the node id and value the actual node.
root::Root      - The root node see [`set_root!`](@ref)
branching_indices::Vector{Int} - The indices to be able to branch on used for [`get_branching_variable`](@ref)
num_nodes::Int  - The number of nodes created in total
sense::Symbol   - The objective sense: `:Max` or `:Min`.
options::Options  - All options for the branch and bound tree. See [`Options`](@ref).
```
"""
mutable struct BnBTree{Node<:AbstractNode,Root,Value,Solution<:AbstractSolution{Node,Value}}
    incumbent::Float64
    incumbent_solution::Union{Nothing,Solution}
    lb::Float64
    solutions::Vector{Solution}
    node_queue::PriorityQueue{Int,Tuple{Float64,Int}}
    nodes::Dict{Int,Node}
    root::Root
    branching_indices::Vector{Int}
    num_nodes::Int
    sense::Symbol
    options::Options
end

mutable struct FrankWolfeSolution{Node<:AbstractNode,Value,T<:Real} <: AbstractSolution{Node,Value}
    objective::T
    solution::Value
    node::Node
    source::Symbol
    time::Float64
end

"""
    NodeInfo

Holds the necessary information of every node.
This needs to be added by every `AbstractNode` as `std::NodeInfo`

This variant is more flexibel than BnBNodeInfo.
"""
mutable struct NodeInfo{T<:Real}
    id::Int
    lb::T
    ub::T
    depth::Int
end

function Base.convert(::Type{NodeInfo{T}}, std::BnBNodeInfo) where {T<:Real}
    return NodeInfo(std.id, T(std.lb), T(std.ub), std.depth)
end
