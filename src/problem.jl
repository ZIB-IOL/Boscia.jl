"""
Enum for the solving stage
"""
@enum Solve_Stage::Int32 begin
    SOLVING = 0
    OPT_GAP_REACHED = 1
    OPT_TREE_EMPTY = 2
    TIME_LIMIT_REACHED = 3
end

abstract type AbstractSimpleOptimizationProblem end

"""
Represents an optimization problem of the form:
```
min_x f(x)
s.t.  x ∈ X (given by the LMO)
      x_j ∈ Z ∀ j in integer_variables
```
"""
mutable struct SimpleOptimizationProblem{
    F,
    G,
    TLMO<:TimeTrackingLMO,
    IB<:IntegerBounds,
} <: AbstractSimpleOptimizationProblem
    f::F
    g::G
    nvars::Int
    integer_variables::Vector{Int64}
    tlmo::TLMO
    integer_variable_bounds::IB
    solving_stage::Solve_Stage
    #constraints_lessthan::Vector{Tuple{MOI.ScalarAffineFunction{T}, MOI.LessThan{T}}}
    #constraints_greaterthan::Vector{Tuple{MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T}}}
    #constraints_equalto::Vector{Tuple{MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}}}
end

SimpleOptimizationProblem(f, g, n, int_vars, tlmo, int_bounds) =
    SimpleOptimizationProblem(f, g, n, int_vars, tlmo, int_bounds, SOLVING)

"""
Returns the indices of the discrete variables for the branching in `Bonobo.BnBTree`
"""
function Bonobo.get_branching_indices(problem::SimpleOptimizationProblem)
    return problem.integer_variables
end

Bonobo.get_branching_indices(root::NamedTuple) = Bonobo.get_branching_indices(root.problem)

"""
Checks if a given vector is valid integral solution. Specifically for mixed problems.
"""
function is_integer_feasible(
    integer_variables::AbstractVector{<:Integer},
    x::AbstractVector;
    atol=1e-6,
    rtol=1e-6,
)
    for idx in integer_variables
        if !isapprox(x[idx], round(x[idx]); atol=atol, rtol=rtol)
            return false
        end
    end
    return true
end

function is_integer_feasible(tree::Bonobo.BnBTree, x::AbstractVector)
    indicator_feasible = indicator_present(tree) ? is_indicator_feasible(tree.root.problem.tlmo.blmo.o, x) : true
    return is_integer_feasible(
        tree.root.problem.integer_variables,
        x;
        atol=tree.options.atol,
        rtol=tree.options.rtol,
    ) && indicator_feasible
end

"""
Return the underlying optimizer
For better access and readability
"""
#function get_optimizer(tree::Bonobo.BnBTree)
#    return tree.root.problem.lmo.lmo.o
#end

"""
Checks if x is valid for all linear and variable bound constraints 
"""
is_linear_feasible(lmo::TimeTrackingLMO, v::AbstractVector) = is_linear_feasible(lmo.blmo, v)

"""
Are indicator constraints present
"""
indicator_present(time_lmo::TimeTrackingLMO) = indicator_present(time_lmo.blmo)
indicator_present(tree::Bonobo.BnBTree) = indicator_present(tree.root.problem.tlmo.blmo)
