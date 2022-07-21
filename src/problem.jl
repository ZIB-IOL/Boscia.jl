
"""
Represents an optimization problem of the form:
```
min_x f(x)
s.t.  A1 x <= b1
      A2 x >= b2
      A3 x == b3
      x_j ∈ {0,1} ∀ j in binary_variables
```
"""
abstract type AbstractSimpleOptimizationProblem end 

mutable struct SimpleOptimizationProblem{F, G, LMO<:FrankWolfe.LinearMinimizationOracle, IB<:IntegerBounds} <: AbstractSimpleOptimizationProblem
    f::F
    g::G
    nvars::Int
    integer_variables::Vector{Int64}
    lmo::LMO
    integer_variable_bounds::IB
    #constraints_lessthan::Vector{Tuple{MOI.ScalarAffineFunction{T}, MOI.LessThan{T}}}
    #constraints_greaterthan::Vector{Tuple{MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T}}}
    #constraints_equalto::Vector{Tuple{MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}}}
end

mutable struct SimpleOptimizationProblemInfeasible{F, G, AT<:FrankWolfe.ActiveSet, DVS<:FrankWolfe.DeletedVertexStorage, LMO<:FrankWolfe.LinearMinimizationOracle, IB<:IntegerBounds} <: AbstractSimpleOptimizationProblem
    f::F
    g::G
    nvars::Int
    integer_variables::Vector{Int64}
    lmo::LMO
    integer_variable_bounds::IB
    active_set::AT
    discarded_verices::DVS
    #constraints_lessthan::Vector{Tuple{MOI.ScalarAffineFunction{T}, MOI.LessThan{T}}}
    #constraints_greaterthan::Vector{Tuple{MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T}}}
    #constraints_equalto::Vector{Tuple{MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}}}
end

"""
Returns the indices of the discrete variables for the branching in `Bonobo.BnBTree`
"""
function Bonobo.get_branching_indices(problem)
    return problem.integer_variables
end

Bonobo.get_branching_indices(root::NamedTuple) = Bonobo.get_branching_indices(root.problem)

"""
Checks if a given vector is valid integral solution. Specifically for mixed problems.
"""
function is_integer_feasible(integer_variables::AbstractVector{<:Integer}, x::AbstractVector; atol=1e-6, rtol=1e-6)
    for idx in integer_variables
        if !isapprox(x[idx], round(x[idx]); atol=atol, rtol=rtol)
            return false
        end
    end
    return true
end

function is_integer_feasible(tree::Bonobo.BnBTree, x::AbstractVector) 
    return is_integer_feasible(tree.root.problem.integer_variables, x; atol=tree.options.atol, rtol=tree.options.rtol) 
end


"""
Return the underlying optimizer
For better access and readability
"""
function get_optimizer(tree::Bonobo.BnBTree)
    return tree.root.problem.lmo.lmo.o
end


"""
Checks if x is valid for all linear and variable bound constraints 
"""
function is_linear_feasible(o::MOI.ModelLike, v::AbstractVector)
    valvar(f) = v[f.value]

    ## ScalarAffine
    # LessThan
    for (F, S) in MOI.get(o, MOI.ListOfConstraintTypesPresent())
        @debug "constraints $F, $S"

        if S != MOI.ZeroOne && !(S <: MOI.Indicator) && S != MOI.Integer
            #@show F, S
            cons_list = MOI.get(o, MOI.ListOfConstraintIndices{F, S}())
            #@show cons_list
            for c_idx in cons_list
                #@show c_idx
                func = MOI.get(o, MOI.ConstraintFunction(), c_idx)
                val = MOIU.eval_variables(valvar, func)
                #@show val
                #@show 
                set = MOI.get(o, MOI.ConstraintSet(), c_idx) # errors here
                @debug("Constraint: $(F)-$(S) $(func) = $(val) in $(set)")
                dist = MOD.distance_to_set(MOD.DefaultDistance(), val, set)
                if dist > 1e-6
                    return false
                end
            end
        end
    end

   # satisfies all constraints
    return true
end

is_linear_feasible(lmo::TimeTrackingLMO, v::AbstractVector) = is_linear_feasible(lmo.lmo.o, v)
is_linear_feasible(lmo::FrankWolfe.LinearMinimizationOracle, v::AbstractVector) = is_linear_feasible(lmo.o, v)
