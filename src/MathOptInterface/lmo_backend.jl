"""
    AbstractLMOBackend

Supertype of all LMO backends used by `Optimizer`.
"""
abstract type AbstractLMOBackend end

"""
    MathOptLMOBackend(optimizer_factory)

LMO backend that solves the linear subproblems with an inner MIP solver,
such as SCIP or HiGHS, wrapped as a `FrankWolfe.MathOptLMO`.
"""
struct MathOptLMOBackend <: AbstractLMOBackend
    optimizer_factory::Any
end

"""
    build_lmo(backend::MathOptLMOBackend, feasible_region_data)

Create a new inner MIP solver, copy the feasible region into it, and wrap it
as a `FrankWolfe.MathOptLMO`.
"""
function build_lmo(backend::MathOptLMOBackend, feasible_region_data)
    inner = MOI.instantiate(backend.optimizer_factory)

    # Silent MIP solver
    if MOI.supports(inner, MOI.Silent())
        MOI.set(inner, MOI.Silent(), true)
    end

    MOI.copy_to(inner, feasible_region_data)
    return FrankWolfe.MathOptLMO(inner)                # Wrap as an LMO
end

"""
    supports_constraint(backend::MathOptLMOBackend, ::Type{F}, ::Type{S}) where {F,S}

Return whether the inner MIP solver supports `F`-in-`S` constraints.
"""
function supports_constraint(backend::MathOptLMOBackend, ::Type{F}, ::Type{S})::Bool where {F,S}
    inner = MOI.instantiate(backend.optimizer_factory)

    return MOI.supports_constraint(inner, F, S)
end

"""
    custom_set_constraints(model)

Return the indices of the `MOI.VectorOfVariables`-in-`CustomSet` constraints
of `model`.
"""
function custom_set_constraints(model)
    s = MOI.ConstraintIndex[]
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        if F == MOI.VectorOfVariables && S <: CustomSet
            append!(s, MOI.get(model, MOI.ListOfConstraintIndices{F,S}()))
        end
    end
    return s
end

"""
    custom_set(model)

Return the custom set of `model`, or `nothing` if there is none.
"""
function custom_set(model)
    s = custom_set_constraints(model)
    if isempty(s)
        return nothing
    end
    return MOI.get(model, MOI.ConstraintSet(), first(s))
end

# Inner LMO of each custom set
custom_lmo(::ZeroOneHypercube) = ZeroOneHypercubeLMO()
custom_lmo(set::ProbabilitySimplex) = ProbabilitySimplexLMO(set.N)
custom_lmo(set::UnitSimplex) = UnitSimplexLMO(set.N)
custom_lmo(set::HyperSimplex) = FrankWolfe.HyperSimplexLMO(set.K, set.radius)
custom_lmo(set::UnitHyperSimplex) = FrankWolfe.UnitHyperSimplexLMO(set.K, set.radius)
custom_lmo(set::ReverseKnapsack) = ReverseKnapsackLMO(set.dimension; N=set.N, upper=set.upper)

# lb, ub of every integer variable for each custom set
custom_bounds(::ZeroOneHypercube) = (0.0, 1.0)
custom_bounds(set::ProbabilitySimplex) = (0.0, floor(set.N))
custom_bounds(set::UnitSimplex) = (0.0, floor(set.N))
custom_bounds(set::HyperSimplex) = (0.0, floor(set.radius))
custom_bounds(set::UnitHyperSimplex) = (0.0, floor(set.radius))
custom_bounds(set::ReverseKnapsack) = (0.0, floor(set.upper))

"""
    integer_variables(model)

Return the sorted indices of the variables of `model` that have an
`MOI.Integer` or `MOI.ZeroOne` constraint.
"""
function integer_variables(model)
    int_vars = Int[]
    for S in (MOI.Integer, MOI.ZeroOne)
        for ci in MOI.get(model, MOI.ListOfConstraintIndices{MOI.VariableIndex,S}())
            x = MOI.get(model, MOI.ConstraintFunction(), ci)
            push!(int_vars, x.value)
        end
    end
    return sort!(unique!(int_vars))
end

"""
    build_custom_set_lmo(model, set::CustomSet)

Build the LMO of `set`, wrapped in a `ManagedLMO` that stores the bounds of the
integer variables and updates them during branching.
"""
function build_custom_set_lmo(model, set::CustomSet)
    n = MOI.get(model, MOI.NumberOfVariables())
    int_vars = integer_variables(model)
    lower, upper = custom_bounds(set)
    lower_bounds = fill(lower, length(int_vars))
    upper_bounds = fill(upper, length(int_vars))
    return ManagedLMO(custom_lmo(set), lower_bounds, upper_bounds, int_vars, n)
end
