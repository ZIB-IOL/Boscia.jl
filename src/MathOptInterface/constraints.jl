# Constraints of Boscia.Optimizer.
#
# Supported constraints:
#
#   VariableIndex        in GreaterThan, LessThan, EqualTo, Interval,
#                           Integer, ZeroOne
#   ScalarAffineFunction in GreaterThan, LessThan, EqualTo
#   VectorAffineFunction in Indicator   (if the inner MIP solver supports it, e.g. SCIP)
#   VectorOfVariables    in CustomSet: ZeroOneHypercube, ProbabilitySimplex,
#                           UnitSimplex, HyperSimplex, UnitHyperSimplex,
#                           ReverseKnapsack

# ---------------------------------------------------------------------------
# VariableIndex constraints: variable bounds and integrality
# ---------------------------------------------------------------------------

# x >= lower
function MOI.add_constraint(
    optimizer::Optimizer,
    variable::MOI.VariableIndex,
    set::MOI.GreaterThan{Float64},
)
    return MOI.add_constraint(optimizer.model, variable, set)
end

function MOI.get(
    optimizer::Optimizer,
    ::MOI.NumberOfConstraints{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)::Int
    return MOI.get(
        optimizer.model,
        MOI.NumberOfConstraints{MOI.VariableIndex,MOI.GreaterThan{Float64}}(),
    )
end


# x <= upper
function MOI.add_constraint(
    optimizer::Optimizer,
    variable::MOI.VariableIndex,
    set::MOI.LessThan{Float64},
)
    return MOI.add_constraint(optimizer.model, variable, set)
end

function MOI.get(
    optimizer::Optimizer,
    ::MOI.NumberOfConstraints{MOI.VariableIndex,MOI.LessThan{Float64}},
)::Int
    return MOI.get(
        optimizer.model,
        MOI.NumberOfConstraints{MOI.VariableIndex,MOI.LessThan{Float64}}(),
    )
end

# lower <= x <= upper
function MOI.add_constraint(
    optimizer::Optimizer,
    variable::MOI.VariableIndex,
    set::MOI.Interval{Float64},
)
    return MOI.add_constraint(optimizer.model, variable, set)
end

function MOI.get(
    optimizer::Optimizer,
    ::MOI.NumberOfConstraints{MOI.VariableIndex,MOI.Interval{Float64}},
)::Int
    return MOI.get(
        optimizer.model,
        MOI.NumberOfConstraints{MOI.VariableIndex,MOI.Interval{Float64}}(),
    )
end

# x == value (fixed variable)
function MOI.add_constraint(
    optimizer::Optimizer,
    variable::MOI.VariableIndex,
    set::MOI.EqualTo{Float64},
)
    return MOI.add_constraint(optimizer.model, variable, set)
end

function MOI.get(
    optimizer::Optimizer,
    ::MOI.NumberOfConstraints{MOI.VariableIndex,MOI.EqualTo{Float64}},
)::Int
    return MOI.get(
        optimizer.model,
        MOI.NumberOfConstraints{MOI.VariableIndex,MOI.EqualTo{Float64}}(),
    )
end

# x integer
function MOI.add_constraint(optimizer::Optimizer, variable::MOI.VariableIndex, set::MOI.Integer)
    return MOI.add_constraint(optimizer.model, variable, set)
end

function MOI.get(
    optimizer::Optimizer,
    ::MOI.NumberOfConstraints{MOI.VariableIndex,MOI.Integer},
)::Int
    return MOI.get(optimizer.model, MOI.NumberOfConstraints{MOI.VariableIndex,MOI.Integer}())
end

# x binary (x in {0, 1})
function MOI.add_constraint(optimizer::Optimizer, variable::MOI.VariableIndex, set::MOI.ZeroOne)
    return MOI.add_constraint(optimizer.model, variable, set)
end

function MOI.get(
    optimizer::Optimizer,
    ::MOI.NumberOfConstraints{MOI.VariableIndex,MOI.ZeroOne},
)::Int
    return MOI.get(optimizer.model, MOI.NumberOfConstraints{MOI.VariableIndex,MOI.ZeroOne}())
end

"""
    MOI.supports_constraint(
        optimizer::Optimizer,
        ::Type{MOI.VariableIndex},
        ::Type{S},
    ) where {S}

Return whether a `VariableIndex`-in-`S` constraint is supported.
"""
function MOI.supports_constraint(
    optimizer::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{S},
) where {
    S<:Union{
        MOI.GreaterThan{Float64},
        MOI.LessThan{Float64},
        MOI.EqualTo{Float64},
        MOI.Interval{Float64},
        MOI.Integer,
        MOI.ZeroOne,
    },
}


    return supports_constraint(optimizer.backend, MOI.VariableIndex, S)

end


# ---------------------------------------------------------------------------
# ScalarAffineFunction constraints: linear constraints a'x + b in S
# ---------------------------------------------------------------------------

# a'x + b >= lower
function MOI.add_constraint(
    optimizer::Optimizer,
    func::MOI.ScalarAffineFunction{Float64},
    set::MOI.GreaterThan{Float64},
)
    return MOI.add_constraint(optimizer.model, func, set)
end

function MOI.get(
    optimizer::Optimizer,
    ::MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}},
)::Int
    return MOI.get(
        optimizer.model,
        MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}(),
    )
end

# a'x + b <= upper
function MOI.add_constraint(
    optimizer::Optimizer,
    func::MOI.ScalarAffineFunction{Float64},
    set::MOI.LessThan{Float64},
)
    return MOI.add_constraint(optimizer.model, func, set)
end

function MOI.get(
    optimizer::Optimizer,
    ::MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}},
)::Int
    return MOI.get(
        optimizer.model,
        MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}(),
    )
end

# a'x + b == value
function MOI.add_constraint(
    optimizer::Optimizer,
    func::MOI.ScalarAffineFunction{Float64},
    set::MOI.EqualTo{Float64},
)
    return MOI.add_constraint(optimizer.model, func, set)
end


function MOI.get(
    optimizer::Optimizer,
    ::MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}},
)::Int
    return MOI.get(
        optimizer.model,
        MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}(),
    )
end

"""
    MOI.supports_constraint(
        optimizer::Optimizer,
        ::Type{MOI.ScalarAffineFunction{Float64}},
        ::Type{S},
    ) where {S}

Return whether a `ScalarAffineFunction`-in-`S` constraint is supported.
"""
function MOI.supports_constraint(
    optimizer::Optimizer,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{S},
) where {S<:Union{MOI.GreaterThan{Float64},MOI.LessThan{Float64},MOI.EqualTo{Float64}}}

    return supports_constraint(optimizer.backend, MOI.ScalarAffineFunction{Float64}, S)
end

# ---------------------------------------------------------------------------
# Indicator constraints: z = 1 (or 0)  =>  a'x + b in S
# ---------------------------------------------------------------------------
function MOI.add_constraint(
    optimizer::Optimizer,
    func::MOI.VectorAffineFunction{Float64},
    set::MOI.Indicator,
)
    return MOI.add_constraint(optimizer.model, func, set)
end

"""
    MOI.supports_constraint(
        optimizer::Optimizer,
        ::Type{MOI.VectorAffineFunction{Float64}},
        ::Type{S},
    ) where {S<:MOI.Indicator}

Return whether indicator constraints are supported. 
"""
function MOI.supports_constraint(
    optimizer::Optimizer,
    func::Type{MOI.VectorAffineFunction{Float64}},
    set::Type{S},
) where {S<:MOI.Indicator}
    return supports_constraint(optimizer.backend, func, set)
end


# ---------------------------------------------------------------------------
# Custom Sets
# ---------------------------------------------------------------------------
"""
    MOI.add_constraint(
        optimizer::Optimizer,
        func::MOI.VectorOfVariables,
        set::CustomSet,
    )

Add a `MOI.VectorOfVariables`-in-`CustomSet` constraint to `optimizer` by
storing it in the cached model.
"""
function MOI.add_constraint(optimizer::Optimizer, func::MOI.VectorOfVariables, set::CustomSet)
    return MOI.add_constraint(optimizer.model, func, set)
end


"""
    MOI.supports_constraint(
        ::Optimizer,
        ::Type{MOI.VectorOfVariables},
        ::Type{<:CustomSet},
    )

Return `true`: `Boscia.Optimizer` supports `MOI.VectorOfVariables`-in-`S`
constraints for every `S` in `CustomSet`. 
"""
function MOI.supports_constraint(::Optimizer, ::Type{MOI.VectorOfVariables}, ::Type{<:CustomSet})
    return true
end




# ---------------------------------------------------------------------------
# Feasible region
# ---------------------------------------------------------------------------

"""
    build_feasible_region(src::Optimizer)

Return a view of the cached model `src.model` that contains everything except
the objective: variables, variable bounds, integrality and linear constraints.
"""
function build_feasible_region(src::Optimizer)
    filtered_src = MOI.Utilities.ModelFilter(src.model) do item
        return !(item isa MOI.ObjectiveSense || item isa MOI.ObjectiveFunction)
    end
    return filtered_src
end
