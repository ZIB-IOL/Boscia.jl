
"""
    IntegerBounds

Keeps track of the bounds of the integer (binary) variables.

`lower_bounds` dictionary of Float64, index is the key.
`upper_bounds` dictionary of Float64, index is the key.
"""
mutable struct IntegerBounds
    lower_bounds::Dict{Int, Float64}
    upper_bounds::Dict{Int, Float64}
end

IntegerBounds() =
    IntegerBounds(Dict{Int, Float64}(), Dict{Int, Float64}())

function Base.push!(ib::IntegerBounds, (idx, bound), sense::Symbol)
    if sense == :greaterthan
        ib.lower_bounds[idx] = bound
    elseif sense == :lessthan
        ib.upper_bounds[idx] = bound
    else
        error("Allowed values for sense are :lessthan and :greaterthan.")
    end
    return ib
end

function Base.isempty(ib::IntegerBounds)
    return isempty(ib.lower_bounds) && isempty(ib.upper_bounds)
end

Base.copy(ib::IntegerBounds) = IntegerBounds(copy(ib.lower_bounds), copy(ib.upper_bounds))

# convenient call
# ib[3, :lessthan] or ib[3, :greaterthan]
function Base.getindex(ib::IntegerBounds, idx::Int, sense::Symbol)
    if sense == :lessthan
        ib.upper_bounds[idx]
    elseif sense == :greaterthan
        ib.lower_bounds[idx]
    else
        error("Allowed values for sense are :lessthan and :greaterthan.")
    end
end

function Base.get(ib::IntegerBounds, (idx, sense), default)
    if sense == :lessthan
        get(ib.upper_bounds, idx, default)
    elseif sense == :greaterthan
        get(ib.lower_bounds, idx, default)
    else
        error("Allowed values for sense are :lessthan and :greaterthan.")
    end
end

function Base.setindex!(ib::IntegerBounds, val, idx::Int, sense::Symbol)
    if sense == :lessthan
        ib.upper_bounds[idx] = val
    elseif sense == :greaterthan
        ib.lower_bounds[idx] = val
    else
        error("Allowed values for sense are :lessthan and :greaterthan.")
    end
end

function Base.haskey(ib::IntegerBounds, (idx, sense))
    if sense == :lessthan
        return haskey(ib.upper_bounds, idx)
    elseif sense == :greaterthan
        return haskey(ib.lower_bounds, idx)
    else
        error("Allowed values for sense are :lessthan and :greaterthan.")
    end
end
