"""
Constant to handle half open interval bounds on variables
"""
const inf_bound = 10.0^6

"""
    IntegerBounds

Keeps track of the bounds of the integer (binary) variables.

`lower_bounds` dictionary of the MOI.GreaterThan, index is the key.
`upper_bounds` dictionary of the MOI.LessThan, index is the key.
"""
mutable struct IntegerBounds #<: AbstractVector{Tuple{Int,MOI.LessThan{Float64}, MOI.GreaterThan{Float64}}}
    lower_bounds::Dict{Int,MOI.GreaterThan{Float64}}
    upper_bounds::Dict{Int,MOI.LessThan{Float64}}
end

IntegerBounds() =
    IntegerBounds(Dict{Int,MOI.GreaterThan{Float64}}(), Dict{Int,MOI.LessThan{Float64}}())

function Base.push!(ib::IntegerBounds, (idx, bound))
    if bound isa MOI.GreaterThan{Float64}
        ib.lower_bounds[idx] = bound
    elseif bound isa MOI.LessThan{Float64}
        ib.upper_bounds[idx] = bound
    end
    return ib
end

#Base.get(ib::GlobalIntegerBounds, i) = (ib.indices[i], ib.lessthan[i], ib.greaterthan[i])
#Base.size(ib::GlobalIntegerBounds) = size(ib.indices)

function Base.isempty(ib::IntegerBounds)
    return isempty(ib.lower_bounds) && isempty(ib.upper_bounds)
end

Base.copy(ib::IntegerBounds) = IntegerBounds(copy(ib.lower_bounds), copy(ib.upper_bounds))

# convenient call
# ib[3, :lessthan] or ib[3, :greaterthan]
function Base.getindex(ib::IntegerBounds, idx::Int, sense::Symbol)
    if sense == :lessthan
        ib.upper_bounds[idx]
    else
        ib.lower_bounds[idx]
    end
end

function Base.get(ib::IntegerBounds, (idx, sense), default)
    if sense == :lessthan
        get(ib.upper_bounds, idx, default)
    else
        get(ib.lower_bounds, idx, default)
    end
end

function Base.setindex!(ib::IntegerBounds, val, idx::Int, sense::Symbol)
    if sense == :lessthan
        ib.upper_bounds[idx] = val
    else
        ib.lower_bounds[idx] = val
    end
end

function Base.haskey(ib::IntegerBounds, (idx, sense))
    if sense == :lessthan
        return haskey(ib.upper_bounds, idx)
    else
        return haskey(ib.lower_bounds, idx)
    end
end

#=function find_bound(ib::GlobalIntegerBounds, vidx)
    @inbounds for idx in eachindex(ib)
        if ib.indices[idx] == vidx
            return idx
        end
    end
    return -1
end =#
