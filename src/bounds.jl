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
    lower_bounds::Dict{Int, MOI.GreaterThan{Float64}}
    upper_bounds::Dict{Int, MOI.LessThan{Float64}}
end

IntegerBounds() = IntegerBounds(Dict{Int, MOI.GreaterThan{Float64}}(), Dict{Int, MOI.LessThan{Float64}}())

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

#=function find_bound(ib::GlobalIntegerBounds, vidx)
    @inbounds for idx in eachindex(ib)
        if ib.indices[idx] == vidx
            return idx
        end
    end
    return -1
end =#

"""
Build node LMO from global LMO

Four action can be taken:
KEEP   - constraint is as saved in the global bounds
CHANGE - lower/upper bound is changed to the node specific one
DELETE - custom bound from the previous node that is invalid at current node and has to be deleted
ADD    - bound has to be added for this node because it does not exist in the global bounds (e.g. variable bound is a half open interval globally) 
"""
function build_LMO(lmo::FrankWolfe.LinearMinimizationOracle, global_bounds::IntegerBounds, nodeBounds::IntegerBounds, int_vars::Vector{Int})
    consLT_list = MOI.get(lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.LessThan{Float64}}()) 
    consGT_list = MOI.get(lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.GreaterThan{Float64}}()) 
    cons_delete = []

    # Lower bounds
    for c_idx in consGT_list
        if c_idx.value in int_vars
            if haskey(global_bounds.lower_bounds, c_idx.value)
                # change 
                 if haskey(nodeBounds.lower_bounds, c_idx.value)
                     if c_idx.value == 5
                        @debug "Found key variable $(nodeBounds.lower_bounds[c_idx.value])"
                     end
                    MOI.set(lmo.o, MOI.ConstraintSet(), c_idx, nodeBounds.lower_bounds[c_idx.value])
                else
                # keep
                    MOI.set(lmo.o, MOI.ConstraintSet(), c_idx, global_bounds.lower_bounds[c_idx.value])
                end
            else
                # delete
                push!(cons_delete, c_idx)
            end
        end
    end

    # Upper bounds
    for c_idx in consLT_list
        if c_idx.value in int_vars
            if haskey(global_bounds.upper_bounds, c_idx.value)
                # change 
                 if haskey(nodeBounds.upper_bounds, c_idx.value) 
                    MOI.set(lmo.o, MOI.ConstraintSet(), c_idx, nodeBounds.upper_bounds[c_idx.value]) 
                else
                # keep
                    MOI.set(lmo.o, MOI.ConstraintSet(), c_idx, global_bounds.upper_bounds[c_idx.value])
                end
            else
                # delete
                push!(cons_delete, c_idx)
            end
        end
    end

    # delete constraints
    for d_idx in cons_delete
        MOI.delete(lmo.o, d_idx)
    end

    # add node specific constraints
    for key in keys(nodeBounds.lower_bounds)
        if !haskey(global_bounds.lower_bounds, key)
            MOI.add_constraint(lmo.o, MOI.VariableIndex(key), nodeBounds.lower_bounds[key])
        end
    end
    for key in keys(nodeBounds.upper_bounds)
        if !haskey(global_bounds.upper_bounds, key)
            MOI.add_constraint(lmo.o, MOI.VariableIndex(key), nodeBounds.upper_bounds[key])
        end
    end

    #print(lmo.o)

end

build_LMO(lmo::TimeTrackingLMO, gb::IntegerBounds, nb::IntegerBounds, int_vars::Vector{Int64}) = build_LMO(lmo.lmo, gb, nb, int_vars)
