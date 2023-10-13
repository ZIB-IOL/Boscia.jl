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
        haskey(ib.upper_bounds, idx)
    else
        haskey(ib.lower_bounds, idx)
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

#=
"""
Build node LMO from global LMO

Four action can be taken:
KEEP   - constraint is as saved in the global bounds
CHANGE - lower/upper bound is changed to the node specific one
DELETE - custom bound from the previous node that is invalid at current node and has to be deleted
ADD    - bound has to be added for this node because it does not exist in the global bounds (e.g. variable bound is a half open interval globally) 
"""
function build_LMO(
    lmo::FrankWolfe.LinearMinimizationOracle,
    global_bounds::IntegerBounds,
    node_bounds::IntegerBounds,
    int_vars::Vector{Int},
)
    free_model(lmo.o)
    consLT_list =
        MOI.get(lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.LessThan{Float64}}())
    consGT_list =
        MOI.get(lmo.o, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.GreaterThan{Float64}}())
    cons_delete = []

    # Lower bounds
    for c_idx in consGT_list
        if c_idx.value in int_vars
            if haskey(global_bounds.lower_bounds, c_idx.value)
                # change 
                if haskey(node_bounds.lower_bounds, c_idx.value)
                    MOI.set(lmo.o, MOI.ConstraintSet(), c_idx, node_bounds.lower_bounds[c_idx.value])
                else
                    # keep
                    MOI.set(
                        lmo.o,
                        MOI.ConstraintSet(),
                        c_idx,
                        global_bounds.lower_bounds[c_idx.value],
                    )
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
                if haskey(node_bounds.upper_bounds, c_idx.value)
                    MOI.set(lmo.o, MOI.ConstraintSet(), c_idx, node_bounds.upper_bounds[c_idx.value])
                else
                    # keep
                    MOI.set(
                        lmo.o,
                        MOI.ConstraintSet(),
                        c_idx,
                        global_bounds.upper_bounds[c_idx.value],
                    )
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
    for key in keys(node_bounds.lower_bounds)
        if !haskey(global_bounds.lower_bounds, key)
            MOI.add_constraint(lmo.o, MOI.VariableIndex(key), node_bounds.lower_bounds[key])
        end
    end
    for key in keys(node_bounds.upper_bounds)
        if !haskey(global_bounds.upper_bounds, key)
            MOI.add_constraint(lmo.o, MOI.VariableIndex(key), node_bounds.upper_bounds[key])
        end
    end

    for list in (node_bounds.lower_bounds, node_bounds.upper_bounds)
        for (idx, set) in list
            c_idx =  MOI.ConstraintIndex{MOI.VariableIndex, typeof(set)}(idx)
            @assert MOI.is_valid(lmo.o, c_idx)
            set2 = MOI.get(lmo.o, MOI.ConstraintSet(), c_idx)
            if !(set == set2)
                MOI.set(lmo.o, MOI.ConstraintSet(), c_idx, set)
                set3 = MOI.get(lmo.o, MOI.ConstraintSet(), c_idx)
                @assert (set3 == set) "$((idx, set3, set))"
            end
        end
    end

end

build_LMO(lmo::TimeTrackingLMO, gb::IntegerBounds, nb::IntegerBounds, int_vars::Vector{Int64}) =
    build_LMO(lmo.lmo, gb, nb, int_vars)=#


    """
Build node LMO from global LMO

Four action can be taken:
KEEP   - constraint is as saved in the global bounds
CHANGE - lower/upper bound is changed to the node specific one
DELETE - custom bound from the previous node that is invalid at current node and has to be deleted
ADD    - bound has to be added for this node because it does not exist in the global bounds (e.g. variable bound is a half open interval globally) 
"""
function build_LMO(
    blmo::BoundedLinearMinimizationOracle,
    global_bounds::IntegerBounds,
    node_bounds::IntegerBounds,
    int_vars::Vector{Int},
)
    # free model data from previous nodes
    free_model(lmo.o)

    consLB_list = get_lower_bounds_list(blmo)
    consUB_list = get_upper_bounds_list(blmo)
    cons_delete = []

    # Lower bounds
    for c_idx in consLB_list
        v_idx = get_variable_index(blmo, c_idx)
        if is_constraint_on_int_var(c_idx, int_vars)
            if is_constraint_in(global_bounds.lower_bounds, c_idx)
                # change
                if is_constraint_in(node.lower_bounds, c_idx)
                    set_lower_bound(blmo, c_idx, node_bounds.lower_bounds[v_idx])
                # keep    
                else
                    set_lower_bound(blmo, c_idx, global_bounds.lower_bounds[v_idx])
                end
            else
                # delete
                push!(cons_delete, c_idx)
            end
        end
    end

    # Upper bounds
    for c_idx in consUB_list
        v_idx = get_variable_index(blmo, c_idx)
        if is_constraint_on_int_var(c_idx, int_vars)
            if is_constraint_in(global_bounds.uppers_bounds, c_idx)
                # change
                if is_constraint_in(node.uppers_bounds, c_idx)
                    set_upper_bound(blmo, c_idx, node_bounds.upper_bounds[v_idx])
                # keep    
                else
                    set_upper_bound(blmo, c_idx, global_bounds.upper_bounds[v_idx])
                end
            else
                # delete
                push!(cons_delete, c_idx)
            end
        end
    end

    # delete constraints
    delete_constraints!(blmo, cons_delete)

    # add node specific constraints
    for key in keys(node_bounds.lower_bounds)
        if !is_constraint_in(global_bounds.lower_bounds, key)
            add_constraint!(blmo, key, node_bounds.lower_bounds[key])
        end
    end
    for key in keys(node_bounds.upper_bounds)
        if !is_constraint_in(global_bounds.upper_bounds, key)
            add_constraint!(blmo, key, node_bounds.upper_bounds[key])
        end
    end

    @assert check_bounds(lmo, node_bounds)
end

build_LMO(tt_lmo::TimeTrackingLMO, gb::IntegerBounds, nb::IntegerBounds, int_vars::Vector{Int64}) =
    build_LMO(tt_lmo.blmo, gb, nb, int_vars)
