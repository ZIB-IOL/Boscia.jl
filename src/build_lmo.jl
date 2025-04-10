
"""
Build node LMO from global LMO

Four action can be taken:
- KEEP   constraint is as saved in the global bounds
- CHANGE lower/upper bound is changed to the node specific one
- DELETE custom bound from the previous node that is invalid at current node and has to be deleted
- ADD    bound has to be added for this node because it does not exist in the global bounds (e.g. variable bound is a half open interval globally) 
"""
function build_LMO(
    blmo::BoundedLinearMinimizationOracle,
    global_bounds::IntegerBounds,
    node_bounds::IntegerBounds,
    int_vars::Vector{Int},
)
    free_model(blmo)

    consLB_list = get_lower_bound_list(blmo)
    consUB_list = get_upper_bound_list(blmo)
    cons_delete = []

    # Lower bounds
    for c_idx in consLB_list
        if is_constraint_on_int_var(blmo, c_idx, int_vars)
            v_idx = get_int_var(blmo, c_idx)
            if is_bound_in(blmo, c_idx, global_bounds.lower_bounds)
                # Change
                if is_bound_in(blmo, c_idx, node_bounds.lower_bounds)
                    set_bound!(blmo, c_idx, node_bounds.lower_bounds[v_idx], :greaterthan)
                    # Keep
                else
                    set_bound!(blmo, c_idx, global_bounds.lower_bounds[v_idx], :greaterthan)
                end
            else
                # Delete
                push!(cons_delete, (c_idx, :greaterthan))
            end
        end
    end

    # Upper bounds
    for c_idx in consUB_list
        if is_constraint_on_int_var(blmo, c_idx, int_vars)
            v_idx = get_int_var(blmo, c_idx)
            if is_bound_in(blmo, c_idx, global_bounds.upper_bounds)
                # Change
                if is_bound_in(blmo, c_idx, node_bounds.upper_bounds)
                    set_bound!(blmo, c_idx, node_bounds.upper_bounds[v_idx], :lessthan)
                    # Keep
                else
                    set_bound!(blmo, c_idx, global_bounds.upper_bounds[v_idx], :lessthan)
                end
            else
                # Delete
                push!(cons_delete, (c_idx, :lessthan))
            end
        end
    end

    # delete constraints
    delete_bounds!(blmo, cons_delete)

    # add node specific constraints 
    # These are bounds constraints where there is no corresponding global bound
    for key in keys(node_bounds.lower_bounds)
        if !haskey(global_bounds.lower_bounds, key)
            add_bound_constraint!(blmo, key, node_bounds.lower_bounds[key], :greaterthan)
        end
    end
    for key in keys(node_bounds.upper_bounds)
        if !haskey(global_bounds.upper_bounds, key)
            add_bound_constraint!(blmo, key, node_bounds.upper_bounds[key], :lessthan)
        end
    end

    return build_LMO_correct(blmo, node_bounds)
end

build_LMO(tlmo::TimeTrackingLMO, gb::IntegerBounds, nb::IntegerBounds, int_vars::Vector{Int64}) =
    build_LMO(tlmo.blmo, gb, nb, int_vars)
