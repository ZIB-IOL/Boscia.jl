"""
    CubeSimpleBLMO{T}(lower_bounds, upper_bounds)

Hypercube with lower and upper bounds implementing the `SimpleBoundableLMO` interface.
"""
struct CubeSimpleBLMO <: SimpleBoundableLMO
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    int_vars::Vector{Int}
end

"""
If the entry is positve, choose the lower bound. Else, choose the upper bound.
"""
function bounded_compute_extreme_point(sblmo::CubeSimpleBLMO, d, lb, ub, int_vars; kwargs...)
    v = zeros(length(d))
    for i in eachindex(d)
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            v[idx] = d[idx] > 0 ? lb[idx] : ub[idx]
        else
            v[idx] = d[idx] > 0 ? sblmo.lower_bounds[idx] : sblmo.upper_bounds[idx]
        end
    end
    return v
end

function is_simple_linear_feasible(sblmo::CubeSimpleBLMO, v)
    for i in setdiff(eachindex(v), sblmo.int_vars)
        if !(sblmo.lower_bounds[i] ≤ v[i] + 1e-6 || !(v[i] - 1e-6 ≤ blmo.upper_bounds[i]))
            @debug(
                "Vertex entry: $(v[i]) Lower bound: $(blmo.bounds[i, :greaterthan]) Upper bound: $(blmo.bounds[i, :lessthan]))"
            )
            return false
        end
    end
    return true
end

#===============================================================================================================================#
"""
CubeSimpleBLMO
"""
function is_decomposition_invariant_oracle_simple(sblmo::CubeSimpleBLMO)
    lbs = sblmo.lower_bounds
    ubs = sblmo.upper_bounds
    indicator = [0.0, 1.0]
    distinct_lbs = unique(lbs)
    distinct_ubs = unique(ubs)
    if !issubset(distinct_lbs, indicator) 
        return false
    end
    if !issubset(distinct_ubs, indicator) 
        return false
    end
    return true
end

function bounded_compute_inface_extreme_point(sblmo::CubeSimpleBLMO, direction, x, lb, ub, int_vars; kwargs...)
    v = copy(x)
    non_fixed_idx = equal_bound_idx(lb, ub, 0)
    non_fixed_int_idx = int_vars[non_fixed_idx]

    for idx in eachindex(direction)
        if (idx in non_fixed_int_idx) && !(x[idx] == 0) 
            v[idx] = direction[idx] > 0 ? 0 : 1
        end
    end
    return v       
end

function bounded_dicg_maximum_step(sblmo::CubeSimpleBLMO, x, direction, lb, ub, int_vars; kwargs...)
    idx = collect(1: length(x))
    gamma_max = 1.0
    lb = sblmo.lower_bounds
    ub = sblmo.upper_bounds
    non_fixed_idx = equal_bound_idx(lb, ub, 1)
    non_fixed_int_idx = int_vars[non_fixed_idx]
    for idx in eachindex(direction)
        if idx in non_fixed_int_idx
            value = direction[idx]
            if (x[idx] === 0 && value > 0) || (x[idx] === 1 && value < 0)
                return 0.0
            end
            if value > 0
                gamma_max = min(gamma_max,  x[idx] / value)
            end
            if value < 0
                gamma_max = min(gamma_max, - (1-x[idx]) / value)
            end
        end
        return gamma_max
    end
end

function equal_bound_idx(lb, ub, sign)
    idx = lb .== ub
    return findall(x->x==sign, idx)
end

#===============================================================================================================================#
"""
    ProbablitySimplexSimpleBLMO(N)

Scaled Probability Simplex: ∑ x = N.
"""
struct ProbabilitySimplexSimpleBLMO <: SimpleBoundableLMO
    N::Float64
end

function is_decomposition_invariant_oracle_simple(sblmo::ProbabilitySimplexSimpleBLMO)
    if !(sblmo.N == 1)
        return false
    end
    return true  
end

"""
Assign the largest possible values to the entries corresponding to the smallest entries of d.
"""
function bounded_compute_extreme_point(sblmo::ProbabilitySimplexSimpleBLMO, d, lb, ub, int_vars; kwargs...)
    v = zeros(length(d))
    indices = collect(1:length(d))
    perm = sortperm(d)

    # The lower bounds always have to be met. 
    v[int_vars] = lb

    for i in indices[perm]
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            v[i] += min(ub[idx]-lb[idx], sblmo.N - sum(v))
        else
            v[i] += sblmo.N - sum(v)
        end
    end
    return v
end

function is_decomposition_invariant_oracle_simple(sblmo::ProbabilitySimplexSimpleBLMO)
    if !(sblmo.N == 1)
        return false
    end
    return true  
end

function bounded_dicg_maximum_step(sblmo::ProbabilitySimplexSimpleBLMO, x, direction, lb, ub, int_vars; kwargs...)
    a = copy(x)
    gamma_max = 1.0
    non_fixed_idx = equal_bound_idx(lb, ub, 0)
    non_fixed_int_idx = int_vars[non_fixed_idx]
    
    lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
    return FrankWolfe.dicg_maximum_step(lmo, x[non_fixed_int_idx], direction[non_fixed_int_idx],)
end

function is_simple_linear_feasible(sblmo::ProbabilitySimplexSimpleBLMO, v)
    if sum(v .≥ 0) < length(v)
        @debug "v has negative entries: $(v)"
        return false
    end
    return isapprox(sum(v), sblmo.N, atol=1e-4, rtol=1e-2)
end

function check_feasibility(sblmo::ProbabilitySimplexSimpleBLMO, lb, ub, int_vars, n)
    m = n - length(int_vars)
    if sum(lb) ≤ sblmo.N ≤ sum(ub) + m*sblmo.N
        return OPTIMAL
    else
        INFEASIBLE 
    end
end

"""
Hyperplane-aware rounding for the probability simplex.
"""
function rounding_hyperplane_heuristic(tree::Bonobo.BnBTree, tlmo::TimeTrackingLMO{ManagedBoundedLMO{ProbabilitySimplexSimpleBLMO}}, x) 
    z = copy(x)
    for idx in tree.branching_indices
        z[idx] = round(x[idx])
    end

    if count(!iszero, z[tree.branching_indices]) == 0
        return [z], false
    end

    N = tlmo.blmo.simple_lmo.N

    non_zero_int = intersect(findall(!iszero, z), tree.branching_indices)
    cont_z = isempty(setdiff(collect(1:tree.root.problem.nvars), tree.branching_indices)) ? 0 : sum(z[setdiff(collect(1:tree.root.problem.nvars), tree.branching_indices)])
    if cont_z + sum(tlmo.blmo.upper_bounds[non_zero_int]) < N || cont_z + sum(tlmo.blmo.lower_bounds[non_zero_int]) > N
        @debug "No heuristics improvement possible, bounds already reached, N=$(N), maximal possible sum $(cont_z + sum(tlmo.blmo.upperbounds[non_zero_int])), minimal possible sum $(cont_z + sum(tlmo.blmo.lower_bounds[non_zero_int]))"
        return [z], false
    end

    if sum(z) < N
        while sum(z) < N
            z = add_to_min(z, tlmo.blmo.upper_bounds, tree.branching_indices)
        end
    elseif sum(z) > N
        while sum(z) > N
            z = remove_from_max(z, tlmo.blmo.lower_bounds, tree.branching_indices)
        end
    end
    return [z], true
end
function add_to_min(x, ub, int_vars)
    perm = sortperm(x)
    j = findfirst(x->x != 0, x[perm])
    
    for i in intersect(perm[j:end], int_vars) 
        if x[i] < ub[i]
            x[i] += 1
            break
        else
            continue
        end
    end
    return x
end
function remove_from_max(x, lb, int_vars)
    perm = sortperm(x, rev = true)
    j = findlast(x->x != 0, x[perm])
    
    for i in intersect(perm[1:j], int_vars) 
       if x[i] > lb[i]
        x[i] -= 1
            break
        else
            continue
        end
    end
    return x
end

"""
    UnitSimplexSimpleBLMO(N)

Scaled Unit Simplex: ∑ x ≤ N.
"""
struct UnitSimplexSimpleBLMO <: SimpleBoundableLMO
    N::Float64
end

function is_decomposition_invariant_oracle_simple(sblmo::UnitSimplexSimpleBLMO)
    if !(sblmo.N == 1)
        return false
    end
    return true  
end

"""
For all positive entries of d, assign the corresponding lower bound.
For non-positive entries, assign largest possible value in increasing order.
"""
function bounded_compute_extreme_point(sblmo::UnitSimplexSimpleBLMO, d, lb, ub, int_vars; kwargs...)
    v = zeros(length(d))
    # The wloer bounds always have to be met.
    v[int_vars] = lb
    cont_vars = setdiff(collect(1:length(d)), int_vars)
    if !isempty(cont_vars)
        v[cont_vars] .= 0.0
    end
    
    idx_neg = findall(x-> x <= 0, d)
    perm = sortperm(d[idx_neg])
    for i in idx_neg[perm]
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            v[i] += min(ub[idx]-lb[idx], sblmo.N - sum(v))
        else
            v[i] += N - sum(v)
        end
    end
    return v
end

function bounded_compute_inface_extreme_point(sblmo::Union{ProbabilitySimplexSimpleBLMO, UnitSimplexSimpleBLMO}, 
                                                direction, x, lb, ub, int_vars; kwargs...)
    a = copy(x)
    idx = findfirst(x->x==1.0, lb)
    if !(idx==nothing)
        a = zeros(x)
        a[idx] = 1.0
        return a
    end
    non_fixed_idx = equal_bound_idx(lb, ub, 0)
    fixed_idx = equal_bound_idx(lb, ub, 1)
    non_fixed_int_idx = int_vars[non_fixed_idx]

    if typeof(sblmo) == ProbabilitySimplexSimpleBLMO
        lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
        a_lmo = FrankWolfe.compute_inface_extreme_point(lmo, direction[non_fixed_int_idx], x[non_fixed_int_idx],)
    else
        lmo = FrankWolfe.UnitSimplexOracle(1.0)
        scaled_hot_vec = FrankWolfe.compute_inface_extreme_point(lmo, direction[non_fixed_int_idx], x[non_fixed_int_idx],)
        active_val = scaled_hot_vec.active_val
        val_idx = scaled_hot_vec.val_idx
        a_lmo = zeros(scaled_hot_vec.len)
        a_lmo[val_idx] = active_val
    end
    
    for idx in eachindex(a)
        if idx in non_fixed_int_idx
            non_idx = findfirst(x -> x==idx, non_fixed_int_idx)
            if non_idx == val_idx
                a[idx] = active_val
            else
                a[idx] = 0.0
            end
        end
    end
    println("Inface vertex:")
    println(a)
    return a     
end

function bounded_dicg_maximum_step(sblmo::UnitSimplexSimpleBLMO, x, direction, lb, ub, int_vars; kwargs...)
    a = copy(x)
    gamma_max = 1.0
    non_fixed_idx = equal_bound_idx(lb, ub, 0)
    non_fixed_int_idx = int_vars[non_fixed_idx]
    
    lmo = FrankWolfe.UnitSimplexOracle(1.0)
    return FrankWolfe.dicg_maximum_step(lmo, x[non_fixed_int_idx], direction[non_fixed_int_idx],)
end

function is_simple_linear_feasible(sblmo::UnitSimplexSimpleBLMO, v)
    if sum(v .≥ 0) < length(v)
        @debug "v has negative entries: $(v)"
        return false
    end
    return sum(v) ≤ sblmo.N + 1e-3
end

function check_feasibility(sblmo::UnitSimplexSimpleBLMO, lb, ub, int_vars, n)
    if sum(lb) ≤ sblmo.N
        return OPTIMAL
    else
        INFEASIBLE 
    end
end

"""
Hyperplane-aware rounding for the unit simplex.
"""
function rounding_hyperplane_heuristic(tree::Bonobo.BnBTree, tlmo::TimeTrackingLMO{ManagedBoundedLMO{UnitSimplexSimpleBLMO}}, x) 
    z = copy(x)
    for idx in tree.branching_indices
        z[idx] = round(x[idx])
    end
    
    N = tlmo.blmo.simple_lmo.N

    non_zero_int = intersect(findall(!iszero, z), tree.branching_indices)
    cont_z = isempty(setdiff(collect(1:tree.root.problem.nvars), tree.branching_indices)) ? 0 : sum(z[setdiff(collect(1:tree.root.problem.nvars), tree.branching_indices)])
    if cont_z + sum(tlmo.blmo.lower_bounds[non_zero_int]) > N
        @debug "No heuristics improvement possible, bounds already reached, N=$(N), minimal possible sum $(cont_z + sum(tlmo.blmo.lower_bounds[non_zero_int]))"
        return [z], false
    end


    if sum(z) > N
        while sum(z) > N
            z = remove_from_max(z, tlmo.blmo.lower_bounds, tree.branching_indices)
        end
    end
    return [z], true
end
