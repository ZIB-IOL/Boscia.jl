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
            v[i] = d[i] > 0 ? lb[idx] : ub[idx]
        else
            v[i] = d[i] > 0 ? sblmo.lower_bounds[i] : sblmo.upper_bounds[i]
        end
    end
    return v
end

function is_simple_linear_feasible(sblmo::CubeSimpleBLMO, v)
    for i in setdiff(eachindex(v), sblmo.int_vars)
        if !(sblmo.lower_bounds[i] ≤ v[i] + 1e-6) || !(v[i] - 1e-6 ≤ sblmo.upper_bounds[i])
            @debug(
                "Vertex entry: $(v[i]) Lower bound: $(blmo.bounds[i, :greaterthan]) Upper bound: $(blmo.bounds[i, :lessthan]))"
            )
            return false
        end
    end
    return true
end

function is_simple_inface_feasible(sblmo::CubeSimpleBLMO, a, x, lb, ub, int_vars; kwargs...)
	return is_simple_inface_feasible_subroutine(sblmo, a, x, lb, ub, int_vars; kwargs)
end

function is_decomposition_invariant_oracle_simple(sblmo::CubeSimpleBLMO)
    return true
end

"""
If the entry in x is at the boundary, choose the corresponding bound.
Otherwise, if the entry in direction is positve, choose the lower bound. Else, choose the upper bound.
"""
function bounded_compute_inface_extreme_point(sblmo::CubeSimpleBLMO, d, x, lb, ub, int_vars; atol = 1e-6, rtol = 1e-4, kwargs...)
    a = zeros(length(d))
    for i in eachindex(d)
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            if isapprox(x[i], ub[idx]; atol = atol, rtol = rtol)
                a[i] = ub[idx]
            elseif isapprox(x[i], lb[idx]; atol = atol, rtol = rtol)
                a[i] = lb[idx]
            else
                a[i] = d[i] > 0 ? lb[idx] : ub[idx]
            end
        else
            if isapprox(x[i], sblmo.upper_bounds[i]; atol = atol, rtol = rtol)
                a[i] = sblmo.upper_bounds[i]
            elseif isapprox(x[i], sblmo.lower_bounds[i]; atol = atol, rtol = rtol)
                a[i] = sblmo.lower_bounds[i]
            else
                a[i] = d[i] > 0 ? sblmo.lower_bounds[i] : sblmo.upper_bounds[i]
            end
        end
    end
    return a
end

"""
Compute the maximum step size for each entry and return the minium of all the possible step sizes.
"""
function bounded_dicg_maximum_step(sblmo::CubeSimpleBLMO, direction, x, lb, ub, int_vars; kwargs...)
    gamma_max = one(eltype(direction))
    for idx in eachindex(x)
        di = direction[idx]
        if idx in int_vars
            i = findfirst(x -> x == idx, int_vars)
            if di < 0
                gamma_max = min(gamma_max, (ub[i] - x[idx]) / -di)
            elseif di > 0
                gamma_max = min(gamma_max, (x[idx] - lb[i]) / di)
            end
        else
            if di < 0
                gamma_max = min(gamma_max, (sblmo.upper_bounds[idx] - x[idx]) / -di)
            elseif di > 0
                gamma_max = min(gamma_max, (x[idx] - sblmo.lower_bounds[idx]) / di)
            end
        end

    end
    return gamma_max
end

"""
    ProbablitySimplexSimpleBLMO(N)

Scaled Probability Simplex: ∑ x = N.
"""
struct ProbabilitySimplexSimpleBLMO <: SimpleBoundableLMO
    N::Float64
end

function is_decomposition_invariant_oracle_simple(sblmo::ProbabilitySimplexSimpleBLMO)
    return true  
end

function is_simple_inface_feasible(sblmo::ProbabilitySimplexSimpleBLMO, a, x, lb, ub, int_vars; kwargs...)
	return is_simple_inface_feasible_subroutine(sblmo, a, x, lb, ub, int_vars; kwargs)
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

"""
Fix the corresponding entries to the boudary based on the given x.
Assign the largest possible values to the unfixed entries corresponding to the smallest entries of d.
"""
function bounded_compute_inface_extreme_point(sblmo::ProbabilitySimplexSimpleBLMO, d, x, lb, ub, int_vars; atol = 1e-6, rtol = 1e-4, kwargs...)
    indices = collect(1:length(d))
    a = zeros(length(d))
    a[int_vars] = lb
    fixed_vars = []

    for i in indices
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            if isapprox(x[i], lb[idx]; atol = atol, rtol = rtol)
                push!(fixed_vars, i)
            elseif isapprox(x[i], ub[idx]; atol = atol, rtol = rtol)
                push!(fixed_vars, i)
                a[i] = ub[idx]
            end
        else
            if isapprox(x[i], 0.0; atol = atol, rtol = rtol)
                push!(fixed_vars, i)
            end
            if isapprox(x[i], 0.0; atol = atol, rtol = rtol)
                push!(fixed_vars, i)
                a[i] = sblmo.N
            end
        end
    end

    if sum(a) == sblmo.N
        return a
    end

    non_fixed_idx = setdiff(indices, fixed_vars)
    d_updated = d[non_fixed_idx]
    perm = sortperm(d_updated)
    sorted = non_fixed_idx[perm]

    for i in sorted
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            a[i] += min(ub[idx] - lb[idx], sblmo.N - sum(a))
        else
            a[i] += sblmo.N - sum(a)
        end
        if sum(a) == sblmo.N
            return a
        end
    end

    return a
end

"""
Compute the maximum step size for each entry and return the minium of all the possible step sizes.
"""
function bounded_dicg_maximum_step(sblmo::ProbabilitySimplexSimpleBLMO, direction, x, lb, ub, int_vars; tol = 1e-6, kwargs...)
    # the direction should never violate the simplex constraint because it would correspond to a gamma_max > 1
    gamma_max = one(eltype(direction))
    for idx in eachindex(x)
        di = direction[idx]
        if di > tol
            gamma_max = min(gamma_max, (x[idx] - lb[idx]) / di)
        elseif di < -tol
            gamma_max = min(gamma_max, (ub[idx] - x[idx]) / -di)
        end

        if gamma_max == 0.0
            return 0.0
        end
    end
    return gamma_max
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
    return true  
end

function is_simple_inface_feasible(sblmo::UnitSimplexSimpleBLMO, a, x, lb, ub, int_vars; kwargs...)
    if isapprox(sum(x), N; atol = atol, rtol = rtol) && !isapprox(sum(a), N; atol = atol, rtol = rtol)
        return false
    end
    return is_simple_inface_feasible_subroutine(sblmo, a, x, lb, ub, int_vars; kwargs)
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

"""
For boundary entries of x, assign the corresponding boudary.
For all positive entries of d, assign the corresponding lower bound.
For non-positive entries, assign largest possible value in increasing order.
"""
function bounded_compute_inface_extreme_point(sblmo::UnitSimplexSimpleBLMO, d, x, lb, ub, int_vars; atol = 1e-6, rtol = 1e-4, kwargs...)
    indices = collect(1:length(d))
    a = zeros(length(d))

    a[int_vars] = lb

    fixed_vars = []

    for i in indices
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            if isapprox(x[i], lb[idx]; atol = atol, rtol = rtol)
                push!(fixed_vars, i)
            elseif isapprox(x[i], ub[idx]; atol = atol, rtol = rtol)
                push!(fixed_vars, i)
                a[i] = ub[idx]
            end
        else
            if isapprox(x[i], 0.0; atol = atol, rtol = rtol)
                push!(fixed_vars, i)
            end
            if isapprox(x[i], 0.0; atol = atol, rtol = rtol)
                push!(fixed_vars, i)
                a[i] = sblmo.N
            end
        end
    end

    if sum(a) == sblmo.N
        return a
    end

    non_fixed_idx = setdiff(indices, fixed_vars)
    d_updated = d[non_fixed_idx]
    idx_neg = findall(x-> x <= 0, d_updated)
    perm = sortperm(d_updated[idx_neg])
    sorted_neg = idx_neg[perm]
    sorted = non_fixed_idx[sorted_neg]

    for i in sorted
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            a[i] += min(ub[idx] - lb[idx], sblmo.N - sum(a))
        else
            a[i] += sblmo.N - sum(a)
        end
        if sum(a) == sblmo.N
            return a
        end
    end

    return a
end

"""
Compute the maximum step size for each entry and the sum of entries should satisfy inequality constraint.
Return the minium of all the possible step sizes.
"""
function bounded_dicg_maximum_step(sblmo::UnitSimplexSimpleBLMO, direction, x, lb, ub, int_vars; tol = 1e-6, kwargs...)
    # the direction should never violate the simplex constraint because it would correspond to a gamma_max > 1.
    gamma_max = one(eltype(direction))
    for idx in eachindex(x)
        di = direction[idx]
        if di > tol
            gamma_max = min(gamma_max, (x[idx] - lb[idx]) / di)
        elseif di < -tol
            gamma_max = min(gamma_max, (ub[idx] - x[idx]) / -di)
        end
		
        if gamma_max == 0.0
            return 0.0
        end
    end

    # the sum of entries should be smaller than N.
    if sum(direction) < 0.0
        gamma_max = min(gamma_max, (sum(x) - sblmo.N) / sum(direction))
    end
	
    return gamma_max
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

function is_simple_inface_feasible_subroutine(sblmo::SimpleBoundableLMO, a, x, lb, ub, int_vars; atol = 1e-6, rtol = 1e-5, kwargs...)
    for i in eachindex(x)
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            if isapprox(x[idx], lb[idx]; atol = atol, rtol = rtol) && !isapprox(a[i], lb[idx]; atol = atol, rtol = rtol)
                return false
            elseif isapprox(x[idx], ub[idx]; atol = atol, rtol = rtol) && !isapprox(a[i], ub[idx]; atol = atol, rtol = rtol)
                return false
            end
        else
            if isapprox(x[i], sblmo.lower_bounds[i]; atol = atol, rtol = rtol) && !isapprox(a[i], sblmo.lower_bounds[i]; atol = atol, rtol = rtol)
                return false
            elseif isapprox(x[i], sblmo.upper_bounds[i]; atol = atol, rtol = rtol) && !isapprox(a[i], sblmo.upper_bounds[i]; atol = atol, rtol = rtol)
                return false
            end
        end
    end
    return true
end
