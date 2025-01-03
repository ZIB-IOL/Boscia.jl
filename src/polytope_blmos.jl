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

    
# After splitting, split variable will be fixed to either 0 or 1.
function bounded_compute_inface_extreme_point(sblmo::CubeSimpleBLMO, direction, x, lb, ub, int_vars; kwargs...)
    n = length(x)
    v = copy(x)
    non_fixed_int = findall(lb .!= ub)
    non_fixed_int_idx = int_vars[non_fixed_int]

    idx = collect(1:n)
    non_int_idx = setdiff(idx, int_vars)
    non_fixed_idx = vcat(non_fixed_int_idx, non_int_idx)

     # For non_fixed coordinates, zero-sum means that they are all fixed to origin.
    sx = sum(x[non_fixed_idx])
    if sx <= 0
        return v
    end
    # Fix the point to the same face.
    # Zero will be return only if d_i is greater than zero.
    for idx in non_fixed_idx
        if x[idx] > 0 
            if x[idx] ≈ 1
                v[idx] = 1
            else
                v[idx] = direction[idx] >  0 ? 0 : 1
            end
        end
    end
    return v       
end

# For DICG each variable can only be either fixed to 0.0 or 1.0, or with normal bound [0.0, 1.0].
# Observe that the entries of direction corresponding to fixed variables will always be 0.0.
# In such case, we can use normal ZeroOneHypercube to compute gamma_max.
# Same as in the Probability and Unit Simplex.
function bounded_dicg_maximum_step(sblmo::CubeSimpleBLMO, x, direction, lb, ub, int_vars; kwargs...)
    return FrankWolfe.dicg_maximum_step(FrankWolfe.ZeroOneHypercube(), x, direction)
end

function dicg_split_vertices_set_simple(sblmo::CubeSimpleBLMO, x, vidx)
    x0_left = copy(x)
    x0_right = copy(x)
    x0_left[vidx] = floor(x[vidx])
    x0_right[vidx] = ceil(x[vidx])
    return x0_left, x0_right
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

function bounded_compute_inface_extreme_point(sblmo::ProbabilitySimplexSimpleBLMO, direction, x, lb, ub, int_vars; kwargs...)
    a = zeros(length(x))
    if sblmo.N in lb
        idx = findfirst(x->x==sblmo.N, lb)
        a[idx] = sblmo.N
        return a
    end
    min_val = Inf
    min_idx = -1
    for idx in eachindex(direction)
        val = direction[idx]
        if val < min_val && x[idx] > 0
            min_val = val
            min_idx = idx
        end
    end
    a[min_idx] = 1.0
    return a
end

function bounded_dicg_maximum_step(sblmo::ProbabilitySimplexSimpleBLMO, x, direction, lb, ub, int_vars; kwargs...)
    return FrankWolfe.dicg_maximum_step(FrankWolfe.ProbabilitySimplexOracle{Float64}(), x, direction)
end

function dicg_split_vertices_set_simple(sblmo::ProbabilitySimplexSimpleBLMO, x, vidx)
    n = length(x)
    x0_left = copy(x)
    sum_val = sum(x) - x[vidx]
    x0_left .+= (n-1) / sum_val
    x0_left[vidx] = floor(x[vidx])
    x0_right = zeros(length(x))
    x0_right[vidx] = 1.0
    return x0_left, x0_right
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

function bounded_compute_inface_extreme_point(sblmo::UnitSimplexSimpleBLMO, direction, x, lb, ub, int_vars; kwargs...)
    if sblmo.N in lb
        idx = findfirst(x->x==sblmo.N, lb)
        a = zeros(length(x))
        a[idx] = sblmo.N
        return a
    end
        
     # For non_fixed dimensions, zero-vector x means fixing to all coordinate faces, return zero-vector
    sx = sum(x)
    if sx <= 0
        return zeros(length(x))
    end
    
    min_val = Inf
    min_idx = -1

    for idx in eachindex(direction)
        val = direction[idx]
        if val < min_val && x[idx] > 0
            min_val = val
            min_idx = idx
        end
    end
        
    if sx ≉ sblmo.N && min_val > 0
        return zeros(length(x))
    end
    a = zeros(length(x))
    a[min_idx] = 1.0
    return a
end


function bounded_dicg_maximum_step(sblmo::UnitSimplexSimpleBLMO, x, direction, lb, ub, int_vars; kwargs...)
    return FrankWolfe.dicg_maximum_step(FrankWolfe.UnitSimplexOracle{Float64}(), x, direction)
end

function dicg_split_vertices_set_simple(sblmo::UnitSimplexSimpleBLMO, x, vidx)
    x0_left = copy(x)
    x0_left[vidx] = floor(x[vidx])
    x0_right = zeros(length(x))
    x0_right[vidx] = 1.0
    return x0_left, x0_right
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
