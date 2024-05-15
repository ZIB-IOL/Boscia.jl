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
        if !(sblmo.lower_bounds[i] ≤ v[i] + 1e-6) || !(v[i] - 1e-6 ≤ blmo.upper_bounds[i])
            @debug(
                "Vertex entry: $(v[i]) Lower bound: $(blmo.bounds[i, :greaterthan]) Upper bound: $(blmo.bounds[i, :lessthan]))"
            )
            return false
        end
    end
    return true
end

function is_decomposition_invariant_oracle_simple(sblmo::CubeSimpleBLMO)
    lb = sblmo.lower_bounds
    ub = sblmo.upper_bounds
    indicator = [0.0, 1.0]
    distinct_lbs = unique(lbs)
    distinct_ubs = unique(ubs)
    if !issubset(distinct_lbs, indicator) || !issubset(distinct_ubs, indicator)
        return false
    else
        retrun true
end

    
# After splitting, split variable will be fixed to either 0 or 1.
function bounded_compute_inface_extreme_point(sblmo::CubeSimpleBLMO, direction, x, lb, ub, int_vars; kwargs...)
    v = copy(x)
    non_fixed_idx = findall(lb .!= ub)
    non_fixed_int_idx = int_vars[non_fixed_idx]

     # For non_fixed coordinates, zero-sum means that they are all fixed to origin.
    sx = sum(x[non_fixed_int_idx])
    if sx <= 0
        return v
    end
    # Fix the point to the same face.
    # Zero will be return only if d_i is greater than zero.
    for idx in non_fixed_int_idx
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

function bounded_compute_inface_extreme_point(sblmo::ProbabilitySimplexSimpleBLMO, direction, x, lb, ub, int_vars; kwargs...)
    a = copy(x)
    if sblmo.N in lb
        idx = findfirst(x->x==sblmo.N, lb)
        a = zeros(length(x))
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
    return FrankWolfe.dicg_maximum_step(FrankWolfe.ProbabilitySimplexOracle(), x, direction)
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
