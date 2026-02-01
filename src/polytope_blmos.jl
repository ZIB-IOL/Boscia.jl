"""
    CubeSimpleLMO{T}(lower_bounds, upper_bounds, int_vars)

Hypercube with lower and upper bounds implementing the `SimpleBoundableLMO` interface.
"""
struct CubeLMO <: FrankWolfe.LinearMinimizationOracle
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    int_vars::Vector{Int}
end

const CubeSimpleBLMO = CubeLMO

"""
     bounded_compute_extreme_point(sblmo::CubeSimpleLMO, d, lb, ub, int_vars; kwargs...)

If the entry is positve, choose the lower bound. Else, choose the upper bound.
"""
function bounded_compute_extreme_point(lmo::CubeLMO, d, lb, ub, int_vars; kwargs...)
    v = zeros(length(d))
    for i in eachindex(d)
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            v[i] = d[i] > 0 ? lb[idx] : ub[idx]
        else
            v[i] = d[i] > 0 ? lmo.lower_bounds[i] : lmo.upper_bounds[i]
        end
    end
    return v
end

function is_simple_linear_feasible(lmo::CubeLMO, v)
    for i in setdiff(eachindex(v), lmo.int_vars)
        if !(lmo.lower_bounds[i] ≤ v[i] + 1e-6 || !(v[i] - 1e-6 ≤ lmo.upper_bounds[i]))
            @debug(
                "Vertex entry: $(v[i]) Lower bound: $(lmo.bounds[i, :greaterthan]) Upper bound: $(lmo.bounds[i, :lessthan]))"
            )
            return false
        end
    end
    return true
end

function is_simple_inface_feasible(lmo::CubeLMO, a, x, lb, ub, int_vars; kwargs...)
    return is_simple_inface_feasible_subroutine(lmo, a, x, lb, ub, int_vars; kwargs)
end

function is_decomposition_invariant_oracle_simple(lmo::CubeLMO)
    return true
end

"""
If the entry in x is at the boundary, choose the corresponding bound.
Otherwise, if the entry in direction is positve, choose the lower bound. Else, choose the upper bound.
"""
function bounded_compute_inface_extreme_point(
    lmo::CubeLMO,
    d,
    x,
    lb,
    ub,
    int_vars;
    atol=1e-6,
    rtol=1e-4,
    kwargs...,
)
    a = zeros(length(d))
    for i in eachindex(d)
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            if isapprox(x[i], ub[idx]; atol=atol, rtol=rtol)
                a[i] = ub[idx]
            elseif isapprox(x[i], lb[idx]; atol=atol, rtol=rtol)
                a[i] = lb[idx]
            else
                a[i] = d[i] > 0 ? lb[idx] : ub[idx]
            end
        else
            if isapprox(x[i], lmo.upper_bounds[i]; atol=atol, rtol=rtol)
                a[i] = lmo.upper_bounds[i]
            elseif isapprox(x[i], lmo.lower_bounds[i]; atol=atol, rtol=rtol)
                a[i] = lmo.lower_bounds[i]
            else
                a[i] = d[i] > 0 ? lmo.lower_bounds[i] : lmo.upper_bounds[i]
            end
        end
    end
    return a
end

"""
Compute the maximum step size for each entry and return the minium of all the possible step sizes.
"""
function bounded_dicg_maximum_step(lmo::CubeLMO, direction, x, lb, ub, int_vars; kwargs...)
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
                gamma_max = min(gamma_max, (lmo.upper_bounds[idx] - x[idx]) / -di)
            elseif di > 0
                gamma_max = min(gamma_max, (x[idx] - lmo.lower_bounds[idx]) / di)
            end
        end

    end
    return gamma_max
end

"""
    ProbablitySimplexSimpleLMO(N)

The scaled probability simplex with `∑ x = N`.
"""
struct ProbabilitySimplexLMO <: FrankWolfe.LinearMinimizationOracle
    N::Float64
end

const ProbabilitySimplexSimpleBLMO = ProbabilitySimplexLMO

function is_decomposition_invariant_oracle_simple(lmo::ProbabilitySimplexLMO)
    return true
end

function is_simple_inface_feasible(lmo::ProbabilitySimplexLMO, a, x, lb, ub, int_vars; kwargs...)
    return is_simple_inface_feasible_subroutine(lmo, a, x, lb, ub, int_vars; kwargs)
end

"""
    bounded_compute_extreme_point(lmo::ProbabilitySimplexLMO, d, lb, ub, int_vars; kwargs...)

Assign the largest possible values to the entries corresponding to the smallest entries of d.
"""
function bounded_compute_extreme_point(lmo::ProbabilitySimplexLMO, d, lb, ub, int_vars; kwargs...)
    v = zeros(length(d))
    indices = collect(1:length(d))
    perm = sortperm(d)

    # Step 1: satisfy integer lower bounds
    v[int_vars] = lb

    # Step 2: distribute remaining N
    for i in indices[perm]
        rem = lmo.N - sum(v)
        if rem ≤ 1e-10
            break
        end

        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            add_int = min(ub[idx] - v[i], floor(rem))  # make sure it is int
            v[i] += add_int
        else
            v[i] += rem
        end
    end

    return v
end


"""
Fix the corresponding entries to the boudary based on the given x.
Assign the largest possible values to the unfixed entries corresponding to the smallest entries of d.
"""
function bounded_compute_inface_extreme_point(
    lmo::ProbabilitySimplexLMO,
    d,
    x,
    lb,
    ub,
    int_vars;
    atol=1e-6,
    rtol=1e-4,
    kwargs...,
)
    indices = collect(1:length(d))
    a = zeros(length(d))
    a[int_vars] = lb
    fixed_vars = []

    for i in indices
        if i in int_vars
            idx = findfirst(==(i), int_vars)
            if isapprox(x[i], lb[idx]; atol=atol, rtol=rtol)
                push!(fixed_vars, i)
                a[i] = lb[idx]
            elseif isapprox(x[i], ub[idx]; atol=atol, rtol=rtol)
                push!(fixed_vars, i)
                a[i] = ub[idx]
            end
        else
            if isapprox(x[i], 0.0; atol=atol, rtol=rtol)
                push!(fixed_vars, i)
                a[i] = 0.0
            end
        end
    end

    if isapprox(sum(a), lmo.N; atol=atol, rtol=rtol)
        return a
    end

    non_fixed_idx = setdiff(indices, fixed_vars)
    d_updated = d[non_fixed_idx]
    perm = sortperm(d_updated)
    sorted = non_fixed_idx[perm]
    rem = lmo.N - sum(a)


    for i in sorted
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            add_int = min(ub[idx] - a[i], floor(rem))
            a[i] += add_int
        else
            a[i] += rem
        end
        rem = lmo.N - sum(a)
        if isapprox(sum(a), lmo.N; atol=atol, rtol=rtol)
            return a
        end
    end

    return a
end

"""
Compute the maximum step size for each entry and return the minium of all the possible step sizes.
"""
function bounded_dicg_maximum_step(
    lmo::ProbabilitySimplexLMO,
    direction,
    x,
    lb,
    ub,
    int_vars;
    tol=1e-6,
    kwargs...,
)
    # the direction should never violate the simplex constraint because it would correspond to a gamma_max > 1
    gamma_max = one(eltype(direction))
    for idx in eachindex(x)
        di = direction[idx]
        if di > tol
            if idx in int_vars
                int_idx = findfirst(==(idx), int_vars)
                gamma_max = min(gamma_max, (x[idx] - lb[int_idx]) / di)
            else
                gamma_max = min(gamma_max, (x[idx] - 0.0) / di)
            end
        elseif di < -tol
            if idx in int_vars
                int_idx = findfirst(==(idx), int_vars)
                gamma_max = min(gamma_max, (ub[int_idx] - x[idx]) / -di)
            else
                gamma_max = min(gamma_max, (lmo.N - x[idx]) / -di)
            end
        end

        if isapprox(gamma_max, 0.0; atol=tol)
            return 0.0
        end
    end
    return gamma_max
end

function is_simple_linear_feasible(lmo::ProbabilitySimplexLMO, v)
    if any(v .< -1e-8)
        @debug "v has negative entries: $(v)"
        return false
    end
    return isapprox(sum(v), lmo.N, atol=1e-4, rtol=1e-2)
end

function check_feasibility(lmo::ProbabilitySimplexLMO, lb, ub, int_vars, n)
    m = n - length(int_vars)
    if length(int_vars) == n && !isinteger(lmo.N)
        error("Invalid problem: all variables are integer but N is non-integer.")
    end
    if sum(lb) ≤ lmo.N ≤ sum(ub) + m * lmo.N
        return OPTIMAL
    else
        return INFEASIBLE
    end
end

"""
     rounding_hyperplane_heuristic(tree::Bonobo.BnBTree, tlmo::TimeTrackingLMO{ManagedBoundedLMO{ProbabilitySimplexLMO}}, x) 

Hyperplane-aware rounding for the probability simplex.
"""
function rounding_hyperplane_heuristic(
    tree::Bonobo.BnBTree,
    tlmo::TimeTrackingLMO{ManagedBoundedLMO{ProbabilitySimplexLMO}},
    x,
)
    z = copy(x)
    for idx in tree.branching_indices
        z[idx] = round(x[idx])
    end

    if count(!iszero, z[tree.branching_indices]) == 0
        return [z], false
    end

    N = tlmo.lmo.lmo.N

    non_zero_int = intersect(findall(!iszero, z), tree.branching_indices)
    cont_z =
        isempty(setdiff(collect(1:tree.root.problem.nvars), tree.branching_indices)) ? 0 :
        sum(z[setdiff(collect(1:tree.root.problem.nvars), tree.branching_indices)])
    if cont_z + sum(tlmo.lmo.upper_bounds[non_zero_int]) < N ||
       cont_z + sum(tlmo.lmo.lower_bounds[non_zero_int]) > N
        @debug "No heuristics improvement possible, bounds already reached, N=$(N), maximal possible sum $(cont_z + sum(tlmo.lmo.upperbounds[non_zero_int])), minimal possible sum $(cont_z + sum(tlmo.lmo.lower_bounds[non_zero_int]))"
        return [z], true
    end

    if sum(z) < N
        while sum(z) < N
            z = add_to_min(z, tlmo.lmo.upper_bounds, tree.branching_indices)
        end
    elseif sum(z) > N
        while sum(z) > N
            z = remove_from_max(z, tlmo.lmo.lower_bounds, tree.branching_indices)
        end
    end
    return [z], false
end
function add_to_min(x, ub, int_vars)
    perm = sortperm(x)
    j = findfirst(x -> x != 0, x[perm])

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
    perm = sortperm(x, rev=true)
    j = findlast(x -> x != 0, x[perm])

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
    UnitSimplexLMO(N)

The scaled unit simplex with `∑ x ≤ N`.
"""
struct UnitSimplexLMO <: FrankWolfe.LinearMinimizationOracle
    N::Float64
end

const UnitSimplexSimpleBLMO = UnitSimplexLMO

function is_decomposition_invariant_oracle_simple(lmo::UnitSimplexLMO)
    return true
end

function is_simple_inface_feasible(lmo::UnitSimplexLMO, a, x, lb, ub, int_vars; kwargs...)
    if isapprox(sum(x), lmo.N; atol=atol, rtol=rtol) &&
       !isapprox(sum(a), lmo.N; atol=atol, rtol=rtol)
        return false
    end
    return is_simple_inface_feasible_subroutine(lmo, a, x, lb, ub, int_vars; kwargs)
end

"""
    bounded_compute_extreme_point(lmo::UnitSimplexSimpleLMO, d, lb, ub, int_vars; kwargs...)

For all positive entries of d, assign the corresponding lower bound.
For non-positive entries, assign largest possible value in increasing order.
"""
function bounded_compute_extreme_point(lmo::UnitSimplexLMO, d, lb, ub, int_vars; kwargs...)
    v = zeros(length(d))
    # The wloer bounds always have to be met.
    v[int_vars] = lb
    cont_vars = setdiff(collect(1:length(d)), int_vars)
    if !isempty(cont_vars)
        v[cont_vars] .= 0.0
    end

    idx_neg = findall(x -> x <= 0, d)
    perm = sortperm(d[idx_neg])
    for i in idx_neg[perm]
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            v[i] += min(ub[idx] - lb[idx], lmo.N - sum(v))
        else
            v[i] += lmo.N - sum(v)
        end
    end
    return v
end



"""
For boundary entries of x, assign the corresponding boudary.
For all positive entries of d, assign the corresponding lower bound.
For non-positive entries, assign largest possible value in increasing order.
"""
function bounded_compute_inface_extreme_point(
    lmo::UnitSimplexLMO,
    d,
    x,
    lb,
    ub,
    int_vars;
    atol=1e-6,
    rtol=1e-4,
    kwargs...,
)
    indices = collect(1:length(d))
    a = zeros(length(d))

    a[int_vars] = lb

    fixed_vars = []

    for i in indices
        if i in int_vars
            idx = findfirst(==(i), int_vars)
            if isapprox(x[i], lb[idx]; atol=atol, rtol=rtol)
                push!(fixed_vars, i)
                a[i] = lb[idx]
            elseif isapprox(x[i], ub[idx]; atol=atol, rtol=rtol)
                push!(fixed_vars, i)
                a[i] = ub[idx]
            end
        else
            if isapprox(x[i], 0.0; atol=atol, rtol=rtol)
                push!(fixed_vars, i)
                a[i] = 0.0
            end
        end
    end

    if isapprox(sum(a), lmo.N; atol=atol, rtol=rtol)
        return a
    end

    non_fixed_idx = setdiff(indices, fixed_vars)
    d_updated = d[non_fixed_idx]
    idx_neg = findall(x -> x <= 0, d_updated)
    perm = sortperm(d_updated[idx_neg])
    sorted_neg = idx_neg[perm]
    sorted = non_fixed_idx[sorted_neg]
    rem = lmo.N - sum(a)


    for i in sorted
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            add_int = min(ub[idx] - a[i], floor(rem))
            a[i] += add_int
        else
            a[i] += rem
        end
        rem = lmo.N - sum(a)
        if isapprox(sum(a), lmo.N; atol=atol, rtol=rtol)
            return a
        end
    end

    return a
end

"""
Compute the maximum step size for each entry and the sum of entries should satisfy inequality constraint.
Return the minium of all the possible step sizes.
"""
function bounded_dicg_maximum_step(
    lmo::UnitSimplexLMO,
    direction,
    x,
    lb,
    ub,
    int_vars;
    tol=1e-6,
    kwargs...,
)
    # the direction should never violate the simplex constraint because it would correspond to a gamma_max > 1.
    gamma_max = one(eltype(direction))
    for idx in eachindex(x)
        di = direction[idx]
        if di > tol
            if idx in int_vars
                int_idx = findfirst(==(idx), int_vars)
                gamma_max = min(gamma_max, (x[idx] - lb[int_idx]) / di)
            else
                gamma_max = min(gamma_max, (x[idx] - 0.0) / di)
            end
        elseif di < -tol
            if idx in int_vars
                int_idx = findfirst(==(idx), int_vars)
                gamma_max = min(gamma_max, (ub[int_idx] - x[idx]) / -di)
            else
                gamma_max = min(gamma_max, (lmo.N - x[idx]) / -di)
            end
        end

        if isapprox(gamma_max, 0.0; atol=tol)
            return 0.0
        end
    end

    # the sum of entries should be smaller than N.
    if sum(direction) < 0.0
        gamma_max = min(gamma_max, (sum(x) - lmo.N) / sum(direction))
    end

    return gamma_max
end

function is_simple_linear_feasible(lmo::UnitSimplexLMO, v)
    if sum(v .≥ 0) < length(v)
        @debug "v has negative entries: $(v)"
        return false
    end
    return sum(v) ≤ lmo.N + 1e-3
end

function check_feasibility(lmo::UnitSimplexLMO, lb, ub, int_vars, n)
    if sum(lb) ≤ lmo.N
        return OPTIMAL
    else
        INFEASIBLE
    end
end

"""
    rounding_hyperplane_heuristic(tree::Bonobo.BnBTree, tlmo::TimeTrackingLMO{ManagedBoundedLMO{UnitSimplexLMO}}, x) 
    
Hyperplane-aware rounding for the unit simplex.
"""
function rounding_hyperplane_heuristic(
    tree::Bonobo.BnBTree,
    tlmo::TimeTrackingLMO{ManagedBoundedLMO{UnitSimplexLMO}},
    x,
)
    z = copy(x)
    for idx in tree.branching_indices
        z[idx] = round(x[idx])
    end

    N = tlmo.lmo.lmo.N

    non_zero_int = intersect(findall(!iszero, z), tree.branching_indices)
    cont_z =
        isempty(setdiff(collect(1:tree.root.problem.nvars), tree.branching_indices)) ? 0 :
        sum(z[setdiff(collect(1:tree.root.problem.nvars), tree.branching_indices)])
    if cont_z + sum(tlmo.lmo.lower_bounds[non_zero_int]) > N
        @debug "No heuristics improvement possible, bounds already reached, N=$(N), minimal possible sum $(cont_z + sum(tlmo.lmo.lower_bounds[non_zero_int]))"
        return [z], true
    end


    if sum(z) > N
        while sum(z) > N
            z = remove_from_max(z, tlmo.lmo.lower_bounds, tree.branching_indices)
        end
    end
    return [z], false
end

function is_simple_inface_feasible_subroutine(
    lmo::FrankWolfe.LinearMinimizationOracle,
    a,
    x,
    lb,
    ub,
    int_vars;
    atol=1e-6,
    rtol=1e-5,
    kwargs...,
)
    for i in eachindex(x)
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            if isapprox(x[idx], lb[idx]; atol=atol, rtol=rtol) &&
               !isapprox(a[i], lb[idx]; atol=atol, rtol=rtol)
                return false
            elseif isapprox(x[idx], ub[idx]; atol=atol, rtol=rtol) &&
                   !isapprox(a[i], ub[idx]; atol=atol, rtol=rtol)
                return false
            end
        else
            if isapprox(x[i], lmo.lower_bounds[i]; atol=atol, rtol=rtol) &&
               !isapprox(a[i], lmo.lower_bounds[i]; atol=atol, rtol=rtol)
                return false
            elseif isapprox(x[i], lmo.upper_bounds[i]; atol=atol, rtol=rtol) &&
                   !isapprox(a[i], lmo.upper_bounds[i]; atol=atol, rtol=rtol)
                return false
            end
        end
    end
    return true
end

"""
    ReverseKnapsackLMO(N, upper_bounds)

BLMO denotes the reverse Knapsack constraint: ∑ x ≥ N.
We assume x ≥ 0. 
Explicit upper bounds are needed, otherwise the feasible region is unbounded.
"""
struct ReverseKnapsackLMO <: FrankWolfe.LinearMinimizationOracle
    N::Float64
    upper_bounds::Vector{Float64}
end

const ReverseKnapsackBLMO = ReverseKnapsackLMO

# Have the same upper bounds for all variables
function ReverseKnapsackLMO(size; N=1.0, upper=1.0)
    return ReverseKnapsackLMO(N, fill(upper, size))
end

"""
Entries corresponding to non positive entries in d, are assigned their upper bound.
"""
function bounded_compute_extreme_point(lmo::ReverseKnapsackLMO, d, lb, ub, int_vars; kwargs...)
    v = copy(lmo.upper_bounds)
    v[int_vars] = min.(v[int_vars], ub)

    idx_pos = findall(x -> x > 0, d)
    #v[idx_pos] = 0.0
    #v[intersect(idx_pos, int_vars)] = lb

    perm = sortperm(d[idx_pos], rev=true)
    for i in idx_pos[perm]
        if i in int_vars
            v[i] += max(lmo.N - sum(v), lb[i] - ub[i], -v[i])
        else
            v[i] += max(N - sum(v), -v[i])
        end
    end
    return v
end

function is_simple_linear_feasible(lmo::ReverseKnapsackLMO, v)
    if sum(v .≥ 0) < length(v)
        @debug "v has negative entries: $(v)"
        return false
    end
    if sum(v .<= lmo.upper_bounds) < length(v)
        @debug begin
            idxs = findall(x -> x == 0, v .<= lmo.upper_bounds)
            @info "vertex violates the upper bounds at indices $(idxs), upper bounds: $(lmo.upper_bounds[idxs]), v: $(v[idxs])"
        end
    end
    return sum(v) ≥ lmo.N - 1e-4
end

function check_feasibility(lmo::ReverseKnapsackLMO, lb, ub, int_vars, n)
    u = copy(lmo.upper_bounds)
    u[int_vars] = min.(u[int_vars], ub)

    if sum(u) ≥ lmo.N
        return OPTIMAL
    else
        return INFEASIBLE
    end
end

"""
Hyperplane-aware rounding for the reverse knapsack constraint.
"""
function rounding_hyperplane_heuristic(
    tree::Bonobo.BnBTree,
    tlmo::TimeTrackingLMO{ManagedBoundedLMO{ReverseKnapsackLMO}},
    x,
)
    z = copy(x)
    for idx in tree.branching_indices
        z[idx] = round(x[idx])
    end

    N = tlmo.lmo.lmo.N

    non_zero_int = intersect(findall(!iszero, z), tree.branching_indices)
    cont_z =
        isempty(setdiff(collect(1:tree.root.problem.nvars), tree.branching_indices)) ? 0 :
        sum(z[setdiff(collect(1:tree.root.problem.nvars), tree.branching_indices)])
    if cont_z + sum(tlmo.lmo.upper_bounds[non_zero_int]) < N
        @debug "No heuristics improvement possible, bounds already reached, N=$(N), maximal possible sum $(cont_z + sum(tlmo.lmo.upper_bounds[non_zero_int]))"
        return [z], true
    end

    if sum(z) < N
        while sum(z) < N
            z = add_to_min(z, tlmo.lmo.upper_bounds, tree.branching_indices)
        end
    end
    return [z], false
end

"""
    bounded_compute_extreme_point(lmo::FrankWolfe.KNormBallLMO, direction, lb, ub, int_vars)
    
    Knorm: C_{K,τ} = conv { B_1(τ) ∪ B_∞(τ / K) }

Compute an extreme point of the K-norm ball using a greedy strategy.
Two candidates are constructed (ℓ∞-type and ℓ1-type), and the one minimizing
the inner product with `direction` is returned, respecting bounds and integrality.
"""
function bounded_compute_extreme_point(
    lmo::FrankWolfe.KNormBallLMO,
    direction,
    lb,
    ub,
    int_vars;
    kwargs...,
)
    K = max(min(lmo.K, length(direction)), 1)
    oinf = zero(eltype(direction))
    v = zeros(eltype(direction), length(direction))

    @inbounds for (i, dir_val) in enumerate(direction)
        temp = -lmo.right_hand_side / K * sign(dir_val)

        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            temp = clamp(temp, ceil(lb[idx]), floor(ub[idx]))
            if sign(temp) >= 0
                temp = floor(temp)
            elseif sign(temp) < 0
                temp = ceil(temp)
            end
        end

        v[i] = temp
        oinf += dir_val * temp

    end

    v1 = zeros(length(direction))

    for i in int_vars
        idx = findfirst(x -> x == i, int_vars)
        if lb[idx] > 0
            v1[i] = ceil(lb[idx])
        elseif ub[idx] < 0
            v1[i] = floor(ub[idx])
        end
    end

    perm = sortperm(abs.(direction); rev=true)

    for i in perm
        total_used = sum(abs.(v1))
        rem = lmo.right_hand_side - total_used

        if isapprox(rem, 0.0; atol=1e-6)
            break
        end

        d = direction[i]
        sgn = -sign(d)   # move opposite to gradient
        step = sgn * rem

        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            if sgn > 0
                v1[i] = clamp(floor(step), ceil(lb[idx]), floor(ub[idx]))
            elseif sgn < 0
                v1[i] = clamp(ceil(step), ceil(lb[idx]), floor(ub[idx]))
            else
                v1[i] = clamp(0, ceil(lb[idx]), floor(ub[idx]))
            end
        else
            v1[i] = step
        end
    end

    o1 = dot(v1, direction)
    if o1 < oinf
        @. v = v1
    end
    return v
end

"""
    is_simple_linear_feasible(lmo, v)

Check if `v` lies in the L₁-ball of radius τ or L∞-ball of radius τ/K.

if `v` satisfies either ball constraint then return true
"""
function is_simple_linear_feasible(lmo::FrankWolfe.KNormBallLMO, v)
    τ = lmo.right_hand_side
    K = lmo.K

    #falls into one of B1(τ) or B∞(τ/K)
    if any(isnan, v)
        @debug "v contains NaN: $(v)"
        return false
    end
    if !((sum(abs, v) <= τ + 1e-8) || (maximum(abs.(v)) <= τ / K + 1e-8))
        @debug "1 norm : $(sum(abs, v)) > τ :$τ and max norm: $(maximum(abs.(v))) > τ / K : $(τ / K)"
        return false
    end
    return true
end

"""
    check_feasibility(lmo, lb, ub, int_vars, n)

Check if there exists a vector within `[lb, ub]` satisfying L₁ or L∞ constraints.
"""
function check_feasibility(lmo::FrankWolfe.KNormBallLMO, lb, ub, int_vars, n)
    τ = lmo.right_hand_side
    K = lmo.K

    l1_min = 0.0
    #The minimum L₁ norm does not exceed τ
    @inbounds for i in eachindex(lb, ub)
        li, ui = lb[i], ub[i]
        if li > 0.0 || ui < 0.0
            l1_min += min(abs(li), abs(ui))
        end
    end
    feas_by_l1 = (l1_min ≤ τ + 1e-8)

    #Is there x ∈ [lb,ub] such that ||x||_∞ ≤ τ/K
    r = τ / K
    feas_by_linf = all(lb .≤ r) || all(-r .≤ ub)

    if feas_by_l1 || feas_by_linf
        return OPTIMAL
    else
        return INFEASIBLE
    end
end
