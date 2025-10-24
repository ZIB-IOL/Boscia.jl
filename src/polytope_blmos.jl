"""
    CubeSimpleLMO{T}(lower_bounds, upper_bounds, int_vars)

Hypercube with lower and upper bounds implementing the `SimpleBoundableLMO` interface.
"""
struct CubeSimpleLMO <: FrankWolfe.LinearMinimizationOracle
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    int_vars::Vector{Int}
end

const CubeSimpleBLMO = CubeSimpleLMO

"""
     bounded_compute_extreme_point(sblmo::CubeSimpleLMO, d, lb, ub, int_vars; kwargs...)

If the entry is positve, choose the lower bound. Else, choose the upper bound.
"""
function bounded_compute_extreme_point(lmo::CubeSimpleLMO, d, lb, ub, int_vars; kwargs...)
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

function is_simple_linear_feasible(lmo::CubeSimpleLMO, v)
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

function is_simple_inface_feasible(lmo::CubeSimpleLMO, a, x, lb, ub, int_vars; kwargs...)
    return is_simple_inface_feasible_subroutine(lmo, a, x, lb, ub, int_vars; kwargs)
end

function is_decomposition_invariant_oracle_simple(lmo::CubeSimpleLMO)
    return true
end

"""
If the entry in x is at the boundary, choose the corresponding bound.
Otherwise, if the entry in direction is positve, choose the lower bound. Else, choose the upper bound.
"""
function bounded_compute_inface_extreme_point(
    lmo::CubeSimpleLMO,
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
function bounded_dicg_maximum_step(lmo::CubeSimpleLMO, direction, x, lb, ub, int_vars; kwargs...)
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
struct ProbabilitySimplexSimpleLMO <: FrankWolfe.LinearMinimizationOracle
    N::Float64
end

const ProbabilitySimplexSimpleBLMO = ProbabilitySimplexSimpleLMO

function is_decomposition_invariant_oracle_simple(lmo::ProbabilitySimplexSimpleLMO)
    return true
end

function is_simple_inface_feasible(
    lmo::ProbabilitySimplexSimpleLMO,
    a,
    x,
    lb,
    ub,
    int_vars;
    kwargs...,
)
    return is_simple_inface_feasible_subroutine(lmo, a, x, lb, ub, int_vars; kwargs)
end

"""
    bounded_compute_extreme_point(lmo::ProbabilitySimplexSimpleLMO, d, lb, ub, int_vars; kwargs...)

Assign the largest possible values to the entries corresponding to the smallest entries of d.
"""
function bounded_compute_extreme_point(
    lmo::ProbabilitySimplexSimpleLMO,
    d,
    lb,
    ub,
    int_vars;
    kwargs...,
)
    v = zeros(length(d))
    indices = collect(1:length(d))
    perm = sortperm(d)

    # The lower bounds always have to be met. 
    v[int_vars] = lb

    for i in indices[perm]
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
Fix the corresponding entries to the boudary based on the given x.
Assign the largest possible values to the unfixed entries corresponding to the smallest entries of d.
"""
function bounded_compute_inface_extreme_point(
    lmo::ProbabilitySimplexSimpleLMO,
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
            idx = findfirst(x -> x == i, int_vars)
            if isapprox(x[i], lb[idx]; atol=atol, rtol=rtol)
                push!(fixed_vars, i)
            elseif isapprox(x[i], ub[idx]; atol=atol, rtol=rtol)
                push!(fixed_vars, i)
                a[i] = ub[idx]
            end
        else
            if isapprox(x[i], 0.0; atol=atol, rtol=rtol)
                push!(fixed_vars, i)
            end
            if isapprox(x[i], 0.0; atol=atol, rtol=rtol)
                push!(fixed_vars, i)
                a[i] = lmo.N
            end
        end
    end

    if sum(a) == lmo.N
        return a
    end

    non_fixed_idx = setdiff(indices, fixed_vars)
    d_updated = d[non_fixed_idx]
    perm = sortperm(d_updated)
    sorted = non_fixed_idx[perm]

    for i in sorted
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            a[i] += min(ub[idx] - lb[idx], lmo.N - sum(a))
        else
            a[i] += lmo.N - sum(a)
        end
        if sum(a) == lmo.N
            return a
        end
    end

    return a
end

"""
Compute the maximum step size for each entry and return the minium of all the possible step sizes.
"""
function bounded_dicg_maximum_step(
    lmo::ProbabilitySimplexSimpleLMO,
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

function is_simple_linear_feasible(lmo::ProbabilitySimplexSimpleLMO, v)
    if sum(v .≥ 0) < length(v)
        @debug "v has negative entries: $(v)"
        return false
    end
    return isapprox(sum(v), lmo.N, atol=1e-4, rtol=1e-2)
end

function check_feasibility(lmo::ProbabilitySimplexSimpleLMO, lb, ub, int_vars, n)
    m = n - length(int_vars)
    if sum(lb) ≤ lmo.N ≤ sum(ub) + m * lmo.N
        return OPTIMAL
    else
        INFEASIBLE
    end
end

"""
     rounding_hyperplane_heuristic(tree::Bonobo.BnBTree, tlmo::TimeTrackingLMO{ManagedBoundedLMO{ProbabilitySimplexSimpleLMO}}, x) 

Hyperplane-aware rounding for the probability simplex.
"""
function rounding_hyperplane_heuristic(
    tree::Bonobo.BnBTree,
    tlmo::TimeTrackingLMO{ManagedBoundedLMO{ProbabilitySimplexSimpleLMO}},
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
    UnitSimplexSimpleLMO(N)

The scaled unit simplex with `∑ x ≤ N`.
"""
struct UnitSimplexSimpleLMO <: FrankWolfe.LinearMinimizationOracle
    N::Float64
end

const UnitSimplexSimpleBLMO = UnitSimplexSimpleLMO

function is_decomposition_invariant_oracle_simple(lmo::UnitSimplexSimpleLMO)
    return true
end

function is_simple_inface_feasible(lmo::UnitSimplexSimpleLMO, a, x, lb, ub, int_vars; kwargs...)
    if isapprox(sum(x), N; atol=atol, rtol=rtol) && !isapprox(sum(a), N; atol=atol, rtol=rtol)
        return false
    end
    return is_simple_inface_feasible_subroutine(lmo, a, x, lb, ub, int_vars; kwargs)
end

"""
    bounded_compute_extreme_point(lmo::UnitSimplexSimpleLMO, d, lb, ub, int_vars; kwargs...)

For all positive entries of d, assign the corresponding lower bound.
For non-positive entries, assign largest possible value in increasing order.
"""
function bounded_compute_extreme_point(lmo::UnitSimplexSimpleLMO, d, lb, ub, int_vars; kwargs...)
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
    lmo::UnitSimplexSimpleLMO,
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
            idx = findfirst(x -> x == i, int_vars)
            if isapprox(x[i], lb[idx]; atol=atol, rtol=rtol)
                push!(fixed_vars, i)
            elseif isapprox(x[i], ub[idx]; atol=atol, rtol=rtol)
                push!(fixed_vars, i)
                a[i] = ub[idx]
            end
        else
            if isapprox(x[i], 0.0; atol=atol, rtol=rtol)
                push!(fixed_vars, i)
            end
            if isapprox(x[i], 0.0; atol=atol, rtol=rtol)
                push!(fixed_vars, i)
                a[i] = lmo.N
            end
        end
    end

    if sum(a) == lmo.N
        return a
    end

    non_fixed_idx = setdiff(indices, fixed_vars)
    d_updated = d[non_fixed_idx]
    idx_neg = findall(x -> x <= 0, d_updated)
    perm = sortperm(d_updated[idx_neg])
    sorted_neg = idx_neg[perm]
    sorted = non_fixed_idx[sorted_neg]

    for i in sorted
        if i in int_vars
            idx = findfirst(x -> x == i, int_vars)
            a[i] += min(ub[idx] - lb[idx], lmo.N - sum(a))
        else
            a[i] += lmo.N - sum(a)
        end
        if sum(a) == lmo.N
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
    lmo::UnitSimplexSimpleLMO,
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
        gamma_max = min(gamma_max, (sum(x) - lmo.N) / sum(direction))
    end

    return gamma_max
end

function is_simple_linear_feasible(lmo::UnitSimplexSimpleLMO, v)
    if sum(v .≥ 0) < length(v)
        @debug "v has negative entries: $(v)"
        return false
    end
    return sum(v) ≤ lmo.N + 1e-3
end

function check_feasibility(lmo::UnitSimplexSimpleLMO, lb, ub, int_vars, n)
    if sum(lb) ≤ lmo.N
        return OPTIMAL
    else
        INFEASIBLE
    end
end

"""
    rounding_hyperplane_heuristic(tree::Bonobo.BnBTree, tlmo::TimeTrackingLMO{ManagedBoundedLMO{UnitSimplexSimpleLMO}}, x) 
    
Hyperplane-aware rounding for the unit simplex.
"""
function rounding_hyperplane_heuristic(
    tree::Bonobo.BnBTree,
    tlmo::TimeTrackingLMO{ManagedBoundedLMO{UnitSimplexSimpleLMO}},
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
    BirkhoffLMO

A simple LMO that computes the extreme point given the node specific bounds on the integer variables.
Can be stateless since all of the bound management is done by the ManagedBoundedLMO.   
"""
struct BirkhoffLMO <: FrankWolfe.LinearMinimizationOracle
    append_by_column::Bool
    dim::Int
    int_vars::Vector{Int}
    atol::Float64
    rtol::Float64
end

const BirkhoffBLMO = BirkhoffLMO

BirkhoffLMO(dim, int_vars; append_by_column=true) = BirkhoffLMO(dim, int_vars; append_by_column=true)

"""
Computes the extreme point given an direction d, the current lower and upper bounds on the integer variables, and the set of integer variables.
"""
function Boscia.bounded_compute_extreme_point(lmo::BirkhoffLMO, d, lb, ub, int_vars; kwargs...)
    n = lmo.dim

    if size(d, 2) == 1
        d = lmo.append_by_column ? reshape(d, (n, n)) : transpose(reshape(d, (n, n)))
    end

    fixed_to_one_rows = Int[]
    fixed_to_one_cols = Int[]
    delete_ub = Int[]
    for j in 1:n
        for i in 1:n
            if lb[(j-1)*n+i] >= 1 - eps()
                if lmo.append_by_column
                    push!(fixed_to_one_rows, i)
                    push!(fixed_to_one_cols, j)
                    append!(delete_ub, union(collect(((j-1)*n+1):(j*n)), collect(i:n:(n^2))))
                else
                    push!(fixed_to_one_rows, j)
                    push!(fixed_to_one_cols, i)
                    append!(delete_ub, union(collect(((i-1)*n+1):(i*n)), collect(j:n:(n^2))))
                end
            end
        end
    end

    sort!(delete_ub)
    unique!(delete_ub)
    nfixed = length(fixed_to_one_cols)
    nreduced = n - nfixed
    reducedub = copy(ub)
    deleteat!(reducedub, delete_ub)

    # stores the indices of the original matrix that are still in the reduced matrix
    index_map_rows = fill(1, nreduced)
    index_map_cols = fill(1, nreduced)
    idx_in_map_row = 1
    idx_in_map_col = 1
    for orig_idx in 1:n
        if orig_idx ∉ fixed_to_one_rows
            index_map_rows[idx_in_map_row] = orig_idx
            idx_in_map_row += 1
        end
        if orig_idx ∉ fixed_to_one_cols
            index_map_cols[idx_in_map_col] = orig_idx
            idx_in_map_col += 1
        end
    end
    type = typeof(d[1, 1])
    d2 = ones(Union{type,Missing}, nreduced, nreduced)
    for j in 1:nreduced
        for i in 1:nreduced
            # interdict arc when fixed to zero
            if reducedub[(j-1)*nreduced+i] <= eps()
                if lmo.append_by_column
                    d2[i, j] = missing
                else
                    d2[j, i] = missing
                end
            else
                if lmo.append_by_column
                    d2[i, j] = d[index_map_rows[i], index_map_cols[j]]
                else
                    d2[j, i] = d[index_map_rows[j], index_map_cols[i]]
                end
            end
        end
    end
    m = SparseArrays.spzeros(n, n)
    for (i, j) in zip(fixed_to_one_rows, fixed_to_one_cols)
        m[i, j] = 1
    end
    res_mat = Hungarian.munkres(d2)
    (rows, cols, vals) = SparseArrays.findnz(res_mat)
    @inbounds for i in eachindex(cols)
        m[index_map_rows[rows[i]], index_map_cols[cols[i]]] = (vals[i] == 2)
    end

    m = if lmo.append_by_column
        # Convert sparse matrix to sparse vector by columns
        I, J, V = SparseArrays.findnz(m)
        linear_indices = (J .- 1) .* n .+ I
        SparseArrays.sparsevec(linear_indices, V, n^2)
    else
        # Convert sparse matrix to sparse vector by rows (transpose first)
        mt = SparseArrays.sparse(LinearAlgebra.transpose(m))
        I, J, V = SparseArrays.findnz(mt)
        linear_indices = (J .- 1) .* n .+ I
        SparseArrays.sparsevec(linear_indices, V, n^2)
    end
    return m
end

"""
Computes the inface extreme point given an direction d, x, the current lower and upper bounds on the integer variables, and the set of integer variables.
"""
function Boscia.bounded_compute_inface_extreme_point(
    lmo::BirkhoffLMO,
    direction,
    x,
    lb,
    ub,
    int_vars;
    kwargs...,
)
    n = lmo.dim

    if size(direction, 2) == 1
        direction =
            lmo.append_by_column ? reshape(direction, (n, n)) :
            transpose(reshape(direction, (n, n)))
    end

    if size(x, 2) == 1
        x = lmo.append_by_column ? reshape(x, (n, n)) : transpose(reshape(x, (n, n)))
    end
    fixed_to_one_rows = Int[]
    fixed_to_one_cols = Int[]
    delete_ub = Int[]

    for idx in eachindex(int_vars)
        if lb[idx] >= 1 - eps()
            var_idx = int_vars[idx]
            if lmo.append_by_column
                j = ceil(Int, var_idx / n)
                i = Int(var_idx - n * (j - 1))
                push!(fixed_to_one_rows, i)
                push!(fixed_to_one_cols, j)
                append!(delete_ub, union(collect(((j-1)*n+1):(j*n)), collect(i:n:(n^2))))
            else
                i = ceil(int, var_idx / n)
                j = Int(var_idx - n * (j - 1))
                push!(fixed_to_one_rows, j)
                push!(fixed_to_one_cols, i)
                append!(delete_ub, union(collect(((i-1)*n+1):(i*n)), collect(j:n:(n^2))))
            end
        end
    end

    for j in 1:n
        if j ∉ fixed_to_one_cols
            for i in 1:n
                if i ∉ fixed_to_one_rows
                    if x[i, j] >= 1 - eps()
                        push!(fixed_to_one_rows, i)
                        push!(fixed_to_one_cols, j)
                        if lmo.append_by_column
                            append!(
                                delete_ub,
                                union(collect(((j-1)*n+1):(j*n)), collect(i:n:(n^2))),
                            )
                        else
                            append!(
                                delete_ub,
                                union(collect(((i-1)*n+1):(i*n)), collect(j:n:(n^2))),
                            )
                        end
                    end
                end
            end
        end
    end

    sort!(delete_ub)
    unique!(delete_ub)
    fixed_to_one_cols = unique!(fixed_to_one_cols)
    fixed_to_one_rows = unique!(fixed_to_one_rows)
    nfixed = length(fixed_to_one_cols)
    nreduced = n - nfixed
    reducedub = copy(ub)
    reducedintvars = copy(int_vars)
    delete_ub_idx = findall(x -> x in delete_ub, int_vars)
    deleteat!(reducedub, delete_ub_idx)
    deleteat!(reducedintvars, delete_ub_idx)
    # stores the indices of the original matrix that are still in the reduced matrix
    index_map_rows = fill(1, nreduced)
    index_map_cols = fill(1, nreduced)
    idx_in_map_row = 1
    idx_in_map_col = 1
    for orig_idx in 1:n
        if orig_idx ∉ fixed_to_one_rows
            index_map_rows[idx_in_map_row] = orig_idx
            idx_in_map_row += 1
        end
        if orig_idx ∉ fixed_to_one_cols
            index_map_cols[idx_in_map_col] = orig_idx
            idx_in_map_col += 1
        end
    end
    type = typeof(direction[1, 1])
    d2 = ones(Union{type,Missing}, nreduced, nreduced)

    for j in 1:nreduced
        for i in 1:nreduced
            idx = (index_map_cols[j] - 1) * n + index_map_rows[i]
            if lmo.append_by_column
                if x[index_map_rows[i], index_map_cols[j]] <= eps()
                    d2[i, j] = missing
                else
                    d2[i, j] = direction[index_map_rows[i], index_map_cols[j]]
                end
            else
                if x[index_map_rows[i], index_map_cols[j]] <= eps()
                    d2[j, i] = missing
                else
                    d2[j, i] = direction[index_map_rows[j], index_map_cols[i]]
                end
            end
            # interdict arc when fixed to zero
            if idx in reducedintvars
                reducedub_idx = findfirst(x -> x == idx, reducedintvars)
                if reducedub[reducedub_idx] <= eps()
                    if lmo.append_by_column
                        d2[i, j] = missing
                    else
                        d2[j, i] = missing
                    end
                end
            end
        end
    end

    m = SparseArrays.spzeros(n, n)
    for (i, j) in zip(fixed_to_one_rows, fixed_to_one_cols)
        m[i, j] = 1
    end
    res_mat = Hungarian.munkres(d2)
    (rows, cols, vals) = SparseArrays.findnz(res_mat)
    @inbounds for i in eachindex(cols)
        m[index_map_rows[rows[i]], index_map_cols[cols[i]]] = (vals[i] == 2)
    end

    m = if lmo.append_by_column
        # Convert sparse matrix to sparse vector by columns
        I, J, V = SparseArrays.findnz(m)
        linear_indices = (J .- 1) .* n .+ I
        SparseArrays.sparsevec(linear_indices, V, n^2)
    else
        # Convert sparse matrix to sparse vector by rows (transpose first)
        mt = SparseArrays.sparse(LinearAlgebra.transpose(m))
        I, J, V = SparseArrays.findnz(mt)
        linear_indices = (J .- 1) .* n .+ I
        SparseArrays.sparsevec(linear_indices, V, n^2)
    end

    return m
end

"""
LMO-like operation which computes a vertex minimizing in `direction` on the face defined by the current fixings.
Fixings are maintained by the oracle (or deduced from `x` itself).
"""
function bounded_dicg_maximum_step(lmo::BirkhoffLMO, direction, x, lb, ub, int_vars; kwargs...)
    n = lmo.dim

    direction =
        lmo.append_by_column ? reshape(direction, (n, n)) : transpose(reshape(direction, (n, n)))
    x = lmo.append_by_column ? reshape(x, (n, n)) : transpose(reshape(x, (n, n)))
    return FrankWolfe.dicg_maximum_step(FrankWolfe.BirkhoffPolytopeLMO(), direction, x)
end

function is_decomposition_invariant_oracle_simple(lmo::BirkhoffLMO)
    return true
end

function dicg_split_vertices_set_simple(lmo::BirkhoffLMO, x, vidx)
    x0_left = copy(x)
    x0_right = copy(x)
    return x0_left, x0_right
end

"""
The sum of each row and column has to be equal to 1.
"""
function is_simple_linear_feasible(lmo::BirkhoffLMO, v::AbstractVector)
    n = lmo.dim
    for i in 1:n
        # append by column ? column sum : row sum 
        if !isapprox(sum(v[((i-1)*n+1):(i*n)]), 1.0, atol=1e-6, rtol=1e-3)
            @debug "Column sum not 1: $(sum(v[((i-1)*n+1):(i*n)]))"
            return false
        end
        # append by column ? row sum : column sum
        if !isapprox(sum(v[i:n:n^2]), 1.0, atol=1e-6, rtol=1e-3)
            @debug "Row sum not 1: $(sum(v[i:n:n^2]))"
            return false
        end
    end
    return true
end

function check_feasibility(lmo::BirkhoffLMO, lb, ub, int_vars, n)
    # For double stochastic matrices, each row and column must sum to 1
    # We check if the bounds allow for feasible assignments

    n0 = Int(sqrt(n))
    # Initialize row and column bound tracking
    row_min_sum = zeros(n)  # minimum possible sum for each row
    row_max_sum = zeros(n)  # maximum possible sum for each row
    col_min_sum = zeros(n)  # minimum possible sum for each column
    col_max_sum = zeros(n)  # maximum possible sum for each column

    # Process each integer variable
    for idx in eachindex(int_vars)
        var_idx = int_vars[idx]

        # Convert linear index to (row, col) based on storage format
        if lmo.append_by_column
            j = ceil(Int, var_idx / n0)  # column index
            i = Int(var_idx - n0 * (j - 1))  # row index
        else
            i = ceil(Int, var_idx / n0)  # row index  
            j = Int(var_idx - n0 * (i - 1))  # column index
        end
        # Add bounds to row and column sums
        row_min_sum[i] += lb[idx]
        row_max_sum[i] += ub[idx]
        col_min_sum[j] += lb[idx]
        col_max_sum[j] += ub[idx]
    end
    # Check feasibility: each row and column must be able to sum to exactly 1
    for i in 1:n0
        # Check row sum constraints
        if row_min_sum[i] > 1 + eps() || row_max_sum[i] < 1 - eps()
            return INFEASIBLE
        end
        # Check column sum constraints  
        if col_min_sum[i] > 1 + eps() || col_max_sum[i] < 1 - eps()
            return INFEASIBLE
        end
    end

    return OPTIMAL
end
