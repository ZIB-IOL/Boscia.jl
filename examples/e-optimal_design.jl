# Minimal working example for E-optimal design with Boscia.
# Fully self-contained - no ODWB dependency. Use in Boscia repo for testing.
#
# Run: julia --project=. e_optimal_boscia_minimal.jl
# Or: include("e_optimal_boscia_minimal.jl")
#
# Dependencies (Boscia brings these in): Boscia, FrankWolfe, Bonobo, LogExpFunctions

using Boscia
using FrankWolfe
using Bonobo
using LinearAlgebra
using LogExpFunctions
using Random
using Dates
using SCIP
using MathOptInterface
const MOI = MathOptInterface

#ENV["JULIA_DEBUG"] = "Boscia"

# ============== Parameters ==============
seed = 5
m = 10
n = Int(floor(sqrt(m)))
corr = false
N = Int(floor(1.5 * n * log(n)))
time_limit = 300
zero_one = true

# ============== Build data (from utilities.jl) ==============
# For corr=true, add: using Distributions  and use MvNormal in the corr branch
function build_data(seed, m, n, fusion, corr; scaling_C=false, zero_one=false, N=-Inf)
    Random.seed!(seed)
    if corr
        error("corr=true requires: using Distributions. Use corr=false for minimal deps.")
    end
    A = rand(m, n)
    @assert rank(A) == n
    C_hat = rand(2n, n)
    C = scaling_C ? 1 / (2n) * transpose(C_hat) * C_hat : transpose(C_hat) * C_hat
    @assert rank(C) == n

    if fusion
        N = N == -Inf ? rand(floor(Int, m / 20):floor(Int, m / 3)) : N
        ub = rand(1.0:m / 10, m)
    else
        N = N == -Inf ? floor(Int, 1.5 * n) : N
        u = floor(Int, N / 3)
        ub = rand(1.0:u, m)
    end

    if zero_one
        return A, C, N, fill(1.0, m), C_hat
    end
    return A, C, N, ub, C_hat
end

A, C, N, ub, _ = build_data(seed, m, n, false, corr; zero_one=zero_one, N=N)
@show m, n, N

# ============== Build E-criterion ==============
function build_e_criterion(A)
    m, n = size(A)
    function inf_matrix(x)
        return Symmetric(A' * diagm(x) * A)
    end

    function f(x)
        X = inf_matrix(x)
        return (-1) * minimum(eigvals(X))
    end

    function sub_grad!(storage, x)
        X = inf_matrix(x)
        λ, V = eigen(X)
        λ_min = minimum(λ)
        tolerance = max(1e-10 * abs(λ_min), 1e-10)
        mult = count(λ_i -> abs(λ_i - λ_min) <= tolerance, λ)
        for i in 1:mult
            push!(storage, -(A * V[:, i]).^2)
        end
        return storage
    end

#=    function generate_smoothing_function(μ)
        function f_mu(x)
            X = inf_matrix(x)
            λ = eigvals(X)
            return μ * LogExpFunctions.logsumexp(-λ ./ μ) - μ * log(n)
        end

        function grad_mu!(storage, x)
            X = inf_matrix(x)
            λ, V = eigen(X)
            frac = -1 / exp(LogExpFunctions.logsumexp(-λ ./ μ))
            storage .= frac * sum(LogExpFunctions.xexpy.((A * V[:, j]).^2, -λ[j] / μ) for j in 1:n)
            return storage
        end
        return f_mu, grad_mu!
    end=#

    function generate_smoothing_function(μ)
        function f_mu(x)
            X = -inf_matrix(x)
            λ = eigvals(X)
            return μ * LogExpFunctions.logsumexp(λ ./ μ) - μ * log(n)
        end

        function grad_mu!(storage, x)
            X = -inf_matrix(x)
            λ, V = eigen(X)
            frac = -1 / exp(LogExpFunctions.logsumexp(λ ./ μ))
            storage .= frac * sum(LogExpFunctions.xexpy.((A * V[:, j]).^2, λ[j] / μ) for j in 1:n)
            return storage
        end
        return f_mu, grad_mu!
    end

    return f, sub_grad!, generate_smoothing_function
end

f, sub_grad!, generate_smoothing_function = build_e_criterion(A)

# ============== Build LMO (from utilities.jl build_blmo) ==============
function build_blmo(m, N, ub)
    simplex_lmo = Boscia.ProbabilitySimplexSimpleBLMO(N)
    blmo = Boscia.ManagedBoundedLMO(simplex_lmo, fill(0.0, m), ub, collect(1:m), m)
    return blmo
end

function build_moi_lmo(m, N, ub)
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    x = MOI.add_variables(o, m)
    for i in 1:m
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, x[i], MOI.ZeroOne())
    end
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(m), x), 0.0), MOI.EqualTo(Float64(N)))
    return FrankWolfe.MathOptLMO(o)
end

lmo = build_blmo(m, N, ub)
#lmo = build_moi_lmo(m, N, ub)

# ============== Heuristics ==============
function linearly_independent_rows(A, m, n_target)
    S = Int[]
    for i in 1:m
        S_i = vcat(S, i)
        if rank(A[S_i, :]) == length(S_i)
            S = S_i
        end
        if length(S) == n_target
            return S
        end
    end
    return S
end

function find_large_leverage_set(A, initial_idx_set, target_size)
    m, n = size(A)
    if target_size >= length(initial_idx_set)
        return initial_idx_set, false
    end
    @assert target_size >= n
    current_set = copy(initial_idx_set)

    if length(current_set) == target_size
        if rank(A[current_set, :]) == n
            return current_set, true
        else
            indep = linearly_independent_rows(A[current_set, :], length(current_set), min(n, length(current_set)))
            current_set = current_set[indep]
        end
    end

    if length(current_set) < n || rank(A[current_set, :]) < min(n, length(current_set))
        if !isempty(current_set)
            indep = linearly_independent_rows(A[current_set, :], length(current_set), min(n, length(current_set)))
            current_set = current_set[indep]
        end
        while length(current_set) < n
            remaining = setdiff(1:m, current_set)
            isempty(remaining) && break
            best_idx = nothing
            for idx in remaining
                test_set = vcat(current_set, idx)
                if rank(A[test_set, :]) > rank(A[current_set, :])
                    best_idx = idx
                    break
                end
            end
            push!(current_set, best_idx !== nothing ? best_idx : remaining[1])
        end
    end

    while length(current_set) < target_size
        remaining = setdiff(1:m, current_set)
        isempty(remaining) && break
        best_idx = nothing
        for idx in remaining
            test_set = vcat(current_set, idx)
            if rank(A[test_set, :]) == n
                best_idx = idx
                break
            end
        end
        if best_idx === nothing
            cr = rank(A[current_set, :])
            for idx in remaining
                test_set = vcat(current_set, idx)
                if rank(A[test_set, :]) >= cr
                    best_idx = idx
                    break
                end
            end
        end
        push!(current_set, best_idx !== nothing ? best_idx : remaining[1])
    end
    return current_set, true
end

function build_follow_subgradient_heuristic(A, k)
    m, n = size(A)
    return function follow_gradient_heuristic(tree, tlmo, x)
        x_new = copy(x)
        sols = []
        sol_hashes = Set{UInt}()
        for i in 1:k
            time = float(Dates.value(Dates.now() - tree.root.problem.tlmo.time_ref))
            if tree.root.options[:time_limit] < Inf && time / 1000.0 ≥ tree.root.options[:time_limit] - 10
                break
            end
            X = A' * Diagonal(x_new) * A
            λ, V = eigen(X)
            nabla = (A * V[:, 1]).^2  # direction to maximize λ_min
            x_new = Boscia.compute_extreme_point(tlmo, nabla)
            sol_hash = hash(x_new)
            sol_hash in sol_hashes && break
            push!(sols, x_new)
            push!(sol_hashes, sol_hash)
        end
        return sols, false
    end
end

function build_simple_randomized_rounding_heuristic(A, N, max_iter; rng=Random.default_rng())
    return function simple_randomized_rounding_heuristic(tree, tlmo, x)
        x_new = copy(x)
        sols = []
        for k in 1:max_iter
            for (i, x_i) in zip(1:length(x), x)
                x_new[i] = rand(rng) < x_i ? min(1.0, ceil(x_i)) : max(0.0, floor(x_i))
            end
            if sum(x_new) == N
                push!(sols, x_new)
                return sols, false
            end
        end
        return sols, false
    end
end

function build_pipage_rounding_heuristic(A, N; threshold=0.8, epsilon=1)
    m, n = size(A)
    inf_matrix(x) = A' * Diagonal(x) * A
    return function pipage_rounding_heuristic(tree, tlmo, x)
        x_new = copy(x)
        idx_set = findall(x .> threshold)
        cut_off = Int(floor(min(max(n * log(n) / epsilon^2, length(idx_set)), N)))
        S, feasible = find_large_leverage_set(A, idx_set, cut_off)
        if !feasible
            return [x], true
        end
        node = tree.nodes[tree.root.current_node_id[]]
        original_bounds = copy(node.local_bounds)
        local_bounds = Boscia.IntegerBounds()
        for i in S
            push!(local_bounds, (i, 1.0), :lessthan)
            push!(local_bounds, (i, 1.0), :greaterthan)
        end
        x_new[S] .= 1.0
        x_new[setdiff(1:m, S)] .= 0.0
        X_inv = inv(inf_matrix(x_new))
        for i in setdiff(1:m, S)
            leverage = A[i, :]' * X_inv * A[i, :]
            if leverage > epsilon^2 / (10 * log(n)) || isapprox(x[i], 0.0, atol=1e-10)
                push!(local_bounds, (i, 0.0), :lessthan)
                push!(local_bounds, (i, 0.0), :greaterthan)
            else
                push!(local_bounds, (i, 1.0), :lessthan)
                push!(local_bounds, (i, 0.0), :greaterthan)
            end
        end
        Boscia.build_LMO(tlmo, tree.root.problem.integer_variable_bounds, local_bounds, tree.root.problem.integer_variables)
        status = Boscia.check_feasibility(tlmo)
        if status == Boscia.INFEASIBLE || status == Boscia.UNBOUNDED
            Boscia.build_LMO(tlmo, tree.root.problem.integer_variable_bounds, original_bounds, tree.root.problem.integer_variables)
            return [x], true
        end
        v = Boscia.compute_extreme_point(tlmo, rand(length(x)))
        active_set = FrankWolfe.ActiveSet([(1.0, v)])
        x_pipage, _, _, _ = Boscia.solve_frank_wolfe(
            tree.root.options[:variant],
            tree.root.problem.f,
            tree.root.problem.g,
            tree.root.problem.tlmo,
            active_set;
            epsilon=node.fw_dual_gap_limit,
            max_iteration=tree.root.options[:max_fw_iter],
            line_search=tree.root.options[:line_search],
            lazy=tree.root.options[:lazy],
            lazy_tolerance=tree.root.options[:lazy_tolerance],
            callback=tree.root.options[:callback],
            verbose=tree.root.options[:fw_verbose],
        )
        for (idx, x_i) in enumerate(x_pipage)
            x_pipage[idx] = rand() < x_i ? min(1.0, ceil(x_i)) : max(0.0, floor(x_i))
        end
        Boscia.build_LMO(tlmo, tree.root.problem.integer_variable_bounds, original_bounds, tree.root.problem.integer_variables)
        return [x_pipage], false
    end
end

function build_greedy_fedorov_heuristic(A, N, max_iter; tolerance=0.0)
    m, n = size(A)
    inf_matrix(x) = A' * Diagonal(x) * A
    return function greedy_fedorov_heuristic(tree, tlmo, x)
        z = copy(tree.incumbent_solution.solution)
        sols = []
        improved = false
        f = tree.root.options[:mode] == Boscia.SMOOTHING_MODE ? tree.root.options[:original_objective] : tree.root.problem.f
        for k in 1:max_iter
            improved && break
            z_idx = findall(z .> 0.0)
            leverage = [A[idx, :]' * inv(inf_matrix(z)) * A[idx, :] for idx in z_idx]
            perm = sortperm(leverage)
            for i in perm
                best_idx = 0
                for j in setdiff(1:m, z_idx)
                    z_new = copy(z)
                    z_new[j] = 1.0
                    z_new[z_idx[i]] = 0.0
                    if sum(z_new) == N && f(z_new) > f(z) + tolerance
                        best_idx = j
                        break
                    end
                end
                if best_idx != 0
                    z[z_idx[i]] = 0.0
                    z[best_idx] = 1.0
                    improved = true
                    push!(sols, copy(z))
                    break
                end
            end
        end
        return sols, false
    end
end

# ============== Assemble heuristics ==============
custom_heu = []
push!(custom_heu, Boscia.Heuristic(build_follow_subgradient_heuristic(A, n), 0.5, :follow_subgradient))
push!(custom_heu, Boscia.Heuristic(build_simple_randomized_rounding_heuristic(A, N, 20), 1.0, :sr_rounding))
if N > 1.5 * n
    push!(custom_heu, Boscia.Heuristic(build_pipage_rounding_heuristic(A, N), 0.3, :pipage_rounding))
end
push!(custom_heu, Boscia.Heuristic(build_greedy_fedorov_heuristic(A, N, 10), 0.4, :fedorov))

# ============== Settings ==============
#branching_strategy = Bonobo.MOST_INFEASIBLE()
branching_strategy = Boscia.BRANCH_ALL()
settings = Boscia.create_default_settings(mode=Boscia.SMOOTHING_MODE)
settings.branch_and_bound[:verbose] = true
settings.branch_and_bound[:time_limit] = time_limit
settings.branch_and_bound[:node_limit] = 2^(m + 1) - 1
settings.branch_and_bound[:use_shadow_set] = true
settings.branch_and_bound[:branching_strategy] = branching_strategy
#settings.branch_and_bound[:start_solution] = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
settings.branch_and_bound[:print_iter] = 1

settings.tolerances[:rel_dual_gap] = 5e-2
settings.tolerances[:fw_epsilon] = 1e-3
settings.tolerances[:min_node_fw_epsilon] = 1e-7

settings.smoothing[:generate_smoothing_objective] = generate_smoothing_function
settings.smoothing[:smoothing_start] = 5.0 #m/20
settings.smoothing[:smoothing_min] = 0.1#1e-2
settings.smoothing[:smoothing_min_valid] = false
settings.smoothing[:smoothing_decay] = 0.9
settings.smoothing[:use_sub_grad_info] = false

settings.frank_wolfe[:max_fw_iter] = 1000
settings.frank_wolfe[:line_search] = FrankWolfe.Secant()
settings.frank_wolfe[:fw_verbose] = false
settings.frank_wolfe[:lazy] = false
settings.frank_wolfe[:variant] = Boscia.BlendedPairwiseConditionalGradient()#Boscia.DecompositionInvariantConditionalGradient()

settings.tightening[:dual_tightening] = true
settings.tightening[:global_dual_tightening] = true

settings.heuristic[:hyperplane_aware_rounding_prob] = 0.0
settings.heuristic[:follow_gradient_prob] = 0.7
settings.heuristic[:follow_gradient_steps] = n
#settings.heuristic[:rounding_lmo_01_prob] = 0.8
#settings.heuristic[:probability_rounding_prob] = 0.8
settings.heuristic[:rounding_prob] = 1.0
settings.heuristic[:custom_heuristics] = custom_heu

# ============== Solve ==============
x, _, result = Boscia.solve(f, sub_grad!, lmo, mode=Boscia.SMOOTHING_MODE, settings=settings)

# ============== Output ==============
@show x
@show result[:primal_objective]
@show result[:status]
@show result[:solution_source]
@show f(x)
