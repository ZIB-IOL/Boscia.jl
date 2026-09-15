# Minimal working example for E-optimal design with Boscia.
# Fully self-contained - no ODWB dependency. Use in Boscia repo for testing.
#
# Run: julia --project=. e_optimal_boscia_minimal.jl
# Or: include("e_optimal_boscia_minimal.jl")
#
# Dependencies (Boscia brings these in): Boscia, FrankWolfe, Bonobo, LogExpFunctions

using Boscia
using FrankWolfe
using LinearAlgebra
using LogExpFunctions
using Random
using StableRNGs
using Dates
using Distributions
using SparseArrays

println("\nE-optimal design example")

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

#ENV["JULIA_DEBUG"] = "Boscia"

# ============== Parameters ==============
m = 50
n = Int(floor(sqrt(m)))
corr = true
N = Int(floor(1.5 * n * log(n)))
time_limit = 300
zero_one = true
reduced_percentage = 0.5
reduced_spectrum = false

# ============== Build data (from utilities.jl) ==============
# For corr=true, add: using Distributions  and use MvNormal in the corr branch
function build_e_optimal_data(seed, m, n, corr)
    # set up
    Random.seed!(seed)
    if corr 
        B = rand(m,n)
        B = B'*B
        @assert isposdef(B)
        D = MvNormal(randn(n),B)
        
        A = rand(D, m)'
        @assert rank(A) == n 
    else 
        A = rand(m,n)
        @assert rank(A) == n # check that A has the desired rank!
    end 
    return A
end

A = build_e_optimal_data(seed, m, n, corr)
ub = fill(1.0, m)
@show m, n, N



# =============== Build the E-criterion, its subgradients, the smoothed objective and gradient ====================
function build_e_criterion(A; L=nothing, tightened=false, N=Inf, reduced_spectrum=false, reduced_percentage=1)
    m, n = size(A)
    # FW line search may evaluate slightly outside [0,1]^m; that can make A'diag(x)A (and eigen) explode.
    gram = Symmetric(A' * A)
    # `eigvals` on sparse matrices can trigger dense fallback anyway; materialize at most once.
    gram_dense_or_not = issparse(gram.data) ? Matrix(gram) : gram    
    # For M = -A'A: singular values are |eigvals(M)| = eigvals(A'A), so
    # sigma_max(M) = lambda_max(A'A) = sigma_max(A)^2.
    sigma_max = eigmax(gram_dense_or_not)
    (!isfinite(sigma_max) || sigma_max <= 0) && return n
    function inf_matrix(x)
        xv = Vector{Float64}(undef, m)
        @inbounds for i in 1:m
            t = x[i]
            xv[i] = isfinite(t) ? clamp(t, 0.0, 1.0) : 0.0
        end
        D = LinearAlgebra.Diagonal(xv)
        return L === nothing ? Symmetric(A' * D * A) : Symmetric(L + A' * D * A)
    end

    function f(x)
        X = inf_matrix(x)   
        #return (-1) * minimum(eigvals(X))  
        return (-1) * LinearAlgebra.eigmin(X)  
    end

    function sub_grad!(storage, x)
        X = inf_matrix(x)
        λ, V = eigen(X)
        λ_min = minimum(λ)
         # Use both relative and absolute tolerance (similar to isapprox)
         tolerance = max(1e-10 * abs(λ_min), 1e-10)
         # Count eigenvalues within tolerance of the minimum
         mult= count(λ_i -> abs(λ_i - λ_min) <= tolerance, λ)
         for i in 1:mult 
            push!(storage, -(A * V[:, i]).^2)
         end
        return storage
    end
    
    k = Int(floor(n/reduced_percentage))
    @show n, k
    function generate_smoothing_function(μ; epsilon=1e-6, node_level=Inf)
        function f_mu(x)
            X = inf_matrix(x)
            λ = eigvals(X)
            add_on = tightened ? μ/(n - N + 1) * sum(x[i] * norm(A[i, :], 2)^2 for i in 1:m) : 0.0
            return μ * LogExpFunctions.logsumexp(-λ ./ μ) - μ * log(n) + add_on
        end

        function grad_mu!(storage, x)
            X = inf_matrix(x)
           # λ, V = eigen(X)
            if reduced_spectrum
                # Arpack needs a plain dense/sparse matrix; Symmetric(L+A'DA) from AGC/ACST
                # may be sparse-backed when L is absent, so materialize once.
                Y = X isa LinearAlgebra.Symmetric ? Matrix(X.data) : (issparse(X) ? Matrix(X) : X)
                λ, V = Arpack.eigs(Y, nev=k, which=:SM)
            else
                k = n
                Y = X isa LinearAlgebra.Symmetric ? Matrix(X.data) : (issparse(X) ? Matrix(X) : X)
                λ, V = eigen(Y)
            end
            add_on = tightened ? μ/(n - N + 1) * norm.(eachrow(A), 2).^2 : 0.0

            # soft max version
            log_z = LogExpFunctions.logsumexp(-λ[1:k] ./ μ)
            storage .= 0.0
            for j in 1:k
                wj = exp(-λ[j] / μ - log_z)   # ∈ (0,1]
                storage .-= wj .* (A * V[:, j]).^2
            end
            storage .+= add_on
            return storage
        end
        return f_mu, grad_mu!
    end 

    return f, sub_grad!, generate_smoothing_function
end

f, sub_grad!, generate_smoothing_function = build_e_criterion(A)

# ================= Correct dual gap with a reduced gradient ====================
function build_node_callback(m, n, A, reduced_percentage, reduced_spectrum; L=nothing)
    # maximum eigenvalue of (AA') hadamard multiplied with itself
    # Densify: for AGC/ACST, A is sparse and eigvals! has no SparseMatrixCSC method.
    AA_t = A * A'
    AA_t = issparse(AA_t) ? Matrix(AA_t) : AA_t
    op_norm = maximum(eigvals(AA_t.^2))
    cut_off = Int(floor(n/reduced_percentage))
    return function node_callback(tree, node, μ, x; primal=Inf, dual_gap=Inf, fw_status=nothing, atoms_set=nothing, resolve_integer_solution=false)
        # E-opt: A'DA. AGC/ACST: L + A'DA (same convention as build_e_criterion).
        if reduced_spectrum
            D = Diagonal(x)
            X = L === nothing ? A' * D * A : L + A' * D * A
            X = issparse(X) ? Matrix(X) : X
            @show x, sum(isnan.(x)) > 0
            @show X
            λ = eigvals(X)
            correction = 2 * op_norm * (n - cut_off) * exp(- (λ[end] - λ[cut_off])/μ) 
            dual_gap += correction
        end
    end
end

node_callback = build_node_callback(m, n, A, reduced_percentage, reduced_spectrum)

# ============== Build LMO (from utilities.jl build_blmo) ==============
function build_blmo(m, N, ub)
    simplex_lmo = Boscia.ProbabilitySimplexSimpleBLMO(N)
    blmo = Boscia.ManagedBoundedLMO(simplex_lmo, fill(0.0, m), ub, collect(1:m), m)
    return blmo
end

lmo = build_blmo(m, N, ub)


# ================== Eigenvalue based pruning =======================
function build_branch_callback_mem(
    A,
    N,
    f,
    sub_grad!;
    L=nothing,
    print_fixings::Bool=false,
    n_random::Int=10,
    tighted_to_one=Dict{Int, Int}(),
    tighted_to_zero=Dict{Int, Int}(),
    processed_tightening_nodes=0,
    number_pruned_nodes=Dict{Int, Int}(),
    processed_pruning_nodes=0,
    record_eigenvalue=false,
    eigenvalue_list=Vector{Float64}(undef, 0),
)
    m, n = size(A)
    T = eltype(A)

    l = zeros(T, m)
    u = ones(T, m)
    fixed_mask = falses(m)

    # Buffers reused across callback invocations (avoid per-node allocations).
    free_indices = Vector{Int}(undef, m)
    V_i = zeros(T, n, n)      # A[vdix,:] * A[vdix,:]'
    G_free = zeros(T, n, n)   # A_free' * A_free
    tmp = zeros(T, n, n)      # workspace for eigmin calls

    return function branch_callback(tree, node, vdix)
        if node.depth > n
            return false, false
        end

        fill!(l, zero(T))
        fill!(u, one(T))
        fill!(fixed_mask, false)
        N_star = Int(N)
        M_0 = L === nothing ? zeros(T, n, n) : copy(L)

        # collecting current fixings
        int_vars = tree.root.problem.integer_variables
        for i in int_vars
            local_ub = get(node.local_bounds.upper_bounds, i, Inf)
            local_lb = get(node.local_bounds.lower_bounds, i, -Inf)
            l[i] = isfinite(local_lb) ? local_lb : zero(T)
            if isfinite(local_lb)
                M_0 += A[i, :] * l[i] * A[i, :]'
            end
            u[i] = isfinite(local_ub) ? local_ub : one(T)
            if isfinite(local_lb) || isfinite(local_ub)
                fixed_mask[i] = true
            end
            N_star = isfinite(local_lb) ? (N_star - 1) : N_star
        end

        prune_left = false
        prune_right = false
        
        if N_star < n
            # down branch
            F_l = eigen(M_0)
            bound_left = F_l.values[N_star + 1]

            F_r = eigen(M_0 + A[vdix, :] * A[vdix, :]')
            bound_right = F_r.values[N_star + 1]

            prune_left = bound_left < -tree.incumbent
            prune_right = bound_right < -tree.incumbent
            if prune_left || prune_right
                push!(number_pruned_nodes, node.id => prune_left + prune_right)
                processed_pruning_nodes += 1
            end
        end
        if record_eigenvalue
            x = node.active_set.x
            X = A' * Diagonal(x) * A
            λ, V = eigen(X)
            if isreal(λ[1])
                push!(eigenvalue_list, λ)
            end
        end

        return prune_left, prune_right
    end
end

branch_callback = build_branch_callback_mem(A, N, f, sub_grad!)

# ============== Heuristics ==============
function build_follow_subgradient_heuristic(A, k; L=nothing)
    m, n = size(A)
    return function follow_gradient_heuristic(tree::Boscia.BnBTree, tlmo::Boscia.TimeTrackingLMO, x)
        x_new = copy(x)
        sols = []
        sol_hashes = Set{UInt}()
        for i in 1:k
            time = float(Dates.value(Dates.now() - tree.root.problem.tlmo.time_ref))
            if tree.root.options[:time_limit] < Inf &&
            time / 1000.0 ≥ tree.root.options[:time_limit] - 10
                break
            end

            # Direction to maximize λ_min: use (A*v_min)² as LMO direction (negative subgradient of -λ_min)
            X = L === nothing ? A' * Diagonal(x_new) * A : L + A' * Diagonal(x_new) * A
            if !isposdef(X)
                return [x], true
            end
            λ, V = eigen(X)
            v_min = V[:, 1]
            nabla = (A * v_min).^2
            x_new = Boscia.compute_extreme_point(tlmo, nabla)
            sol_hash = hash(x_new)
            if in(sol_hash, sol_hashes)
                break
            end
            push!(sols, x_new)
            push!(sol_hashes, sol_hash)
        end
        return sols, false
    end
end

function build_simple_randomized_rounding_heuristic(A, N, max_iter; rng=Random.default_rng())
    m, n = size(A)
    return function simple_randomized_rounding_heuristic(tree::Boscia.BnBTree, tlmo::Boscia.TimeTrackingLMO, x)
        x_new = copy(x)
        sols = []
        no_feasible_solution_found = true
        k = 1
        while k <= max_iter && no_feasible_solution_found
            for (i, x_i) in zip(collect(1:m), x)
                x_new[i] = rand(rng) < x_i ? min(1.0, ceil(x_i)) : max(0.0, floor(x_i))
            end
            if sum(x_new) == N 
                push!(sols, x_new)
                no_feasible_solution_found = false
            end
            k += 1
        end
        return sols, false
    end
end

custom_heu = []
push!(custom_heu, Boscia.Heuristic(build_follow_subgradient_heuristic(A, n), 0.5, :follow_subgradient))
push!(custom_heu, Boscia.Heuristic(build_simple_randomized_rounding_heuristic(A, N, 20), 1.0, :sr_rounding))


# ============== Compute the smoothing parameter values =====================
function estimate_design_lambda_scale(A, N; L=nothing, n_samples::Int=50, rng=Random.default_rng(),
    λ_tol::Float64=1e-8)
    m = size(A, 1)
    N_int = Int(round(N))
    @assert 1 <= N_int <= m "N=$N_int out of range for m=$m"
    λs = Float64[]
    sizehint!(λs, n_samples)
    for _ in 1:n_samples
        S = randperm(rng, m)[1:N_int]
        AS = A[S, :]
        X = AS' * AS
        if L !== nothing
            X = L + X
        end
        Xd = issparse(X) ? Matrix(X) : Matrix(X isa Symmetric ? X.data : X)
        λ = eigmin(Symmetric(Xd))
        if isfinite(λ) && λ > λ_tol
            push!(λs, λ)
        end
    end
    if !isempty(λs)
        return median(λs)
    end
    # Fallback: continuous relaxation scale (avoids μ≈0 when all random supports are singular).
    gram = A' * A
    X = L === nothing ? (N_int / m) * gram : L + (N_int / m) * gram
    Xd = issparse(X) ? Matrix(X) : Matrix(X isa Symmetric ? X.data : X)
    λ_fb = eigmin(Symmetric(Xd))
    return max(λ_fb, λ_tol, eps(Float64))
end

mu_scale_c_start=0.15
mu_scale_c_min=0.003
mu_scale_n_samples=50

λ_hat = estimate_design_lambda_scale(
            A, N;
            L=nothing,
            n_samples=mu_scale_n_samples,
            rng=rng,
        )
λ_hat = max(λ_hat, eps(Float64))
smoothing_start = mu_scale_c_start * λ_hat
smoothing_min = mu_scale_c_min * λ_hat

# ============== Settings ==============
branching_strategy = Boscia.MOST_INFEASIBLE()
settings = Boscia.create_default_settings(mode=Boscia.SMOOTHING_MODE)
settings.branch_and_bound[:verbose] = true
settings.branch_and_bound[:time_limit] = time_limit
settings.branch_and_bound[:use_shadow_set] = true
settings.branch_and_bound[:branching_strategy] = branching_strategy
settings.branch_and_bound[:print_iter] = 10
settings.branch_and_bound[:branch_callback] = branch_callback

settings.tolerances[:rel_dual_gap] = 1e-2
settings.tolerances[:fw_epsilon] = 1e-3
settings.tolerances[:min_node_fw_epsilon] = 1e-7

settings.smoothing[:generate_smoothing_objective] = generate_smoothing_function
settings.smoothing[:smoothing_start] = smoothing_start
settings.smoothing[:smoothing_min] = smoothing_min
settings.smoothing[:smoothing_min_valid] = false
settings.smoothing[:smoothing_decay] = 0.8
settings.smoothing[:max_restart_fw_iter] = 100
settings.smoothing[:node_callback] = node_callback

settings.frank_wolfe[:max_fw_iter] = 5000
settings.frank_wolfe[:line_search] = FrankWolfe.Secant()
settings.frank_wolfe[:fw_verbose] = false
settings.frank_wolfe[:lazy] = false
settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()

settings.tightening[:dual_tightening] = true
settings.tightening[:global_dual_tightening] = true

settings.heuristic[:hyperplane_aware_rounding_prob] = 0.0
settings.heuristic[:follow_gradient_prob] = 0.7
settings.heuristic[:follow_gradient_steps] = n
settings.heuristic[:custom_heuristics] = custom_heu

# ============== Solve ==============
x, _, result = Boscia.solve(f, sub_grad!, lmo, mode=Boscia.SMOOTHING_MODE, settings=settings)

# ============== Output ==============
@show x
@show result[:primal_objective]
@show result[:status]
@show result[:solution_source]
@show f(x)
