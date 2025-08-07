using Boscia
using Random
using LinearAlgebra
using Distributions
using StableRNGs
using FrankWolfe

seed = 1
m = 50
n = Int(floor(sqrt(m)))
corr = false

function build_data(seed, m, n, fusion, corr; scaling_C=false)
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
    C_hat = rand(2n, n)
    C = scaling_C ? 1/2n*transpose(C_hat)*C_hat : transpose(C_hat)*C_hat

    @assert rank(C) == n
    
    if fusion
        N = rand(floor(m/20):floor(m/3))
        ub = rand(1.0:m/10, m)
    else
        N = floor(1.5*n)
        u = floor(N/3)
        ub = rand(1.0:u, m)
    end
        
    return A, C, N, ub, C_hat
end

function build_integer_data(seed, m, n, fusion, corr; scaling_C=false, M=5)
    rng = StableRNG(seed)
    if corr 
        B = rand(rng, m,n)
        B = B'*B
        @assert isposdef(B)
        D = MvNormal(randn(rng, n),B)
        
        A = round.(rand(rng, D, m)')
        @assert rank(A) == n 
    else 
        A = rand(rng, -M:M, m,n)
        @assert rank(A) == n # check that A has the desired rank!
    end 
    C_hat = rand(rng, -M:M, 2n, n)
    C = scaling_C ? 1/2n*transpose(C_hat)*C_hat : transpose(C_hat)*C_hat

    @assert rank(C) == n
    
    if fusion
        N = rand(rng, floor(m/20):floor(m/3))
        ub = rand(rng, 1.0:m/10, m)
    else
        N = floor(1.5*n)
        u = floor(N/3)
        ub = rand(rng, 1.0:u, m)
    end
        
    return A, C, N, ub, C_hat
end

function build_e_criterion(A)
    m, n = size(A)
    function inf_matrix(x)
        return Symmetric(A' * diagm(x) * A)
    end

    function f(x)
        X = inf_matrix(x)   
        return (-1) * minimum(eigvals(X))    
    end

    function generate_smoothing_function(μ)

        function f_mu(x)
            X = inf_matrix(x)
            λ = eigvals(X)
            return μ * log(sum(exp.(-λ ./ μ))) - μ * log(n)
        end

        function grad_mu!(storage, x)
            X = inf_matrix(x)
            λ, V = eigen(X)
            frac = - 1/(sum(exp.(-λ ./ μ)))
            # VERSION 1: want I have figured out by hand
            #sum_exp = sum(exp(-λ[j]/ μ) * norm(V[:,j])^2 for j in 1:n)
            #for i in 1:length(x)
            #    storage[i] = frac * norm(A[i,:])^2 * sum_exp
            #end

            # VERSION 2: ChatGPT solution
            storage .= frac * sum(exp.(-λ[j]/ μ) * (A * V[:,j]).^2 for j in 1:n) 
            if any(isinf, storage)
                @show λ
                @show λ ./ μ
                @show frac, sum(exp.(-λ./ μ))
                @show V
            end
            return storage
        end
        return f_mu, grad_mu!
    end

    return f, generate_smoothing_function
end

#A, _, N, ub, _ = build_integer_data(seed, m, n, false, corr, M=10)
A, _, N, ub, _ = build_data(seed, m, n, false, corr)
simplex_lmo = Boscia.ProbabilitySimplexSimpleBLMO(N)
lmo = Boscia.ManagedBoundedLMO(simplex_lmo, fill(0.0, m), ub, collect(1:m), m)

f, generate_smoothing_function = build_e_criterion(A)

x, _, result = Boscia.solve(f, nothing, lmo; 
mode = Boscia.SMOOTHING_MODE,
settings_bnb = Boscia.settings_bnb(verbose=true, print_iter=10),
settings_smoothing = Boscia.settings_smoothing(mode=Boscia.SMOOTHING_MODE, generate_smoothing_objective = generate_smoothing_function),## , smoothing_start=5.0, smoothing_min=1.0
settings_frank_wolfe = Boscia.settings_frank_wolfe(mode=Boscia.SMOOTHING_MODE, max_fw_iter=1000, fw_verbose=false, line_search=FrankWolfe.Adaptive()),
)