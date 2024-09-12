"""
    build_data

seed - for the Random functions.
m    - number of experiments.
fusion - boolean deiciding whether we build the fusion or standard problem.
corr - boolean deciding whether we build the independent or correlated data.   
"""
function build_data(seed, m)
    # set up
    Random.seed!(seed)

    n = Int(floor(m/10))
    
    B = rand(m,n)
    B = B'*B
    @assert isposdef(B)
    D = MvNormal(randn(n),B)
    
    A = rand(D, m)'
    @assert rank(A) == n 

    N = floor(1.5*n)
    u = floor(N/3)
    ub = rand(1.0:u, m)
        
    return A, n, N, ub 
end

"""
Build Probability Simplex BLMO for Boscia
"""
function build_blmo(m, N, ub)
    simplex_lmo = Boscia.ProbabilitySimplexSimpleBLMO(N)
    blmo = Boscia.ManagedBoundedLMO(simplex_lmo, fill(0.0, m), ub, collect(1:m), m)
    return blmo
end

"""
Build function and the gradient for the A-criterion. 
There is the option to build a safe version of the objective and gradient. If x does not
satisfy the domain oracle, infinity is returned. 
If one builds the unsafe version, a FrankWolfe line search must be chosen that can take 
a domain oracle as an input like Secant or MonotonicGenericStepSize. 
"""
function build_a_criterion(A; μ=1e-4, build_safe=false)
    m, n = size(A) 
    a = m
    domain_oracle = build_domain_oracle(A, n)

    function f_a(x)
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X)
        U = cholesky(X)
        X_inv = U \ I
        return LinearAlgebra.tr(X_inv)/a 
    end

    function grad_a!(storage, x)
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X*X)
        F = cholesky(X)
        for i in 1:length(x)
            storage[i] = LinearAlgebra.tr(- (F \ A[i,:]) * transpose(A[i,:]))/a
        end
        return storage 
    end 

    function f_a_safe(x)
        if !domain_oracle(x)
            return Inf
        end
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X)
        X_inv = LinearAlgebra.inv(X)
        return LinearAlgebra.tr(X_inv)/a 
    end

    function grad_a_safe!(storage, x)
        if !domain_oracle(x)
            return fill(Inf, length(x))        
        end
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X*X)
        F = cholesky(X)
        for i in 1:length(x)
            storage[i] = LinearAlgebra.tr(- (F \ A[i,:]) * transpose(A[i,:]))/a
        end
        return storage
    end

    if build_safe
        return f_a_safe, grad_a_safe!
    end

    return f_a, grad_a!
end

"""
Build function and gradient for the D-criterion.
There is the option to build a safe version of the objective and gradient. If x does not
satisfy the domain oracle, infinity is returned. 
If one builds the unsafe version, a FrankWolfe line search must be chosen that can take 
a domain oracle as an input like Secant or MonotonicGenericStepSize. 
"""
function build_d_criterion(A; μ =0.0, build_safe=false)
    m, n = size(A)
    a = m
    domain_oracle = build_domain_oracle(A, n)

    function f_d(x)
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X)
        return float(-log(det(X))/a)
    end

    function grad_d!(storage, x)
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X)
        F = cholesky(X) 
        for i in 1:length(x)        
            storage[i] = 1/a * LinearAlgebra.tr(-(F \ A[i,:] )*transpose(A[i,:])) * 1/γ
        end
        return storage
    end

    function f_d_safe(x)
        if !domain_oracle(x)
            return Inf
        end
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X)
        return float(-log(det(X))/a)
    end

    function grad_d_safe!(storage, x)
        if !domain_oracle(x)
            return fill(Inf, length(x))
        end
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X)
        F = cholesky(X) 
        for i in 1:length(x)        
            storage[i] = 1/a * LinearAlgebra.tr(-(F \ A[i,:] )*transpose(A[i,:])) * 1/γ
        end
        # https://stackoverflow.com/questions/46417005/exclude-elements-of-array-based-on-index-julia
        return storage
    end

    if build_safe
        return f_d_safe, grad_d_safe!
    end

    return f_d, grad_d!
end

"""
Find n linearly independent rows of A to build the starting point.
"""
function linearly_independent_rows(A)
    S = []
    m, n = size(A)
    for i in 1:m
        S_i= vcat(S, i)
        if rank(A[S_i,:])==length(S_i)
            S=S_i
        end
        if length(S) == n # we only n linearly independent points
            return S
        end
    end 
    return S # then x= zeros(m) and x[S] = 1
end

function add_to_min(x, u)
    perm = sortperm(x)
    j = findfirst(x->x != 0, x[perm])
    
    for i in j:length(x)
        if x[perm[i]] < u[perm[i]]
            x[perm[i]] += 1
            break
        else
            continue
        end
    end
    return x
end

function remove_from_max(x)
    perm = sortperm(x, rev = true)
    j = findlast(x->x != 0, x[perm])
    
    for i in 1:j
        if x[perm[i]] > 1
            x[perm[i]] -= 1
            break
        else
            continue
        end
    end
    return x
end

"""
Build start point used in FrankWolfe and Boscia for the A-Optimal and D-Optimal Design Problem.
The functions are self concordant and so not every point in the feasible region
is in the domain of f and grad!.
""" 
function build_start_point(A, N, ub)
    # Get n linearly independent rows of A
    m, n = size(A)
    S = linearly_independent_rows(A)
    @assert length(S) == n
    
    x = zeros(m)
    E = []
    V = Vector{Float64}[]

    while !isempty(setdiff(S, findall(x-> !(iszero(x)),x)))
        v = zeros(m)
        while sum(v) < N
            idx = isempty(setdiff(S, findall(x-> !(iszero(x)),v))) ? rand(setdiff(collect(1:m), S)) : rand(setdiff(S, findall(x-> !(iszero(x)),v)))
            if !isapprox(v[idx], 0.0)
                @debug "Index $(idx) already picked"
                continue
            end
            v[idx] = min(ub[idx], N - sum(v))
            push!(E, idx)
        end
        push!(V,v)
        x = sum(V .* 1/length(V)) 
    end
    unique!(V)
    a = length(V)
    x = sum(V .* 1/a)
    active_set= FrankWolfe.ActiveSet(fill(1/a, a), V, x)

    return x, active_set, S
end

"""
Create first incumbent for Boscia and custom BB in a greedy fashion.
"""
function greedy_incumbent(A, N, ub)
    # Get n linearly independent rows of A
    m, n = size(A)
    S = linearly_independent_rows(A)
    @assert length(S) == n

    # set entries to their upper bound
    x = zeros(m)
    x[S] .= ub[S]

    if isapprox(sum(x), N; atol=1e-4, rtol=1e-2)
        return x
    elseif sum(x) > N
        while sum(x) > N
            remove_from_max(x)
        end
    elseif sum(x) < N
        S1 = S
        while sum(x) < N
            jdx = rand(setdiff(collect(1:m), S1))
            x[jdx] = min(N-sum(x), ub[jdx])
            push!(S1,jdx)
            sort!(S1)
        end
    end
    @assert isapprox(sum(x), N; atol=1e-4, rtol=1e-2)
    @assert sum(ub - x .>= 0) == m 
    return x
end

"""
Check if given point is in the domain of f, i.e. X = transpose(A) * diagm(x) * A 
positive definite.
"""
function build_domain_oracle(A, n)
    return function domain_oracle(x)
        S = findall(x-> !iszero(x),x)
        #@show rank(A[S,:]) == n
        return length(S) >= n && rank(A[S,:]) == n 
    end
end