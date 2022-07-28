using LinearAlgebra
using Distributions
"""
Testing of the interface function branch_wolfe
"""
n = 20
diffi = Random.rand(Bool,n)*0.6.+0.3

@testset "Interface - norm hyperbox" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
        MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
    end
    lmo = FrankWolfe.MathOptLMO(o)

    function f(x)
        return sum(0.5*(x.-diffi).^2)
    end
    function grad!(storage, x)
        @. storage = x-diffi
    end

    x, _, dual_gap = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = false)

    @show dual_gap
    @test x == round.(diffi)
end


# min h(sqrt(y' * M * y)) - r' * y
# s.t. a' * y <= b 
#           y >= 0
#           y_i in Z for i in I

n = 10
const ri = 10 * rand(n)
const ai = rand(n)
const Ωi = 3 * rand(Float64)
const bi = sum(ai)
Ai = randn(n,n)
Ai = Ai' * Ai
const Mi =  (Ai + Ai')/2
@assert isposdef(Mi)

@testset "Interface - Buchheim" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o,n)
    I =  rand(1:n, Int64(floor(n/2)))  #collect(1:n)
    for i in 1:n
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        if i in I
            MOI.add_constraint(o, x[i], MOI.Integer())
        end
    end 
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai,x), 0.0), MOI.LessThan(bi))
    #MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai,x), 0.0), MOI.GreaterThan(minimum(ai)))
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n),x), 0.0), MOI.GreaterThan(1.0))
    lmo = FrankWolfe.MathOptLMO(o)

    function h(x)
        return Ωi
    end
    function f(x)
        return h(x) * (x' * Mi * x) - ri' * x
    end
    function grad!(storage, x)
        storage.= 2 * Mi * x - ri
        return storage
    end

    x, _, dual_gap = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = false)
    @show x
    @show dual_gap
    @test sum(ai'* x) <= bi + eps()

    # Run without binary constraints
    q = SCIP.Optimizer()
    MOI.set(q, MOI.Silent(), true)
    MOI.empty!(q)
    z = MOI.add_variables(q,n)
    I =  rand(1:n, Int64(floor(n/2)))  #collect(1:n)
    for i in 1:n
        MOI.add_constraint(q, z[i], MOI.GreaterThan(0.0))
    end 
    MOI.add_constraint(q, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai,z), 0.0), MOI.LessThan(bi))
    #MOI.add_constraint(q, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai,x), 0.0), MOI.GreaterThan(minimum(ai)))
    MOI.add_constraint(q, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n),z), 0.0), MOI.GreaterThan(1.0))
    lmo = FrankWolfe.MathOptLMO(q)
    direction = Vector{Float64}(undef,n)
    Random.rand!(direction)
    v = compute_extreme_point(lmo, direction)
    active_set = FrankWolfe.ActiveSet([(1.0, v)])
    z,_,primal,dual_gap,_ , active_set = FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo,
        active_set,
    ) 
    @show z
    @show f(z)
    @show dual_gap
    @show f(x)-f(z) 
end


# Sparse Poisson regression
# min_{w, b, z} ∑_i exp(w x_i + b) - y_i (w x_i + b) + α norm(w)^2
# s.t. -N z_i <= w_i <= N z_i
# b ∈ [-N, N]
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p

n=30
p = n

# underlying true weights
const ws = rand(Float64, p) 
# set 50 entries to 0
for _ in 1:20
    ws[rand(1:p)] = 0
end
const bs = rand(Float64) 
const Xs = randn(Float64, n, p) 
const ys = map(1:n) do idx
    a = dot(Xs[idx,:], ws) + bs
    rand(Distributions.Poisson(exp(a)))
end
Ns = 5.0

@testset "Interface - sparse regression" begin
k = 10
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    w = MOI.add_variables(o, p)
    z = MOI.add_variables(o, p)
    b = MOI.add_variable(o)
    for i in 1:p
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())
    end
    for i in 1:p
        MOI.add_constraint(o, -Ns * z[i]- w[i], MOI.LessThan(0.0))
        MOI.add_constraint(o, Ns * z[i]- w[i], MOI.GreaterThan(0.0))
        # Indicator: z[i] = 1 => -N <= w[i] <= N
        gl = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),],
            [0.0, 0.0], )
        gg = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),],
            [0.0, 0.0], )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(Ns)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-Ns))) 
    end
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(o, sum(z, init=0.0), MOI.GreaterThan(1.0))
    MOI.add_constraint(o, b, MOI.LessThan(Ns))
    MOI.add_constraint(o, b, MOI.GreaterThan(-Ns))
    lmo = FrankWolfe.MathOptLMO(o)

    α = 1.3
    function f(θ)
        w = @view(θ[1:p])
        b = θ[end]
        s = sum(1:n) do i
            a = dot(ws, Xs[:,i]) + b
            1/n * (exp(a) - ys[i] * a)
        end
        s + α * norm(ws)^2
    end
    function grad!(storage, θ)
        w = @view(θ[1:p])
        b = θ[end]
        storage[1:p] .= 2α .* ws
        storage[p+1:2p] .= 0
        storage[end] = 0
        for i in 1:n
            xi = @view(Xs[:,i])
            a = dot(w, xi) + b
            storage[1:p] .+= 1/n * xi * exp(a)
            storage[1:p] .-= 1/n * ys[i] * xi
            storage[end] += 1/n * (exp(a) - ys[i])
        end
        storage ./= norm(storage)
        return storage
    end

    x, _, dual_gap = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = false)
    @show x
    @show dual_gap
    @show f(x)
    @test sum(x[p+1:2p]) <= k

    # Run without binary constraints
    k = p
    q = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    w = MOI.add_variables(q, p)
    z = MOI.add_variables(q, p)
    b = MOI.add_variable(q)
    MOI.set(q, MOI.Silent(), true)
    for i in 1:p
        MOI.add_constraint(q, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(q, z[i], MOI.LessThan(1.0))
       # MOI.add_constraint(o, z[i], MOI.ZeroOne())
    end
    for i in 1:p
        MOI.add_constraint(q, -Ns * z[i]- w[i], MOI.LessThan(0.0))
        MOI.add_constraint(q, Ns * z[i]- w[i], MOI.GreaterThan(0.0))
    end
    MOI.add_constraint(q, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(q, sum(z, init=0.0), MOI.GreaterThan(1.0))
    MOI.add_constraint(q, b, MOI.LessThan(Ns))
    MOI.add_constraint(q, b, MOI.GreaterThan(-Ns))
    lmo = FrankWolfe.MathOptLMO(q)
    direction = Vector{Float64}(undef,2p+1)
    Random.rand!(direction)
    v = compute_extreme_point(lmo, direction)
    active_set = FrankWolfe.ActiveSet([(1.0, v)])
    y,_,primal,dual_gap,_ , active_set = FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo,
        active_set,
    ) 
    @show y
    @show f(y)
    @show dual_gap
    @show (f(x)-f(y)) 
end