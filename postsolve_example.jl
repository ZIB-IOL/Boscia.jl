using Boscia
using FrankWolfe
using StableRNGs
using SCIP
using Distributions
using LinearAlgebra
using MathOptInterface
const MOI = MathOptInterface


### Approx Planted Point

seed = 1234
n = 20 #50
rng = StableRNG(seed)


diffi = rand(rng, Bool, n) * 0.6 .+ 0.3
    
function f(x)
    return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
end
function grad!(storage, x)
    @. storage = x - diffi
end
int_vars = unique!(rand(collect(1:n), Int(floor(n / 2))))

solution = copy(diffi)
solution[int_vars] .= round.(solution[int_vars])
@show diffi
@show solution
@show f(solution)

# Cube Simple LMO
lbs = zeros(n)
ubs = ones(n)
lmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)
Boscia.solve(f, grad!, lmo, lbs[int_vars], ubs[int_vars], int_vars, n; variant=Boscia.BPCG(), verbose=true)

# MOI
o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
MOI.empty!(o)
x = MOI.add_variables(o, n)
for xi in x
    MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
    MOI.add_constraint(o, xi, MOI.LessThan(1.0))
    if xi.value in int_vars
        MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
    end
end
lmo = Boscia.MathOptBLMO(o)
x, _, _ = Boscia.solve(f, grad!, lmo; variant=Boscia.BPCG(), verbose=true) 



### Sparse Regresssion

"""
Builds data according to 'Sparse Regression' example in Boscia.jl

num vars: 2p = 2(5n) = 10n
"""
function build_sparse_reg(; n::T=20, seed::T=1234) where T<:Integer
    rng = StableRNG(seed)

    # define constants 
    p = 5 * n;
    k = ceil(n/5);
    lambda_0 = rand(rng, Float64)
    lambda_2 = 10 * rand(rng, Float64)
    A = rand(rng, n, p)
    y = rand(rng, n)
    M = 2 * var(A)

    # init optimizer for LMO
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, 2p)

    for i in p+1:2p
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, x[i], MOI.ZeroOne())
    end

    for i in 1:p
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, M], [x[i], x[i+p]]), 0.0),
            MOI.GreaterThan(0.0),
        )
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -M], [x[i], x[i+p]]), 0.0),
            MOI.LessThan(0.0),
        )
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p), x[p+1:2p]), 0.0),
        MOI.LessThan(k),
    )
    lmo = FrankWolfe.MathOptLMO(o)

    function f(x)
        xv = @view(x[1:p])
        return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
    end

    function grad!(storage, x)
        storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
        storage[p+1:2p] .= lambda_0
        return storage
    end

    return f, grad!, lmo
end

println("SparseReg example")
args2 = build_sparse_reg(n=10)
#Boscia.solve(args2...; variant=Boscia.Blended(), verbose=true, print_iter=1)
Boscia.solve(args2...; variant=Boscia.BPCG(), verbose=true, print_iter=1)



### Portfolio

"""
Builds data for portfolio example for Boscia.

num vars: n

Reference: https://github.com/ZIB-IOL/Boscia.jl/blob/main/examples/portfolio.jl 
"""
function build_portfolio(; n=30, seed=1234)
    rng = StableRNG(seed)

    ri = rand(rng, n)
    ai = rand(rng, n)
    oi = rand(rng, Float64)
    bi = sum(ai)

    Ai = randn(rng, n, n)
    Ai = Ai' * Ai

    Mi = (Ai + Ai') / 2
    
    @assert isposdef(Mi)

    # lmo
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)

    # x positive Integer
    for i in 1:n
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.Integer())
    end

    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai, x), 0.0),
        MOI.LessThan(bi),
    )
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.GreaterThan(1.0),
    )
    
    lmo = FrankWolfe.MathOptLMO(o)

    # objective
    function f(x)
        return 1/2 * oi * dot(x, Mi, x) - dot(ri, x)
    end

    function grad!(storage, x)
        mul!(storage, Mi, x, oi, 0)
        storage .-= ri
        return storage
    end

    return f, grad!, lmo
end

# Portfolio
args3 = build_portfolio(n=30, seed=1234)
#Boscia.solve(args3...; variant=Boscia.Blended(), verbose=true)
Boscia.solve(args3...; variant=Boscia.BPCG(), verbose=true)

