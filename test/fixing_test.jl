using LinearAlgebra
using Random
using Boscia
using SCIP
import FrankWolfe
import MathOptInterface as MOI
using Test

Random.seed!(0)

n = 5
p = 4
k = 3
X = randn(n, p)
W = randn(n, k)

n, p = size(X)
n, k = size(W)
A = (I - W*inv(W'*W)*W')^2
B = X*X'

function f(x)
    n = isqrt(length(x))
    P = reshape(x, n, n)
    s = 1/2 * sum(eachindex(x)) do j
        x[j]^2 - x[j]
    end
    return tr(P*B*P'*A) + n * s
end
function grad!(storage, x)
    P = reshape(x, n, n)
    storage .= (2*A*P*B)[:]
    storage .+= 
    return storage
end

function build_birkhoff_lmo(n::Int)
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    P = reshape(MOI.add_variables(o, n^2), n, n)
    MOI.add_constraint.(o, P, MOI.ZeroOne())
    # doubly stochastic constraints
    MOI.add_constraint.(
        o,
        vec(sum(P; dims=1, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
        MOI.EqualTo(1.0),
    )
    MOI.add_constraint.(
        o,
        vec(sum(P; dims=2, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
        MOI.EqualTo(1.0),
    )
    return FrankWolfe.MathOptLMO(o)
end

x_tight, res_tight = Boscia.solve(f, grad!,  build_birkhoff_lmo(n); verbose=true, print_iter=1, dual_tightening=true)
x_notight, res_notight = Boscia.solve(f, grad!, lmo; verbose=true, print_iter=1, dual_tightening=false)

@test f(x_notight) ≈ f(x_tight)

lmo = build_birkhoff_lmo(n)
x0 = FrankWolfe.compute_extreme_point(lmo, randn(n^2))
x, _, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, x0, verbose=true)

gradient = similar(x0)

grad!(gradient, x)

rhs = 1.6238670348873667

lmo_ref = FrankWolfe.BirkhoffPolytopeLMO()
x, _, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo_ref, x0, verbose=true)

x[16]


Random.seed!(30)
lmo = build_birkhoff_lmo(n)
for _ in 1:1000
    x1 = FrankWolfe.compute_extreme_point(lmo, randn(n^2))
    x2 = FrankWolfe.compute_extreme_point(lmo, randn(n^2))
    for a in 0.1:0.1:0.9
        if ! (a * f(x1) + (1-a) * f(x2) >= f(a * x1 + (1-a) * x2) - 100 * eps())
            println(x1)
            println(x2)
            @show(a)
            error()
        end
    end
end


x1 = x_bk = [0.8341698427457216, 0.0, 0.0, 0.0, 0.16583015725427835, 0.0, 0.08712774717235677, 0.0, 0.9128722528276432, 0.0, 0.0, 0.6385446097251585, 0.0, 0.08712774717235677, 0.27432764310248475, 0.0, 0.27432764310248475, 0.42007048276220466, 0.0, 0.3056018741353106, 0.16583015725427835, 0.0, 0.5799295172377954, 0.0, 0.2542403255079263]
x2 = XREF = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
for a in 0.1:0.1:0.9
    if ! (a * f(x1) + (1-a) * f(x2) >= f(a * x1 + (1-a) * x2) - 100 * eps())
        println(x1)
        println(x2)
        @show(a)
        error()
    end
end
