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
    P = reshape(x, n, n)
    return tr(P*B*P'*A)
end
function grad!(storage, x)
    P = reshape(x, n, n)
    storage .= (2*A*P*B)[:]
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

lmo = build_birkhoff_lmo(n)

x_notight, res_notight = Boscia.solve(f, grad!, lmo; verbose=true, print_iter=1, dual_tightening=false)
x_tight, res_tight = Boscia.solve(f, grad!, lmo; verbose=true, print_iter=1, dual_tightening=true)

@test f(x_notight) ≈ f(x_tight)
