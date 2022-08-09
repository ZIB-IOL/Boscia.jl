module BranchWolfe

using FrankWolfe
using Random
using SCIP
import MathOptInterface
import Bonobo
using Printf
using Dates
const MOI = MathOptInterface
const MOIU = MOI.Utilities

import MathOptSetDistances
const MOD = MathOptSetDistances

include("time_tracking_lmo.jl")
include("bounds.jl")
include("node.jl")
include("callbacks.jl")
include("problem.jl")
include("infeasible_pairwise.jl")
include("heuristics.jl")
include("strong_branching.jl")
include("utilities.jl")
include("interface.jl")

using Statistics
using Random
using Distributions
using LinearAlgebra
import HiGHS

seed = 0x10d1634d9c36da46
Random.seed!(seed)

n = 20
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

# TODO: document better

    k = 10
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    w = MOI.add_variables(o, p)
    z = MOI.add_variables(o, p)
    b = MOI.add_variable(o)
    for i in 1:p
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())
    end
    for i in 1:p
        MOI.add_constraint(o, Ns * z[i] + w[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, -Ns * z[i] + w[i], MOI.LessThan(0.0))
        # Indicator: z[i] = 1 => -N <= w[i] <= N
        #=gl = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),],
            [0.0, 0.0], )
        gg = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),],
            [0.0, 0.0], )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(Ns)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-Ns))) =#
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
            a = dot(w, Xs[:,i]) + b
            1/n * (exp(a) - ys[i] * a)
        end
        s + α * norm(w)^2
    end
    function grad!(storage, θ)
        w = @view(θ[1:p])
        b = θ[end]
        storage[1:p] .= 2α .* w
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

    x, _,result = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = true)
    @show x
    @show f(x)
    @test f(x) <= f(result[:raw_solution])
    @test sum(x[p+1:2p]) <= k 

end # module
