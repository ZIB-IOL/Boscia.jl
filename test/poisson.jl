using Statistics
using Random
using Distributions
using LinearAlgebra
import HiGHS
import SCIP
import MathOptInterface
const MOI = MathOptInterface
import Boscia
import Bonobo
using FrankWolfe
using Dates
# Sparse Poisson regression
# min_{w, b, z} ∑_i exp(w x_i + b) - y_i (w x_i + b) + α norm(w)^2
# s.t. -N z_i <= w_i <= N z_i
# b ∈ [-N, N]
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p


n0 = 30
p = n0

# underlying true weights
const w0 = rand(Float64, p) 
# set 50 entries to 0
for _ in 1:20
    w0[rand(1:p)] = 0
end
const b0 = rand(Float64) 
const X0 = rand(Float64, n0, p) 
const y0 = map(1:n0) do idx
    a = dot(X0[idx, :], w0) + b0
    return rand(Distributions.Poisson(exp(a)))
end
N = 1.0

@testset "Poisson sparse regression" begin
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
        MOI.add_constraint(o, -N * z[i] - w[i], MOI.LessThan(0.0))
        MOI.add_constraint(o, N * z[i] - w[i], MOI.GreaterThan(0.0))
    end
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(o, b, MOI.LessThan(N))
    MOI.add_constraint(o, b, MOI.GreaterThan(-N))
    lmo = FrankWolfe.MathOptLMO(o)
    
    α = 1.3
    function f(θ)
        w = @view(θ[1:p])
        b = θ[end]
        s = sum(1:n0) do i
            a = dot(w, X0[:, i]) + b
            return 1 / n0 * (exp(a) - y0[i] * a)
        end
        return s + α * norm(w)^2
    end
    function grad!(storage, θ)
        w = @view(θ[1:p])
        b = θ[end]
        storage[1:p] .= 2α .* w
        storage[p+1:2p] .= 0
        storage[end] = 0
        for i in 1:n0
            xi = @view(X0[:, i])
            a = dot(w, xi) + b
            storage[1:p] .+= 1 / n0 * xi * exp(a)
            storage[1:p] .-= 1 / n0 * y0[i] * xi
            storage[end] += 1 / n0 * (exp(a) - y0[i])
        end
        storage ./= norm(storage)
        return storage
    end

    x, _, result = Boscia.solve(f, grad!, lmo, verbose = true, time_limit=500)

    @test f(x) <= f(result[:raw_solution]) + 1e-6
    @test sum(x[p+1:2p]) <= k
end

@testset "Hybrid branching poisson sparse regression" begin
    k = 5
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
        MOI.add_constraint(o, -N * z[i] - w[i], MOI.LessThan(0.0))
        MOI.add_constraint(o, N * z[i] - w[i], MOI.GreaterThan(0.0))
        # Indicator: z[i] = 1 => -N <= w[i] <= N
        gl = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),
            ],
            [0.0, 0.0],
        )
        gg = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),
            ],
            [0.0, 0.0],
        )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(N)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-N)))
    end
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(o, b, MOI.LessThan(N))
    MOI.add_constraint(o, b, MOI.GreaterThan(-N))
    lmo = FrankWolfe.MathOptLMO(o)
    α = 1.3
    function f(θ)
        w = @view(θ[1:p])
        b = θ[end]
        s = sum(1:n0) do i
            a = dot(w, X0[:, i]) + b
            return 1 / n0 * (exp(a) - y0[i] * a)
        end
        return s + α * norm(w)^2
    end
    function grad!(storage, θ)
        w = @view(θ[1:p])
        b = θ[end]
        storage[1:p] .= 2α .* w
        storage[p+1:2p] .= 0
        storage[end] = 0
        for i in 1:n0
            xi = @view(X0[:, i])
            a = dot(w, xi) + b
            storage[1:p] .+= 1 / n0 * xi * exp(a)
            storage[1:p] .-= 1 / n0 * y0[i] * xi
            storage[end] += 1 / n0 * (exp(a) - y0[i])
        end
        storage ./= norm(storage)
        return storage
    end

    branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, HiGHS.Optimizer())
    MOI.set(branching_strategy.optimizer, MOI.Silent(), true)
    
    x_,result = Boscia.solve(f, grad!, lmo, verbose = true, branching_strategy = branching_strategy)
    @test sum(x[p+1:2p]) <= k
    @test f(x) <= f(result[:raw_solution]) + 1e-6
    @test sum(x[p+1:2p]) <= k
end

n0g = 20
pg = n0g

# underlying true weights
const w0g = 2 * rand(Float64, pg) .- 1
# set 50 entries to 0
for _ in 1:15
    w0g[rand(1:pg)] = 0
end
const b0g = 2 * rand(Float64) - 1
const X0g = 2 * rand(Float64, n0g, pg) .- 1
const y0g = map(1:n0g) do idx
    a = dot(X0g[idx, :], w0g) + b0g
    return rand(Distributions.Poisson(exp(a)))
end
Ng = 5.0

k = 10
group_size = convert(Int64, floor(pg / k))
groups = []
for i in 1:(k-1)
    push!(groups, ((i-1)*group_size+1):(i*group_size))
end
push!(groups, ((k-1)*group_size+1):pg)

@testset "Sparse Group Poisson" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    w = MOI.add_variables(o, pg)
    z = MOI.add_variables(o, pg)
    b = MOI.add_variable(o)
    for i in 1:pg
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())
    end
    for i in 1:pg
        MOI.add_constraint(o, -Ng * z[i] - w[i], MOI.LessThan(0.0))
        MOI.add_constraint(o, Ng * z[i] - w[i], MOI.GreaterThan(0.0))
        # Indicator: z[i] = 1 => -Ng <= w[i] <= Ng
        gl = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),
            ],
            [0.0, 0.0],
        )
        gg = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),
            ],
            [0.0, 0.0],
        )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(Ng)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-Ng)))
    end
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(o, b, MOI.LessThan(Ng))
    MOI.add_constraint(o, b, MOI.GreaterThan(-Ng))
    for i in 1:k
        #MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(group_size),x[groups[i]]), 0.0), MOI.GreaterThan(1.0))
        MOI.add_constraint(o, sum(z[groups[i]], init=0.0), MOI.GreaterThan(1.0))
    end
    lmo = FrankWolfe.MathOptLMO(o)

    α = 1.3
    function f(θ)
        w = @view(θ[1:pg])
        b = θ[end]
        s = sum(1:n0g) do i
            a = dot(w, X0g[:, i]) + b
            return 1 / n0g * (exp(a) - y0g[i] * a)
        end
        return s + α * norm(w)^2
    end
    function grad!(storage, θ)
        w = @view(θ[1:pg])
        b = θ[end]
        storage[1:pg] .= 2α .* w
        storage[pg+1:2pg] .= 0
        storage[end] = 0
        for i in 1:n0g
            xi = @view(X0g[:, i])
            a = dot(w, xi) + b
            storage[1:pg] .+= 1 / n0g * xi * exp(a)
            storage[1:pg] .-= 1 / n0g * y0g[i] * xi
            storage[end] += 1 / n0g * (exp(a) - y0g[i])
        end
        storage ./= norm(storage)
        return storage
    end
   
    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true)
    #@show x
    @show result[:raw_solution]
    @test f(x) <= f(result[:raw_solution]) + 1e-6
    @test sum(x[p+1:2p]) <= k
end


@testset "Strong branching sparse group poisson" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    w = MOI.add_variables(o, pg)
    z = MOI.add_variables(o, pg)
    b = MOI.add_variable(o)
    for i in 1:pg
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())
    end
    for i in 1:pg
        MOI.add_constraint(o, -Ng * z[i] - w[i], MOI.LessThan(0.0))
        MOI.add_constraint(o, Ng * z[i] - w[i], MOI.GreaterThan(0.0))
        # Indicator: z[i] = 1 => -Ng <= w[i] <= Ng
        gl = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),
            ],
            [0.0, 0.0],
        )
        gg = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),
            ],
            [0.0, 0.0],
        )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(Ng)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-Ng)))
    end
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(o, b, MOI.LessThan(Ng))
    MOI.add_constraint(o, b, MOI.GreaterThan(-Ng))
    for i in 1:k
        #MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(group_size),x[groups[i]]), 0.0), MOI.GreaterThan(1.0))
        MOI.add_constraint(o, sum(z[groups[i]], init=0.0), MOI.GreaterThan(1.0))
    end
    lmo = FrankWolfe.MathOptLMO(o)
    α = 1.3
    function f(θ)
        w = @view(θ[1:pg])
        b = θ[end]
        s = sum(1:n0g) do i
            a = dot(w, X0g[:, i]) + b
            return 1 / n0g * (exp(a) - y0g[i] * a)
        end
        return s + α * norm(w)^2
    end
    function grad!(storage, θ)
        w = @view(θ[1:pg])
        b = θ[end]
        storage[1:pg] .= 2α .* w
        storage[pg+1:2pg] .= 0
        storage[end] = 0
        for i in 1:n0g
            xi = @view(X0g[:, i])
            a = dot(w, xi) + b
            storage[1:pg] .+= 1 / n0g * xi * exp(a)
            storage[1:pg] .-= 1 / n0g * y0g[i] * xi
            storage[end] += 1 / n0g * (exp(a) - y0g[i])
        end
        storage ./= norm(storage)
        return storage
    end

    branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, HiGHS.Optimizer())
    MOI.set(branching_strategy.optimizer, MOI.Silent(), true)

    x, _, result_strong_branching =
        Boscia.solve(f, grad!, lmo, verbose=true, branching_strategy=branching_strategy)

        @test f(x) <= f(result[:raw_solution]) + 1e-6
        @test sum(x[p+1:2p]) <= k
end
