using Boscia
using JuMP
using LinearAlgebra
using Random
using FrankWolfe
using StableRNGs
using Test
using HiGHS

println("\nMode Tests")

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

n = 300
Q = Symmetric(randn(rng, n, n))
@show isposdef(Q)
const q = -abs.(diag(Q)) .- 1.0
const D = abs.(randn(rng, n, n))
const c = vec(sum(D, dims=2))

@testset "Non convex QUBP" begin
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:n], Bin)
    I = collect(1:n) 
    @constraint(model, x[1:n] >= 0)
    @constraint(model, D * x <= c)
    lmo = FrankWolfe.MathOptLMO(model.moi_backend)

    function f(x)
        return 1 / 2 * dot(x, Q, x) + dot(q, x)
    end
    function grad!(storage, x)
        mul!(storage, Q, x, 1, 0)
        storage .+= q
        return storage
    end

    settings = Boscia.create_default_settings(mode=Boscia.HEURISTIC_MODE)
    settings.branch_and_bound[:verbose] = true
    settings.branch_and_bound[:print_iter] = 1
    settings.branch_and_bound[:node_limit] = 500
    x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)
    @test result[:status] in [Boscia.TIME_LIMIT_REACHED, Boscia.NODE_LIMIT_REACHED, Boscia.USER_STOP, Boscia.OPT_TREE_EMPTY]
end