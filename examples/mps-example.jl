using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
using StableRNGs

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)


# A MIPLIB instance: 22433
# https://miplib.zib.de/instance_details_22433.html
# Objective function: Minimize the distance to randomely picked vertices
# Number of variables   429
# Number of integers      0
# Number of binaries    231
# Number of constraints 198

src = MOI.FileFormats.Model(filename="22433.mps")
MOI.read_from_file(src, joinpath(@__DIR__, "mps-examples/mps-files/22433.mps"))

o = SCIP.Optimizer()
MOI.copy_to(o, src)
MOI.set(o, MOI.Silent(), true)
n = MOI.get(o, MOI.NumberOfVariables())
lmo = FrankWolfe.MathOptLMO(o)

#trick to push the optimum towards the interior
const vs = [FrankWolfe.compute_extreme_point(lmo, randn(rng, n)) for _ in 1:20]
# done to avoid one vertex being systematically selected
unique!(vs)
filter!(vs) do v
    return v[end] != 21477.0
end

@assert !isempty(vs)
const b_mps = randn(rng, n)

function f(x)
    r = dot(b_mps, x)
    for v in vs
        r += 1 / 2 * norm(x - v)^2
    end
    return r
end

function grad!(storage, x)
    mul!(storage, length(vs) * I, x)
    storage .+= b_mps
    for v in vs
        @. storage -= v
    end
end

@testset "MPS 22433 instance" begin
    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true)
    @test f(x) <= f(result[:raw_solution])
end
