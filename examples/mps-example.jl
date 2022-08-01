using BranchWolfe
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface

# Example reading a polytope from a MIPLIB instance

src = MOI.FileFormats.Model(filename="22433.mps")
MOI.read_from_file(src, joinpath(@__DIR__, "22433.mps"))

o = SCIP.Optimizer()
MOI.copy_to(o, src)
MOI.set(o, MOI.Silent(), true)
n = MOI.get(o, MOI.NumberOfVariables())

lmo = FrankWolfe.MathOptLMO(o)

#trick to push the optimum towards the interior
const vs = [FrankWolfe.compute_extreme_point(lmo, randn(n)) for _ in 1:10]
unique!(vs)
@assert !isempty(vs)

b_mps = rand(n)

function f(x)
    r = dot(b_mps, x)
    for v in vs
        r += 1/2 * norm(x - v)^2
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

@testset "MPS instance" begin
    x, _, result,_ = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = true)
end
