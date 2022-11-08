using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
import Ipopt


# A MIPLIB instance: pg5_34
# https://miplib.zib.de/instance_details_pg5_34.html
# Objective function: Minimize the distance to randomely picked vertices
# Number of variables  2600
# Number of integers      0
# Number of binaries    100
# Number of constraints 225

seed = rand(UInt64)
@show seed
Random.seed!(seed)

src = MOI.FileFormats.Model(filename="pg5_34.mps")
MOI.read_from_file(src, joinpath(@__DIR__, "mps-files/pg5_34.mps"))

o = SCIP.Optimizer()
MOI.copy_to(o, src)
MOI.set(o, MOI.Silent(), true)
n = MOI.get(o, MOI.NumberOfVariables())
lmo = FrankWolfe.MathOptLMO(o)

#trick to push the optimum towards the interior
const vs = [FrankWolfe.compute_extreme_point(lmo, randn(n)) for _ in 1:5]
# done to avoid one vertex being systematically selected
unique!(vs)

@assert !isempty(vs)
const b_mps = randn(n)

const max_norm = maximum(norm.(vs))

function f(x)
    r = dot(b_mps, x)
    for v in vs
        r += 1 / (2 * max_norm) * norm(x - v)^2
    end
    return r
end

function grad!(storage, x)
    mul!(storage, length(vs)/max_norm * I, x)
    storage .+= b_mps
    for v in vs
        @. storage -= 1/max_norm * v
    end
end

@testset "MPS pg5_34 instance" begin
    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, print_iter = 10, fw_epsilon = 1e-1, min_node_fw_epsilon = 1e-3, time_limit=3000)
    @test f(x) <= f(result[:raw_solution])
end


# Relaxed version
filtered_src = MOI.Utilities.ModelFilter(o) do item
    if item isa Tuple
        (_, S) = item
        if S <: Union{MOI.Indicator,MOI.Integer,MOI.ZeroOne}
            return false
        end
    end
    return !(item isa MOI.ConstraintIndex{<:Any,<:Union{MOI.ZeroOne,MOI.Integer,MOI.Indicator}})
end
ipopt_optimizer = MOI.Bridges.full_bridge_optimizer(Ipopt.Optimizer(), Float64)
index_map = MOI.copy_to(ipopt_optimizer, filtered_src)
# sanity check, otherwise the functions need permuted indices
for (v1, v2) in index_map
    if v1 isa MOI.VariableIndex
        @assert v1 == v2
    end
end
