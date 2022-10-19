using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
import Ipopt
using ProfileView


seed = rand(UInt64)
#seed = 0xab30b963fc4f3488
@show seed
Random.seed!(seed)

# Example reading a polytope from a MIPLIB instance
# fastxgemm-n2r6s0t2
# ran14x18-disj-8
# pg5_34
src = MOI.FileFormats.Model(filename="ran14x18-disj-8.mps")
MOI.read_from_file(src, joinpath(@__DIR__, "mps-files/ran14x18-disj-8.mps"))

o = SCIP.Optimizer()
MOI.copy_to(o, src)
MOI.set(o, MOI.Silent(), true)
n = MOI.get(o, MOI.NumberOfVariables())

lmo = FrankWolfe.MathOptLMO(o)

#trick to push the optimum towards the interior
const vs = [FrankWolfe.compute_extreme_point(lmo, randn(n)) for _ in 1:5]
# done to avoid one vertex being systematically selected
unique!(vs)
#filter!(vs) do v
 #   return v[end] != 21477.0
#end

@assert !isempty(vs)
const b_mps = randn(n)

max_norm = maximum(norm.(vs))

function f(x)
    r = dot(b_mps, x)
    for v in vs
        r += 1 / (2) * norm(x - v)^2
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

@testset "MPS pk1 instance" begin
    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, print_iter = 10, fw_epsilon = 1e-1, min_node_fw_epsilon = 1e-3, time_limit=2400)
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
