using Statistics
using Distributions
using Boscia
using FrankWolfe
using Random
using Test
using SCIP
import Bonobo
import MathOptInterface
const MOI = MathOptInterface
using Dates
using Printf

# Lasso

# Constant parameters for the sparse regression
# min norm(y-A β)² + λ_0 ∑ z_i + λ_2 ||β||²
# s.t. -Mz_i<=β_i <= Mz_i
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p 

n=20
p = 5*n
k = ceil(n/5)
group_size = convert(Int64, floor(p/k))
M_g = 5.0

const lambda_0_g = 0.0
const lambda_2_g = 0.0
const A_g = rand(Float64, n, p)
β_sol = rand(Distributions.Uniform(-M_g, M_g), p)
k_int = convert(Int64, k)

for i in 1:k_int
    for _ in 1:group_size-1
        β_sol[rand(((i-1)*group_size+1):(i*group_size))] = 0
    end
end
const y_g = A_g * β_sol
k=0 # correct k
for i in 1:p
    if β_sol[i] == 0 
        global k += 1
    end
end
k = p-k 

groups= []
for i in 1:(k_int-1)
    push!(groups, ((i-1)*group_size+1):(i*group_size))
end
push!(groups,((k_int-1)*group_size+1):p)

@testset "Sparse Regression Group" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o,p)
    z = MOI.add_variables(o,p)
    for i in 1:p
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne()) # or MOI.Integer()
    end 
    for i in 1:p
        MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,M_g], [x[i], z[i]]), 0.0), MOI.GreaterThan(0.0))
        MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-M_g], [x[i], z[i]]), 0.0), MOI.LessThan(0.0))
        # Indicator: x[i+p] = 1 => -M_g <= x[i] <= M_g
        gl = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[i])),],
            [0.0, 0.0], )
        gg = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, x[i])),],
            [0.0, 0.0], )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(M_g)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-M_g)))
    end
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p),z), 0.0), MOI.LessThan(1.0*k))
    for i in 1:k_int
        MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(group_size),z[groups[i]]), 0.0), MOI.GreaterThan(1.0))
    end
    lmo = FrankWolfe.MathOptLMO(o)
    global_bounds = Boscia.IntegerBounds()
    for i = 1:p
        push!(global_bounds, (i+p, MOI.GreaterThan(0.0)))
        push!(global_bounds, (i+p, MOI.LessThan(1.0)))
        push!(global_bounds, (i, MOI.GreaterThan(-M_g)))
        push!(global_bounds, (i, MOI.LessThan(M_g)))
    end

    function f(x)
        return sum((y_g-A_g*x[1:p]).^2) + lambda_0_g*sum(x[p+1:2p]) + lambda_2_g*FrankWolfe.norm(x[1:p])^2
    end
    function grad!(storage, x)
        storage.=vcat(2*(transpose(A_g)*A_g*x[1:p] - transpose(A_g)*y_g + lambda_2_g*x[1:p]), lambda_0_g*ones(p))
        return storage
    end
   
    x,_,result = Boscia.solve(f, grad!, lmo, verbose = true, rel_dual_gap = 1e-5)

    # println("Solution: $(x[1:p])")
    z = x[p+1:2p]
    @test sum(z) <= k
    for i in 1:k_int
        @test sum(z[groups[i]]) >= 1
    end
    @test f(x) <= f(result[:raw_solution])
    @show f(x)
    @show f(vcat(β_sol, zeros(p)))
    @show x[1:p]
end
