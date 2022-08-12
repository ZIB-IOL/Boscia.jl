using Statistics
using Boscia
using FrankWolfe
using Random
using SCIP
import Bonobo
using Test
import MathOptInterface
const MOI = MathOptInterface

# Integer sparse regression

# min norm(y-A x)² 
# s.t. 0 <= x_i <= r
# ∑ x_i <= k 
# x_i ∈ Z for i = 1,..,n

# There A represents the collection of data points and 
# is a very tall matrix, i.e. number of rows = m >> number of columns = n.
# y - is the vector of results.
# r - controls how often we have to maximal split on a index.
# k - is the sparsity parameter. We only want a few non zero entries.

# For bug hunting:
seed = rand(UInt64)
@show seed
#seed = 0xf019ccfbe2bfbe09 
Random.seed!(seed)

n = 10
m = 30
l = 5
k = 4

sol_x = rand(1:l, n)
for _ in 1:(n-k)
    sol_x[rand(1:n)] = 0
end

#=k=0 # correct k
for i in 1:n
    if sol_x[i] == 0 
        global k += 1
    end
end
k = n-k =#

const D = rand(m,n)
const y_d = D*sol_x

@testset "Integer sparse regression" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o,n)
    z = MOI.add_variables(o,n)
    for i in 1:n
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0*l))
        MOI.add_constraint(o, x[i], MOI.Integer())

        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())

        MOI.add_constraint(o, 1.0 * x[i] + 1.0 * l * z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, 1.0 * x[i] - 1.0 * l * z[i], MOI.LessThan(0.0))
    end 
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0*k))
   # MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(zeros(n),x), sum(Float64.(iszero.(x)))), MOI.GreaterThan(1.0*(n-k)))
   # MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n),z), 0.0), MOI.GreaterThan(1.0*k))
    lmo = FrankWolfe.MathOptLMO(o)

    function f(x)
        xv = @view(x[1:n])
        return 1/2 * sum(abs2, y_d - D * xv)  #+ lambda_2*FrankWolfe.norm(x)^2 + lambda_0*sum(x[p+1:2p])
    end
    
    function grad!(storage, x)
        storage .= 0
        @view(storage[1:n]) .= transpose(D)* (D*@view(x[1:n]) - y_d)
        return storage
    end

   #= function perform_strong_branch(tree, node)
        return node.level <= length(tree.root.problem.integer_variables)
    end
    branching_strategy = Boscia.HybridStrongBranching(10, 1e-3, HiGHS.Optimizer(), perform_strong_branch)
    MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)=#


    x, _,result = Boscia.solve(f, grad!, lmo, verbose = true, max_fw_iter = 10001, rel_dual_gap = 1e-3)

    val_min, x_min = Boscia.sparse_min_via_enum(f, n, k, fill(0:l, n))
    #@show x_min
    @show x[1:n]
    @show x_min
    @test val_min == f(x)
    @test isapprox(x[1:n], x_min)
    @test isapprox(f(x), f(result[:raw_solution]), atol = 1e-6, rtol = 1e-6)
end
