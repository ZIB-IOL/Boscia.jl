using Statistics
using BranchWolfe
using FrankWolfe
using Random
using SCIP
import Bonobo
import MathOptInterface
const MOI = MathOptInterface

# Integer sparse regression

# min norm(y-A x)² 
# s.t. 0 <= x_i <= r
# ∑ x_i <= k 
# x_i ∈ Z for i = 1,..,n

# There A represents the collection of data points and 
# is a very flat matrix, i.e. number of rows = m >> number of columns = n.
# y is the vector of results.
# r controls how often we have to maximal split on a index.
# k is the sparsity parameter. We only want a few non zero entries.

n = 10
m = 30
l = 5
k = 5

sol_x = rand(1:l, n)
for _ in 1:(n-k)
    sol_x[rand(1:n)] = 0
end

k=0 # correct k
for i in 1:n
    if sol_x[i] == 0 
        global k += 1
    end
end

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

        MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,1.0*l], [x[i], z[i]]), 0.0), MOI.GreaterThan(0.0))
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-1.0*l], [x[i], z[i]]), 0.0), MOI.LessThan(0.0))
    end 
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n),z), 0.0), MOI.LessThan(1.0*k))
    MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n),z), 0.0), MOI.GreaterThan(1.0*k))
    lmo = FrankWolfe.MathOptLMO(o)

    function f(x)
        return sum((y_d-D*x[1:n]).^2)  #+ lambda_2*FrankWolfe.norm(x)^2 + lambda_0*sum(x[p+1:2p]) 
    end
    
    function grad!(storage, x)
        storage.=vcat(2*(transpose(D)*D*x[1:n] - transpose(D)*y_d + lambda_2*x[1:n]), zeros(n))
        return storage
    end

    function perform_strong_branch(tree, node)
        return node.level <= length(tree.root.problem.integer_variables)
    end
    branching_strategy = BranchWolfe.HybridStrongBranching(10, 1e-3, HiGHS.Optimizer(), perform_strong_branch)
    MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)

    x, _,_, dual_gap = BranchWolfe.branch_wolfe(f, grad!, lmo, verbose = true, branching_strategy = branching_strategy, print_iter = 1)

    @test f(x) == f(sol_x)
    @test isapprox(x[1:n], sol_x)
end