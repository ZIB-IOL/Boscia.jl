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
# seed = rand(UInt64)
# @show seed
# #seed = 0xeadb922ca734998b  
# Random.seed!(seed)

# n = 10
# m = 35
# l = 30
# k = 4

# sol_x = rand(1:l, n)
# for _ in 1:(n-k)
#     sol_x[rand(1:n)] = 0
# end

#=k=0 # correct k
for i in 1:n
    if sol_x[i] == 0 
        global k += 1
    end
end
k = n-k =#

# const D = rand(m, n)
# const y_d = D * sol_x


function int_sparse_regression(n, m, l, k, seed)
    Random.seed!(seed)
    sol_x = rand(1:l, n)
    for _ in 1:(n-k)
        sol_x[rand(1:n)] = 0
    end
    D = rand(m, n)
    y_d = D * sol_x

    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    z = MOI.add_variables(o, n)
    for i in 1:n
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0 * l))
        MOI.add_constraint(o, x[i], MOI.Integer())

        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())

        MOI.add_constraint(o, 1.0 * x[i] - 1.0 * l * z[i], MOI.LessThan(0.0))
    end
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    # MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(zeros(n),x), sum(Float64.(iszero.(x)))), MOI.GreaterThan(1.0*(n-k)))
    # MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n),z), 0.0), MOI.GreaterThan(1.0*k))
    lmo = Boscia.MathOptBLMO(o)

    function f(x)
        xv = @view(x[1:n])
        return 1 / 2 * sum(abs2, y_d - D * xv)  #+ lambda_2*FrankWolfe.norm(x)^2 + lambda_0*sum(x[p+1:2p])
    end

    function grad!(storage, x)
        storage .= 0
        @view(storage[1:n]) .= transpose(D) * (D * @view(x[1:n]) - y_d)
        return storage
    end

    return lmo, f, grad!
end



@testset "Integer sparse regression" begin
    seed = rand(UInt64)
    @show seed
    #seed = 0xeadb922ca734998b  
    Random.seed!(seed)
    n = 10
    m = 30
    l = 5
    k = 4
    lmo, f, grad! = int_sparse_regression(n,m,l,k, seed)

    #= function perform_strong_branch(tree, node)
         return node.level <= length(tree.root.problem.integer_variables)
     end
     branching_strategy = Boscia.HybridStrongBranching(10, 1e-3, HiGHS.Optimizer(), perform_strong_branch)
     MOI.set(branching_strategy.pstrong.optimizer, MOI.Silent(), true)=#


    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, max_fw_iter=10001, rel_dual_gap=1e-3, time_limit=30)

    # val_min, x_min = Boscia.sparse_min_via_enum(f, n, k, fill(0:l, n))
    # #@show x_min
    # @show x[1:n]
    # @show x_min
    # @test val_min == f(x)
    # @test isapprox(x[1:n], x_min)
    # @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-6)
end




############ Decide which strategies to run #####################
strategies = Any[
    "MOST_INFEASIBLE", "Strong_Branching"
]

for iterations_stable in Int64[5,10,20]
    for decision_function in [
        "product", 
        "weighted_sum"
        ]
        if decision_function == "product"
            μ = 1e-6
            push!(strategies, Dict(:iterations_stable => iterations_stable, :μ => μ, :decision_function => decision_function))
        else
            for μ in [0.7]
                push!(strategies, Dict(:iterations_stable => iterations_stable, :μ => μ, :decision_function => decision_function))
            end
        end
    end
end


############## Example sizes ######################

example_dimensions = [10, 15, 20, 25]
fac_choices = [3,5]
seeds = rand(UInt64, 3)



############## Set Parameters for all runs ######################
verbose = true
print_iter = 100
time_limit = 600
rel_dual_gap=1e-2
# Set parameters for saving results
file_name = "int_sparse_reg_examples_a_c"
#example_name = string("int_sparse_reg_n_", n, "_m_", m, "_l_",l, "_k_", k)
#################################################################
for seed in seeds
    for dim in example_dimensions
        for fac in fac_choices
            n = dim
            m = 3*n
            l = 30
            k = ceil(n/fac)
            example_name = string("int_sparse_reg_n_", n, "_m_", m, "_l_",l, "_k_", k)
            for branching_strategy in strategies
                lmo, f, grad! = int_sparse_regression(n,m,l,k, seed)
                if branching_strategy == "Strong_Branching"
                    blmo = Boscia.MathOptBLMO(SCIP.Optimizer())
                    branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
                    MOI.set(branching_strategy.bounded_lmo.o, MOI.Silent(), true)
                    x, _, result =
                        Boscia.solve(
                            f, 
                            grad!, 
                            lmo,  
                            branching_strategy=branching_strategy,verbose=verbose,
                            print_iter=print_iter, 
                            time_limit=time_limit,
                            rel_dual_gap=rel_dual_gap
                        )
                    settings = "Strong_Branching"
                    Boscia.save_results(result, settings, example_name, seed, file_name, false) 
                
                elseif branching_strategy == "MOST_INFEASIBLE"
                    x, _, result = Boscia.solve(
                        f, 
                        grad!, 
                        lmo, 
                        verbose=verbose, 
                        print_iter=print_iter, 
                        time_limit=time_limit,
                        rel_dual_gap=rel_dual_gap
                        )
                    settings = "MOST_INFEASIBLE"
                    Boscia.save_results(result, settings, example_name, seed, file_name, false) 
                else
                    iterations_stable = branching_strategy[:iterations_stable]
                    decision_function = branching_strategy[:decision_function]
                    μ = branching_strategy[:μ]
                    x, _, result = Boscia.solve(
                        f, 
                        grad!, 
                        lmo,
                        branching_strategy=Boscia.PSEUDO_COST(iterations_stable,false, lmo, μ, decision_function),
                        verbose=verbose, 
                        print_iter=print_iter, 
                        time_limit=time_limit,
                        rel_dual_gap=rel_dual_gap
                        )
                        settings = "PSEUDO_COST_" * decision_function * "_" * string(iterations_stable) * "_μ=" * string(μ)
                    Boscia.save_results(result, settings, example_name, seed, file_name, false)
                end
            end
        end
    end
end










