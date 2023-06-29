# include("sparse_reg.jl")
# include("poisson.jl")
# include("int_regression.jl")
# include("lasso.jl")
using Statistics
using Boscia
using FrankWolfe
using Random
using SCIP
import Bonobo
import MathOptInterface
const MOI = MathOptInterface
using DataFrames
using CSV
# include("build_csv.jl")
include("plot_bigM_vs_indicator.jl")

function test(example, dimension, fac, seed, use_indicator; time_limit = Inf, rtol = 1e-2)
    if example == "lasso"
        build_lasso(dimension, fac, seed, use_indicator, time_limit, rtol)
    elseif example == "int_reg"
        build_int_reg(dimension, fac, seed, use_indicator, time_limit, rtol)
    elseif example == "sparse_reg"
        build_sparse_reg(dimension, fac, seed, use_indicator, time_limit, rtol)
    elseif example == "poisson_reg"
        build_poisson_reg(dimension, fac, seed, use_indicator, time_limit, rtol)
    end
end

#=
dimensions = [20,50,70,100,150] # 20, 50, 70, 100, 150
iter = 1
factors = [2, 5, 10] # 2, 5, 10
# "lasso", "int_reg", "sparse_reg", "poisson_reg"
examples =["lasso", "int_reg", "sparse_reg", "poisson_reg"] 
not_the_same = 0
for example in examples
    for seed in [1,2,3] #, 2,3 ,4, 5
        for dim in dimensions
            for fac in factors
                println("Example: $(example) Dimension: $(dim) Factor: $(fac) Seed: $(seed)")
                Random.seed!(seed)

                println("Big M Formulation")
                x_M, result_M = test(example, dim, fac, seed, false, time_limit = 1800)
                println("Indicator Formulation")
                x_I, result_I = test(example, dim, fac, seed, true, time_limit = 1800)
                #@show x_I

              
               if !isapprox(result_M, result_I, atol = 1e-3, rtol = 5e-2)
                    @warn "Big M result: $(result_M) Indicator result: $(result_I)"
                    global not_the_same += 1
                end
            end
        end
   end
end
println("There are $(not_the_same) instances with different results") 
=#

mode = "lasso"
for dim in [20,50,70,100,150]
    for factor in [2,5,10]
        for seed in [1,2,3]
            plot_bigM_vs_indicator(mode; dim=dim, factor=factor, seed=seed)
        end
    end
end

mode = "sparse_reg"
for dim in [20,50,70,100,150]
    for factor in [2,5,10]
        for seed in [1,2,3]
            println("seed ", seed)
            plot_bigM_vs_indicator(mode; dim=dim, factor=factor, seed=seed)
        end
    end
end


mode = "poisson_reg"
for dim in [20,50,70,100,150]
    for factor in [2,5,10]
        for seed in [1,2,3]
            plot_bigM_vs_indicator(mode; dim=dim, factor=factor, seed=seed)
        end
    end
end


mode = "int_reg"
for dim in [20,50,70,100,150]
    for factor in [2,5,10]
        for seed in [1,2,3]
            plot_bigM_vs_indicator(mode; dim=dim, factor=factor, seed=seed)
        end
    end
end 

