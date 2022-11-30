using Statistics
using Distributions
using Boscia
using FrankWolfe
using Random
using LinearAlgebra
using SCIP
import Bonobo
import MathOptInterface
MOI = MathOptInterface
using Dates
using Printf
using Test
using DataFrames
using CSV

# Sparse regression

# Constant parameters for the sparse regression
# min norm(y-A β)² + λ_0 ∑ z_i + λ_2 ||β||²
# s.t. -Mz_i <= β_i <= Mz_i
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p 

# A - matrix of observations.
# y - vector of results.
# We want to match Aβ as closely as possible to y 
# while having relative few non zero entries in β.
# Each continuous variable β_i is assigned a binary z_i,
# z_i = 0 => β_i = 0

seed = 1
Random.seed!(seed)

samples = 10
data_1 = rand(MvNormal(ones(2), [0.3,0.3]),samples)
y_1 = zeros(samples)

data_2 = rand(MvNormal([2,2], [0.3,0.3]),samples)
y_2 = ones(samples)

A = hcat(data_1, data_2)'
y = vcat(y_1, y_2)

n0 = 2
p = 2
k = 2.0 #ceil(n0/2)

mu = 10.0 * rand(Float64);
M = 2 * var(A)
lambda_0 = 0#rand(Float64);
lambda_2 = 0#10.0 * rand(Float64);

# n0 = 10;
# p = 5 * n0;
# k = ceil(n0 / 5);
# lambda_0 = rand(Float64);
# lambda_2 = 10.0 * rand(Float64);
# A = rand(Float64, n0, p)
# y = rand(Float64, n0)
# M = 2 * var(A)

# load heart disease data
# file_name = "processed.cleveland.data"
# df_cleveland = DataFrame(CSV.File(file_name, header=false))
# headers = [:age,:sex,:cp,:trestbps,:chol,:fbs,:restecg,:thalach,:exang,
#     :oldpeak,:slope,:ca,:thal,:diagnosis]
# rename!(df_cleveland,headers)
# df_cleveland.thal .= replace.(df_cleveland.thal, "?" => -9.0)
# df_cleveland.ca .= replace.(df_cleveland.ca, "?" => -9.0)
# df_cleveland[!,:ca] = parse.(Float64,df_cleveland[!,:ca])
# df_cleveland[!,:thal] = parse.(Float64,df_cleveland[!,:thal])

# # labels of -1, 1
# df_cleveland[df_cleveland.diagnosis .> 0,:diagnosis] .= 1
# df_cleveland[df_cleveland.diagnosis .== 0,:diagnosis] .= -1
# y = df_cleveland[!,:diagnosis]
# A = Matrix(select!(df_cleveland, Not(:diagnosis)))
# n0 = size(A)[1] # 303
# p = size(A)[2]  # 13
# k = 13.0
# M = 2 * var(A)
# lambda_0 = rand(Float64);
# lambda_2 = 10.0 * rand(Float64);

# "Sparse Regression" 
@testset "Sparse regression" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, 2p)
    for i in p+1:2p
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, x[i], MOI.ZeroOne()) # or MOI.Integer()
    end
    for i in 1:p
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, M], [x[i], x[i+p]]), 0.0),
            MOI.GreaterThan(0.0),
        )
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -M], [x[i], x[i+p]]), 0.0),
            MOI.LessThan(0.0),
        )
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p), x[p+1:2p]), 0.0),
        MOI.LessThan(k),
    )
    lmo = FrankWolfe.MathOptLMO(o)

    function f(x)
        xv = @view(x[1:p])
        return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
    end

    function grad!(storage, x)
        storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
        storage[p+1:2p] .= lambda_0
        return storage
    end

    x, _, result = Boscia.solve(f, grad!, lmo, verbose=false, fw_epsilon=1e-3, print_iter=10)
    @show x
    @show f(x) 
    # @show result // too large to be output
    # @test f(x) <= f(result[:raw_solution]) + 1e-6
end
