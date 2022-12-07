using Statistics
using Boscia
using FrankWolfe
using Random
using LinearAlgebra
using SCIP
import Bonobo
import MathOptInterface
const MOI = MathOptInterface
using Dates
using Printf
using Test
using DataFrames
using CSV
using Random

# Sparse logistic regression

# Constant parameters for the sparse regression
# min 1/N ∑ log(1 + exp(-y_i * β @ a_i)) + λ_0 ∑ z_i + μ/2 * ||β||²
# s.t. -Mz_i <= β_i <= Mz_i
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p 

# A - matrix of observations.
# y - vector of results.
# while having relative few non zero entries in β.
# Each continuous variable β_i is assigned a binary z_i,
# z_i = 0 => β_i = 0

function sparse_log_regression(seed=1, dimension=10, M=3, k=5.0; data="random", bo_mode="boscia") 
    Random.seed!(seed)

    if data=="heart_disease"
        file_name = "processed.cleveland.data"
        df_cleveland = DataFrame(CSV.File(file_name, header=false))
        headers = [:age,:sex,:cp,:trestbps,:chol,:fbs,:restecg,:thalach,:exang,
            :oldpeak,:slope,:ca,:thal,:diagnosis]
        rename!(df_cleveland,headers)
        df_cleveland.thal .= replace.(df_cleveland.thal, "?" => -9.0)
        df_cleveland.ca .= replace.(df_cleveland.ca, "?" => -9.0)
        df_cleveland[!,:ca] = parse.(Float64,df_cleveland[!,:ca])
        df_cleveland[!,:thal] = parse.(Float64,df_cleveland[!,:thal])

        # labels of -1, 1
        df_cleveland[df_cleveland.diagnosis .> 0,:diagnosis] .= 1
        df_cleveland[df_cleveland.diagnosis .== 0,:diagnosis] .= -1
        # print(df_cleveland[!,:diagnosis])
        # display(first(df_cleveland, 5))
        # display(df_cleveland)
        y = df_cleveland[!,:diagnosis]
        A = Matrix(select!(df_cleveland, Not(:diagnosis)))
        # print(size(A))  # (303, 13) 
        # print(size(y))  # (303,)
        n0 = size(A)[1] # 303
        p = size(A)[2]  # 13
        # k = 5.0
        #M = 2 * var(A)
        mu = 0 #10.0 * rand(Float64);

    elseif data == "random"
        n0 = dimension
        p = 5 * n0;
        #k = 5.0#ceil(n0 / 5);
        A = rand(Float64, n0, p)
        y = Random.bitrand(n0)
        y = [i == 0 ? -1 : 1 for i in y]
        # y = rand(Float64, n0)
        #M = 2 * var(A)
        mu = 0#10.0 * rand(Float64);
    end

    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, 2p)
    for i in p+1:2p
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, x[i], MOI.ZeroOne()) 
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

    function build_objective_gradient(A, y, mu)
        # just flexing with unicode
        # reusing notation from Bach 2010 Self-concordant analyis for LogReg
        ℓ(u) = log(exp(u/2) + exp(-u/2))
        dℓ(u) = -1/2 + inv(1 + exp(-u))
        n = length(y)
        invn = inv(n)
        function f(x)
            xv = @view(x[1:p])
            err_term = invn * sum(eachindex(y)) do i # 1/N
                dtemp = dot(A[i,:], xv) # predicted label
                ℓ(dtemp) - y[i] * dtemp / 2
            end
            pen_term = mu * dot(xv, xv) / 2
            err_term + pen_term
        end
        function grad!(storage, x)
            storage .= 0
            xv = @view(x[1:p])
            for i in eachindex(y)
                dtemp = dot(A[i,:], xv)
                @. storage += invn * A[i] * (dℓ(dtemp) - y[i] / 2)
            end
            @. storage += mu * x
            storage
        end
        (f, grad!)
    end

    f, grad! = build_objective_gradient(A, y, mu)

    x, _, result = Boscia.solve(f, grad!, lmo, verbose=false, fw_epsilon=1e-3, print_iter=10)
    @show x
    @show f(x) 
    return f, x

    xv = @view(x[1:p])
    predictions = [p > 0.5 ? 1 : -1 for p in A*xv]
    @show (predictions, y)
end

# f,x_5 = sparse_log_regression(1, 5, 5.0, data="random")
# _,x_10 = sparse_log_regression(1, 5, 10.0, data="random")
# @assert f(x_10)<=f(x_5)

# TODO: assertionerror f,x_13 = sparse_log_regression(1, 1, 12.0, data="heart_disease")
# f,x_13 = sparse_log_regression(1, 1, 12.0, data="heart_disease")
# _,x_12 = sparse_log_regression(1, 1, 11.0, data="heart_disease")
# @assert f(x_13)<=f(x_12)

# good time ? sparse_log_regression(1, 50, 15, 10.0, data="random")