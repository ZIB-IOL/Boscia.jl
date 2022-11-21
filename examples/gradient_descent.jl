using LinearAlgebra
using DataFrames
using CSV
using Random

seed = 1
Random.seed!(seed)

function gradient_descent(f, grad!, n)
    # get start point
    x = ones(n)
    x = reshape(x, length(x), 1)'
    storage = similar(x)

    alpha = 1
    alpha_max = 100
    theta = .0001
    verbose = false
    iterations = 0
    current_phi = f(x)
    current_J = grad!(storage, x)
    delta = - current_J / LinearAlgebra.norm(current_J) # direction

    while ((LinearAlgebra.norm(alpha*delta)) >= theta) 
        if iterations%5==0 && verbose
            println("Iteration: ", iterations)
            println("Current cost: ", current_phi[0])
        end
        iterations += 1

        potential_x = x.+alpha*delta
        potential_phi = f(potential_x)
        potential_J = grad!(storage, potential_x)
        # @show potential_J
        # @show storage
        # @show 0.00001 * [LinearAlgebra.dot(current_J,(alpha*delta)')]
        # @show current_phi
        # @show potential_phi
        while potential_phi > current_phi + 0.00001 * LinearAlgebra.dot(current_J,(alpha*delta)')
            alpha = alpha/2
            potential_x = x.+alpha*delta
            potential_phi = f(potential_x)
            potential_J = grad!(storage, potential_x)
        end
        x_new = x + alpha*delta
        x = x_new
        alpha = min(2*alpha, alpha_max)

        current_phi = f(x)
        current_J = grad!(storage, x)
        delta = - current_J / LinearAlgebra.norm(current_J) 
    end 

    if verbose
        println("-------------------------------------------")
        println("Solution: ", x, " with cost: ", current_phi[0])
        println("Found in ", iterations, " iterations")
    end 

    return x
end

function process_data()
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
    # # print(df_cleveland[!,:diagnosis])
    # # display(first(df_cleveland, 5))
    # # display(df_cleveland)
    # y = df_cleveland[!,:diagnosis]
    # A = Matrix(select!(df_cleveland, Not(:diagnosis)))
    n0 = 10;
    p = 5 * n0;
    A = rand(Float64, n0, p)
    y = rand(Float64, n0)
    return A, y
end

function build_objective_gradient(A, y, mu)
    # just flexing with unicode
    # reusing notation from Bach 2010 Self-concordant analyis for LogReg
    ℓ(u) = log(exp(u/2) + exp(-u/2))
    dℓ(u) = -1/2 + inv(1 + exp(-u))
    n = length(y)
    invn = inv(n)
    p = size(A)[2]
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
        @. storage +=  mu * x
        storage
    end
    (f, grad!)
end

# function f(x)
#     n = length(x)
#     x = reshape(x, n, 1)'
#     c = 10
#     C = LinearAlgebra.Diagonal([c^((i-1)/(n-1)) for i in 1:n])
#     # println(C)
#     result = (C * x')' * (C * x')
#     reshape(result, 1)
# end

# function grad(x)
#     n = length(x)
#     x = reshape(x, n, 1)'    
#     c = 10
#     C = LinearAlgebra.Diagonal([c^((i-1)/(n-1)) for i in 1:n])
#     (2 * x * C' * C)
# end

A,y = process_data()
mu = 10.0 * rand(Float64)
n = size(A)[2]
storage = similar(ones(n))
f, grad! = build_objective_gradient(A, y, mu)

x = gradient_descent(f, grad!, n)
@show x
@show f(x) 