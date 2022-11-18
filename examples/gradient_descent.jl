import LinearAlgebra

function gradient_descent(f, grad, n)
    # get start point
    x = ones(n)
    x = reshape(x, length(x), 1)'

    alpha = 1
    alpha_max = 100
    theta = .0001
    verbose = false
    iterations = 0
    current_phi = f(x)
    current_J = grad(x)
    delta = - current_J / LinearAlgebra.norm(current_J) # direction

    while ((LinearAlgebra.norm(alpha*delta)) >= theta) 
        if iterations%5==0 && verbose
            println("Iteration: ", iterations)
            println("Current cost: ", current_phi[0])
        end
        iterations += 1

        potential_x = x.+alpha*delta
        potential_phi = f(potential_x)
        potential_J = grad(potential_x)

        while potential_phi > current_phi + 0.00001 * [LinearAlgebra.dot(current_J,(alpha*delta)')]
            alpha = alpha/2
            potential_x = x.+alpha*delta
            potential_phi = f(potential_x)
            potential_J = grad(potential_x)
        end
        x_new = x + alpha*delta
        x = x_new
        alpha = min(2*alpha, alpha_max)

        current_phi = f(x)
        current_J = grad(x)
        delta = - current_J / LinearAlgebra.norm(current_J) 
    end 

    if verbose
        println("-------------------------------------------")
        println("Solution: ", x, " with cost: ", current_phi[0])
        println("Found in ", iterations, " iterations")
    end 

    return x
end

function f(x)
    n = length(x)
    x = reshape(x, n, 1)'
    c = 10
    C = LinearAlgebra.Diagonal([c^((i-1)/(n-1)) for i in 1:n])
    # println(C)
    result = (C * x')' * (C * x')
    reshape(result, 1)
end

function grad(x)
    n = length(x)
    x = reshape(x, n, 1)'    
    c = 10
    C = LinearAlgebra.Diagonal([c^((i-1)/(n-1)) for i in 1:n])
    (2 * x * C' * C) # todo reshape
end

# n = 5 
# x = ones(n)
# println(f(x))
# println(grad(x))

x = gradient_descent(f, grad, n)
println(x)
println(f(x))