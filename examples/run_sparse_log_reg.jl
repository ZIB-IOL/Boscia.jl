include("sparse_log_reg.jl") 

bo_mode="boscia"
for dimension in [15:1:30;]
    for seed in 1:10
        for M in [0.1,1.0,3.0,5.0]
            @show seed, dimension
            try 
                sparse_log_regression(seed, dimension, M, k; bo_mode=bo_mode)
            catch e
                println(e)
                open("sparse_log_regression_errors.txt","a") do io
                    println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
                end
            end
        end
    end
end

bo_mode = "as"
for dimension in [15:1:30;]
    for seed in 1:10
        for M in [0.1,1.0,3.0,5.0]
            @show seed, dimension
            try 
                sparse_log_regression(seed, dimension, M, k; bo_mode=bo_mode)
            catch e
                println(e)
                open("sparse_log_regression_errors.txt","a") do io
                    println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
                end
            end
        end
    end
end

bo_mode = "ss"
for dimension in [15:1:30;]
    for seed in 1:10
        for M in [0.1,1.0,3.0,5.0]
            @show seed, dimension
            try 
                sparse_log_regression(seed, dimension, M, k; bo_mode=bo_mode)
            catch e
                println(e)
                open("sparse_log_regression_errors.txt","a") do io
                    println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
                end
            end
        end
    end
end

bo_mode = "as_ss"
for dimension in [15:1:30;]
    for seed in 1:10
        for M in [0.1,1.0,3.0,5.0]
            @show seed, dimension
            try 
                sparse_log_regression(seed, dimension, M, k; bo_mode=bo_mode)
            catch e
                println(e)
                open("sparse_log_regression_errors.txt","a") do io
                    println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
                end
            end
        end
    end
end

bo_mode = "afw"
for dimension in [15:1:30;]
    for seed in 1:10
        for M in [0.1,1.0,3.0,5.0]
            @show seed, dimension
            try 
                sparse_log_regression(seed, dimension, M, k; bo_mode=bo_mode)
            catch e
                println(e)
                open("sparse_log_regression_errors.txt","a") do io
                    println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
                end
            end
        end
    end
end

bo_mode = "boscia"
for dimension in [15:1:30;]
    for seed in 1:10
        for M in [0.1,1.0,3.0,5.0]
            @show seed, dimension
            try 
                sparse_log_reg_scip(seed, dimension, M, k)
            catch e
                println(e)
                open("sparse_log_regression_errors.txt","a") do io
                    println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
                end
            end
        end
    end
end
