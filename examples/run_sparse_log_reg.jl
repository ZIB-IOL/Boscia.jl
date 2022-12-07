include("sparse_log_reg.jl") 

bo_mode="boscia"
for dimension in [15:1:30;]
    for seed in 1:10
        @show seed, dimension
        try 
            sparse_log_regression(seed, dimension, M, k, 1; bo_mode=bo_mode)
        catch e
            println(e)
            open("sparse_log_regression_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "as"
for dimension in [15:1:30;]
    for seed in 1:10
        @show seed, dimension
        try 
            sparse_log_regression(seed, dimension, 1; bo_mode=bo_mode)
        catch e
            println(e)
            open("sparse_log_regression_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "ss"
for dimension in [15:1:30;]
    for seed in 1:10
        @show seed, dimension
        try 
            sparse_log_regression(seed, dimension, 1; bo_mode=bo_mode)
        catch e
            println(e)
            open("sparse_log_regression_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "as_ss"
for dimension in [15:1:30;]
    for seed in 1:10
        @show seed, dimension
        try 
            sparse_log_regression(seed, dimension, 1; bo_mode=bo_mode)
        catch e
            println(e)
            open("sparse_log_regression_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "afw"
for dimension in [15:1:30;]
    for seed in 1:10
        @show seed, dimension
        try 
            sparse_log_regression(seed, dimension, 1; bo_mode=bo_mode)
        catch e
            println(e)
            open("sparse_log_regression_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

