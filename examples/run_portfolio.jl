include("portfolio.jl")

# different tightening strategies
mode = "integer"

bo_mode="boscia"
for dimension in [15,20,25,30,35,45]
    for seed in 4:5
        @show seed, dimension
        try 
            portfolio(seed, dimension, true; bo_mode=bo_mode,mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "local_tightening"
for dimension in [15,20,25,30,35,45]
    for seed in 4:5
        @show seed, dimension
        try 
            portfolio(seed, dimension, true; bo_mode=bo_mode,mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "global_tightening"
for dimension in [15,20,25,30,35,45]
    for seed in 4:5
        @show seed, dimension
        try 
            portfolio(seed, dimension, true; bo_mode=bo_mode,mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "no_tightening"
for dimension in [15,20,25,30,35,45]
    for seed in 4:5
        @show seed, dimension
        try 
            portfolio(seed, dimension, true; bo_mode=bo_mode,mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

mode = "mixed"

bo_mode="boscia"
for dimension in [35,45,50]
    for seed in 1:3
        @show seed, dimension
        try 
            portfolio(seed, dimension, true; bo_mode=bo_mode,mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "local_tightening"
for dimension in [35,45,50]
    for seed in 1:3
        @show seed, dimension
        try 
            portfolio(seed, dimension, true; bo_mode=bo_mode,mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "global_tightening"
for dimension in [35,45,50]
    for seed in 1:3
        @show seed, dimension
        try 
            portfolio(seed, dimension, true; bo_mode=bo_mode,mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "no_tightening"
for dimension in [35,45,50]
    for seed in 1:3
        @show seed, dimension
        try 
            portfolio(seed, dimension, true; bo_mode=bo_mode,mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

# different solvers and boscia methods 
mode = "integer"

bo_mode="boscia"
for dimension in [20:5:120;]
    for seed in 1:10
        @show seed, dimension
        try 
            portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end 

bo_mode="ss"
for dimension in [20:5:120;]
    for seed in 1:10
        @show seed, dimension
        try 
            portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode="as"
for dimension in [20:5:120;]
    for seed in 1:10
        @show seed, dimension
        try 
            portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode="as_ss"
for dimension in [20:5:120;]
    for seed in 1:10
        @show seed, dimension
        try 
            portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode="afw"
for dimension in [20:5:120;]
    for seed in 1:10
        @show seed, dimension
        try 
            portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "scip_oa"
for dimension in [20:5:120;]
    for seed in 1:10
        @show seed, dimension
        try 
            portfolio_scip(seed, dimension; mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "ipopt"
for dimension in [20:5:120;] #[20:5:120;]
    for seed in 1:10#1:10
        @show seed, dimension
        try 
            portfolio_ipopt(seed, dimension; mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

mode = "mixed"

bo_mode="boscia"
for dimension in [20:5:120;]
    for seed in 1:10
        @show seed, dimension
        try 
            portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end 

bo_mode="ss"
for dimension in [20:5:120;]
    for seed in 1:10
        @show seed, dimension
        try 
            portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode="as"
for dimension in [20:5:120;]
    for seed in 1:10
        @show seed, dimension
        try 
            portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode="as_ss"
for dimension in [20:5:120;]
    for seed in 1:10
        @show seed, dimension
        try 
            portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode="afw"
for dimension in [20:5:120;]
    for seed in 1:10
        @show seed, dimension
        try 
            portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "scip_oa"
for dimension in [20:5:120;]
    for seed in 1:10
        @show seed, dimension
        try 
            portfolio_scip(seed, dimension; mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "ipopt"
for dimension in [20:5:120;] #[20:5:120;]
    for seed in 1:10#1:10
        @show seed, dimension
        try 
            portfolio_ipopt(seed, dimension; mode=mode)
        catch e
            println(e)
            open("portfolio_" * mode * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end