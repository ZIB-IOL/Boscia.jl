include("poisson_reg.jl")

bo_mode="afw"
for dimension in [50:20:100;]
   for seed in 1:10
       for ns in [0.1,1,5,10]
           @show seed, dimension
           try 
               poisson(seed, dimension, ns, 1; bo_mode=bo_mode)
           catch e 
               println(e)
               open("poisson_errors.txt","a") do io
                   println(io, seed, " ", dimension, " ", ns, " ", bo_mode, " : ", e)
               end
           end
       end
   end
end

include("tailed_cardinality.jl")

bo_mode = "afw"
for dimension in [50:10:150;]
  for seed in 1:10
      @show seed, dimension
      try 
          sparse_regression(seed, dimension; bo_mode=bo_mode)
      catch e
          println(e)
          open("tailed_cardinality_errors.txt","a") do io
              println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
          end
      end
  end
end

include("tailed_cardinality_sparse_log_reg.jl")

bo_mode = "afw"
for dimension in [10:5:30;]
    for seed in 1:5
        @show seed, dimension
        for ns in [0.1,1.0]
            for var_A in [1,5]
                try 
                    sparse_log_regression(seed, dimension, ns, var_A; bo_mode=bo_mode)
                catch e
                    println(e)
                    open("sparse_log_reg_tailed_cardinality_errors.txt","a") do io
                        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
                    end
                end
            end
        end
    end
end

include("portfolio.jl")

mode = "mixed"
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

mode = "integer"
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

include("sparse_log_reg.jl")    

bo_mode = "afw"
for dimension in [5:5:20;]
    for seed in 1:3
        @show seed, dimension
        for ns in [0.1,1]
            for var_A in [1,5]
                k = Float64(dimension)
                try 
                    sparse_log_regression(seed, dimension, ns, k, var_A; bo_mode=bo_mode)
                catch e
                    println(e)
                    open("sparse_log_reg_errors.txt","a") do io
                        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
                    end
                end
            end
        end
    end
end

include("sparse_reg.jl")

bo_mode = "afw"
for dimension in [15:1:30;]
   for seed in 1:10
       @show seed, dimension
       try 
           sparse_reg(seed, dimension, 1; bo_mode=bo_mode)
       catch e
           println(e)
           open("sparse_reg_errors.txt","a") do io
               println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
           end
       end
   end
end
