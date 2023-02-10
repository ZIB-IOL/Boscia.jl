include("tailed_cardinality.jl")

bo_mode="boscia"
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

bo_mode = "as"
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

bo_mode = "ss"
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

bo_mode = "as_ss"
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

# bo_mode = "scip_oa"
# for dimension in [50:10:150;]
#     for seed in 1:10
#         @show seed, dimension
#         try 
#             sparse_reg_scip(seed, dimension)
#         catch e
#             println(e)
#             open("tailed_cardinality_errors.txt","a") do io
#                 println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end
#=
bo_mode = "ipopt"
for dimension in [50:10:150;]
    for seed in 1:10
        @show seed, dimension
        try 
            sparse_reg_ipopt(seed, dimension)
        catch e
            println(e)
            open("tailed_cardinality_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end=#

