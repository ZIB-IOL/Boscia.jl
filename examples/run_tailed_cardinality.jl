include("tailed_cardinality.jl")

# boscia methods and scip oa
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

bo_mode = "scip_oa"
for dimension in [50:10:150;]
    for seed in 1:10
        @show seed, dimension
        try 
            sparse_reg_scip(seed, dimension)
        catch e
            println(e)
            open("tailed_cardinality_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

# different tightening methods
bo_mode = "local_tightening"
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

bo_mode = "global_tightening"
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

bo_mode = "no_tightening"
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



