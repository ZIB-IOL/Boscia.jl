include("poisson_reg.jl")

bo_mode="boscia"
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

#=bo_mode="as"
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

bo_mode="ss"
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

bo_mode="as_ss"
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


bo_mode="scip_oa"
for dimension in [50:20:100;]#[50:20:100;]
    for seed in 4:10
        for ns in [0.1,1.0,5.0,10.0]
            @show seed, dimension, ns
            try 
                poisson_scip(seed, dimension, ns, 1)            
            catch e 
                println(e)
                open("poisson_errors.txt","a") do io
                    println(io, seed, " ", dimension, " ", ns, " ", bo_mode, " : ", e)
                end
            end
        end
    end
end
=#

# bo_mode="ipopt"
# for dimension in [50:20:100;]#[50:20:100;]
#     for seed in 1:10#1:10
#         for ns in [0.1,1,5,10]
#             @show seed, dimension
#             try 
#                 poisson_ipopt(seed, dimension, ns, 1)            
#             catch e
#                 println(e)
#                 open("poisson_errors.txt","a") do io
#                     println(io, seed, " ", dimension, " ", ns, " ", bo_mode, " : ", e)
#                 end
#             end
#         end
#     end
# end 
