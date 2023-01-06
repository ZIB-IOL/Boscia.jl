include("sparse_reg.jl")

# bo_mode="boscia"
# for dimension in [15:1:30;]
#     for seed in 1:10
#         @show seed, dimension
#         try 
#             sparse_reg(seed, dimension, 1; bo_mode=bo_mode)
#         catch e
#             println(e)
#             open("sparse_reg_errors.txt","a") do io
#                 println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end

#bo_mode = "as"
#for dimension in [15:1:30;]
#    for seed in 1:10
#        @show seed, dimension
#        try 
#            sparse_reg(seed, dimension, 1; bo_mode=bo_mode)
#        catch e
#            println(e)
#            open("sparse_reg_errors.txt","a") do io
#                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#            end
#        end
#    end
#end

#bo_mode = "ss"
#for dimension in [15:1:30;]
#    for seed in 1:10
#        @show seed, dimension
#        try 
#            sparse_reg(seed, dimension, 1; bo_mode=bo_mode)
#        catch e
#            println(e)
#            open("sparse_reg_errors.txt","a") do io
#                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#            end
#        end
#    end
#end

#bo_mode = "as_ss"
#for dimension in [15:1:30;]
#    for seed in 1:10
#        @show seed, dimension
#        try 
#            sparse_reg(seed, dimension, 1; bo_mode=bo_mode)
#        catch e
#            println(e)
#            open("sparse_reg_errors.txt","a") do io
#                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#            end
#        end
#    end
#end

#bo_mode = "afw"
#for dimension in [15:1:30;]
#    for seed in 1:10
#        @show seed, dimension
#        try 
#            sparse_reg(seed, dimension, 1; bo_mode=bo_mode)
#        catch e
#            println(e)
#            open("sparse_reg_errors.txt","a") do io
#                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#            end
#        end
#    end
#end

#=bo_mode = "scip_oa"
for dimension in [16:1:30;]
    for seed in 1:10
        for tol in [1e-7, 1e-9, 1e-11]
            @show seed, dimension
            try 
                sparse_reg_scip(seed, dimension, 1; tol=tol)
            catch e
                println(e)
                open("sparse_reg_errors.txt","a") do io
                    println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
                end
            end
        end
    end
end=#

bo_mode = "ipopt"
for dimension in [18] #[15:1:30;]
    for seed in 1:10 #1:10
        @show seed, dimension
        try
            sparse_reg_ipopt(seed, dimension, 1)
        catch e
            println(e)
            open("sparse_reg_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end