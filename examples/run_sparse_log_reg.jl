include("sparse_log_reg.jl")

bo_mode="boscia"
#for dimension in [1:1:30;]
#    for seed in 1:3
#        @show seed, dimension
#        for ns in [0.1,1,5.0,10]
#            k = dimension/5
#            try 
#                sparse_log_regression(seed, dimension, 1; bo_mode=bo_mode)
#            catch e
#                println(e)
#                open("sparse_log_reg_errors.txt","a") do io
#                    println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#                end
#            end
#        end
#    end
#end

#bo_mode = "as"
# for dimension in [1:5:30;]
#     for seed in 1:3
#         @show seed, dimension
#         for ns in [0.1,1,5.0,10]
#             for k in [ceil(dimension/5)]
#                 try 
#                     sparse_log_regression(seed, dimension, 1; bo_mode=bo_mode)
#                 catch e
#                     println(e)
#                     open("sparse_log_reg_errors.txt","a") do io
#                         println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#                     end
#                 end
#             end
#         end
#     end
# end

#bo_mode = "ss"
# for dimension in [1:5:30;]
#     for seed in 1:3
#         @show seed, dimension
#         for ns in [0.1,1,5.0,10]
#             for k in [ceil(dimension/5)]
#                 try 
#                     sparse_log_regression(seed, dimension, 1; bo_mode=bo_mode)
#                 catch e
#                     println(e)
#                     open("sparse_log_reg_errors.txt","a") do io
#                         println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#                     end
#                 end
#             end
#         end
#     end
# end

#bo_mode = "as_ss"
# for dimension in [1:5:30;]
#     for seed in 1:3
#         @show seed, dimension
#         for ns in [0.1,1,5.0,10]
#             for k in [ceil(dimension/5)]
#                 try 
#                     sparse_log_regression(seed, dimension, 1; bo_mode=bo_mode)
#                 catch e
#                     println(e)
#                     open("sparse_log_reg_errors.txt","a") do io
#                         println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#                     end
#                 end
#             end
#         end
#     end
# end

#bo_mode = "afw"
# for dimension in [1:5:30;]
#     for seed in 1:3
#         @show seed, dimension
#         for ns in [0.1,1,5.0,10]
#             for k in [ceil(dimension/5)]
#                 try 
#                     sparse_log_regression(seed, dimension, 1; bo_mode=bo_mode)
#                 catch e
#                     println(e)
#                     open("sparse_log_reg_errors.txt","a") do io
#                         println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#                     end
#                 end
#             end
#         end
#     end
# end

bo_mode = "scip_oa"
for dimension in [1:1:20;]
    for seed in 1:3
        @show seed, dimension
        for ns in [0.1,1,5.0,10]
            k = dimension/5
            try 
                sparse_log_reg_scip(seed, dimension, 1)
            catch e
                println(e)
                open("sparse_log_reg_errors.txt","a") do io
                    println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
                end
            end
        end
    end
end
