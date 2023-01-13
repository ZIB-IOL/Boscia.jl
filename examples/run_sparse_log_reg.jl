include("sparse_log_reg.jl")    

bo_mode="boscia"
for dimension in [5:5:10;]
    for seed in 1:10
        @show seed, dimension
        for ns in [0.1,1,5.0]
            for var_A in [1,5,10]
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

# bo_mode = "as"
# for dimension in [10:2:20;]
#     for seed in 1:3
#         @show seed, dimension
#         for ns in [0.1,1,5.0,10]
#             k = Float64(dimension)
#             try 
#                 sparse_log_regression(seed, dimension, ns, k, var_A; bo_mode=bo_mode)
#             catch e
#                 println(e)
#                 open("sparse_log_reg_errors.txt","a") do io
#                     println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#                 end
#             end
#         end
#     end
# end

# bo_mode = "ss"
# for dimension in [10:2:20;]
#     for seed in 1:3
#         @show seed, dimension
#         for ns in [0.1,1,5.0,10]
#             k = Float64(dimension)
#             try 
#                 sparse_log_regression(seed, dimension, ns, k, var_A; bo_mode=bo_mode)
#             catch e
#                 println(e)
#                 open("sparse_log_reg_errors.txt","a") do io
#                     println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#                 end
#             end
#         end
#     end
# end

# bo_mode = "as_ss"
# for dimension in [10:2:20;]
#     for seed in 1:3
#         @show seed, dimension
#         for ns in [0.1,1,5.0,10]
#             k = Float64(dimension)
#             try 
#                 sparse_log_regression(seed, dimension, ns, k, var_A; bo_mode=bo_mode)
#             catch e
#                 println(e)
#                 open("sparse_log_reg_errors.txt","a") do io
#                     println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#                 end
#             end
#         end
#     end
# end

# bo_mode = "afw"
# for dimension in [10:2:20;]
#     for seed in 1:3
#         @show seed, dimension
#         for ns in [0.1,1,5.0,10]
#             k = Float64(dimension)
#             try 
#                 sparse_log_regression(seed, dimension, ns, k, var_A; bo_mode=bo_mode)
#             catch e
#                 println(e)
#                 open("sparse_log_reg_errors.txt","a") do io
#                     println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#                 end
#             end
#         end
#     end
# end

bo_mode = "scip_oa"
for dimension in [5:5:10;]
    for seed in 1:1
        @show seed, dimension
        for ns in [0.1,1,5.0]
            for var_A in [5,10]
                k = Float64(dimension)
                try 
                    sparse_log_reg_scip(seed, dimension, ns, k, var_A)
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
