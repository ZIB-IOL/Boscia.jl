include("tailed_cardinality_sparse_log_reg.jl")

bo_mode="boscia"
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

bo_mode = "as"
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

bo_mode = "ss"
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

bo_mode = "as_ss"
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

# bo_mode = "scip_oa"
# for dimension in [10:5:30;]
#     for seed in 1:5
#         @show seed, dimension
#         for ns in [0.1,1.0]
#             for var_A in [1,5]
#                 try 
#                     sparse_log_reg_scip(seed, dimension, ns, var_A)
#                 catch e
#                     println(e)
#                     open("sparse_log_reg_tailed_cardinality_errors.txt","a") do io
#                         println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#                     end
#                 end
#             end
#         end
#     end
# end
