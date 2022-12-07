include("poisson_reg.jl")

bo_mode="boscia"
for dimension in [70:20:100;]
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

# for dimension in [20:5:100;]
#     for seed in 1:10
#         @show seed, dimension
#         poisson(seed, dimension, 1; bo_mode="as")
#     end
# end

# for dimension in [20:5:100;]
#     for seed in 1:10
#         @show seed, dimension
#         poisson(seed, dimension, 1; bo_mode="ss")
#     end
# end

# for dimension in [20:5:100;]
#     for seed in 1:10
#         @show seed, dimension
#         poisson(seed, dimension, 1; bo_mode="as_ss")
#     end
# end

# for dimension in [20:5:100;]
#     for seed in 1:10
#         @show seed, dimension
#         poisson(seed, dimension, 1; bo_mode="afw")
#     end
# end

bo_mode="scip_oa"
for dimension in [50:20:100;]
    for seed in 1:10
        for ns in [0.1,1,5,10]
            @show seed, dimension
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
