include("sparse_reg.jl")

#for dimension in [18:1:30;]
#    for seed in 1:10
#        @show seed, dimension
#        sparse_reg(seed, dimension, 1; bo_mode="boscia")
#    end
#end

# for dimension in [1:1:30;]
#     for seed in 1:10
#         @show seed, dimension
#         sparse_reg(seed, dimension, 1; bo_mode="as")
#     end
# end

# for dimension in [1:1:30;]
#     for seed in 1:10
#         @show seed, dimension
#         sparse_reg(seed, dimension, 1; bo_mode="ss")
#     end
# end

# for dimension in [1:1:30;]
#     for seed in 1:10
#         @show seed, dimension
#         sparse_reg(seed, dimension, 1; bo_mode="as_ss")
#     end
# end

# for dimension in [1:1:30;]
#     for seed in 1:10
#         @show seed, dimension
#         sparse_reg(seed, dimension, 1; bo_mode="afw")
#     end
# end

for dimension in [18:1:30;]
    for seed in 1:10
        @show seed, dimension
        sparse_reg_scip(seed, dimension, 1)
    end
end
