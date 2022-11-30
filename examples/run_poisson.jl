include("poisson_reg.jl")

for dimension in [100:20:1000;]
    for seed in 1:10
        @show seed, dimension
        poisson(seed, dimension, 1; bo_mode="boscia")
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

for dimension in [100:20:1000;]
    for seed in 1:10
        @show seed, dimension
        poisson_scip(seed, dimension, 1)
    end
end
