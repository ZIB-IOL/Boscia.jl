include("poisson_reg.jl")

for dimension in [50:20:200;]
    for seed in 1:10
        for ns in [0.1,1,10,100,1000]
            @show seed, dimension
            poisson(seed, dimension, ns, 1; bo_mode="boscia")
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

for dimension in [50:20:200;]
    for seed in 1:10
        for ns in [0.1,1,10,100,1000]
            @show seed, dimension
            poisson_scip(seed, dimension, ns, 1)
        end
    end
end