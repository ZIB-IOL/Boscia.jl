include("boscia_vs_afw.jl")

for dimension in [20:5:50;]
    for seed in 1:7
        @show seed, dimension
        boscia_vs_afw(seed, dimension, 1, mode="integer", bo_mode="ss")
    end
end

for dimension in [20:5:50;]
    for seed in 1:7
        @show seed, dimension
        boscia_vs_afw(seed, dimension, 1, mode="integer", bo_mode="as")
    end
end

for dimension in [20:5:50;]
    for seed in 1:7
        @show seed, dimension
        boscia_vs_afw(seed, dimension, 1, mode="integer", bo_mode="as_ss")
    end
end