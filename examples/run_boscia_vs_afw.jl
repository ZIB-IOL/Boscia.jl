include("boscia_vs_afw.jl")

for dimension in [20:5:50;]
    for seed in 1:7
        @show seed, dimension
        boscia_vs_afw(seed, dimension, 1)
    end
end
