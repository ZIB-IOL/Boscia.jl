include("boscia_vs_afw.jl")

for dimension in [20:5:80;]
    for seed in 1:4
        @show seed, dimension
        boscia_vs_afw(seed, dimension, 1, mode="mixed", bo_mode="ss")
    end
end

