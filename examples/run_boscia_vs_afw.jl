include("boscia_vs_afw.jl")

# integer
for dimension in [20:5:60;]
    for seed in 1:10
        @show seed, dimension
        boscia_vs_afw(seed, dimension, 1, mode="integer", bo_mode="ss")
    end
end

for dimension in [20:5:60;]
    for seed in 1:10
        @show seed, dimension
        boscia_vs_afw(seed, dimension, 1, mode="integer", bo_mode="as_ss")
    end
end

for dimension in [20:5:60;]
    for seed in 1:10
        @show seed, dimension
        boscia_vs_afw(seed, dimension, 1, mode="integer", bo_mode="as")
    end
end

for dimension in [20:5:60;]
    for seed in 1:10
        @show seed, dimension
        boscia_vs_afw(seed, dimension, 1, mode="integer", bo_mode="afw")
    end
end

# mixed
for dimension in [20:5:80;]
    for seed in 1:10
        @show seed, dimension
        boscia_vs_afw(seed, dimension, 1, mode="mixed", bo_mode="ss")
    end
end

for dimension in [20:5:80;]
    for seed in 1:10
        @show seed, dimension
        boscia_vs_afw(seed, dimension, 1, mode="mixed", bo_mode="as_ss")
    end
end

for dimension in [20:5:80;]
    for seed in 1:10
        @show seed, dimension
        boscia_vs_afw(seed, dimension, 1, mode="mixed", bo_mode="as")
    end
end

for dimension in [20:5:80;]
    for seed in 1:10
        @show seed, dimension
        boscia_vs_afw(seed, dimension, 1, mode="mixed", bo_mode="afw")
    end
end