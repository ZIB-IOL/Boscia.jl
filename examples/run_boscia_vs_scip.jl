include("boscia_vs_scip.jl")

for dimension in [20:5:50;]
    for seed in 8:10
        @show seed, dimension
        boscia_vs_scip(seed, dimension, 2)
    end
end

for dimension in [55:5:60;]
    for seed in 1:10
        @show seed, dimension
        boscia_vs_scip(seed, dimension, 2)
    end
end
