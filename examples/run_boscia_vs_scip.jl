include("boscia_vs_scip.jl")

for dimension in [20:5:60;]
    for seed in 1:10
        @show seed, dimension
        boscia_vs_scip("integer", seed, dimension, 1)
    end
end

for dimension in [20:5:80;]
    for seed in 1:10
        @show seed, dimension
        boscia_vs_scip("mixed", seed, dimension, 1)
    end
end