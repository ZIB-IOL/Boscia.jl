include("boscia_vs_scip.jl")

for seed in 1:3
    for dimension in [20, 30, 40, 50]
        @show seed, dimension
        boscia_vs_scip(seed, dimension, 2)
    end
end
