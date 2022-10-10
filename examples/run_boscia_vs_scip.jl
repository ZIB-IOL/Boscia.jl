include("boscia_vs_scip.jl")

for seed in 2:2
    for dimension in [20]
        @show seed, dimension
        boscia_vs_scip(seed, dimension, 1)
    end
end

# seed 1, dim 40 scip needs super long or stuck
# seed 2, dim 30 scip needs super long or stuck !! time == 600 == limit
# seed 3, dim 0 scip needs super long or stuck
