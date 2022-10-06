include("boscia_vs_scip.jl")

for seed in 1:3
    for dimension in [20,30,40,50]
        @show seed, dimension
        boscia_vs_scip(seed, dimension, 2)
    end
end

# seed 1, dim 40 scip needs super long or stuck
# seed 2, dim 30 scip needs super long or stuck !! time == 600 == limit
# seed 3, dim 0 scip needs super long or stuck