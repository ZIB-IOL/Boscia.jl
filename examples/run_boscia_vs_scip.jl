include("boscia_vs_scip.jl")

# for dimension in [30:5:30;]
#     for seed in 6:7
#         @show seed, dimension
#         boscia_vs_scip(seed, dimension, 2)
#     end
# end

boscia_vs_scip(1, 70, 2)
boscia_vs_scip(2, 55, 2)