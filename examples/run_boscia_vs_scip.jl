include("boscia_vs_scip.jl")

#for dimension in [20:5:60;]
#    for seed in 1:10
#        @show seed, dimension
#        boscia_vs_scip("integer", seed, dimension, 1)
#    end
#end

#boscia_vs_scip("mixed", 8, 70, 1)
#boscia_vs_scip("mixed", 9, 70, 1)
#boscia_vs_scip("mixed", 10, 70, 1)
for dimension in [85:5:100;]
    for seed in 1:10
        @show seed, dimension
        boscia_vs_scip("mixed", seed, dimension, 1; scip_oa=false)
    end
end
