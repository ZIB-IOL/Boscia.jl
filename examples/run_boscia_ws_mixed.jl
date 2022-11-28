include("boscia_vs_afw.jl")


#for dimension in [85:5:100;]
#    for seed in 1:10
#        @show seed, dimension
#        boscia_vs_afw(seed, dimension, 1, mode="mixed", bo_mode="ss")
#    end
#end

#for dimension in [85:5:100;]
#    for seed in 1:10
#        @show seed, dimension
#        boscia_vs_afw(seed, dimension, 1, mode="mixed", bo_mode="as")
#    end
#end

for dimension in [85:5:100;]
    for seed in 1:10
        @show seed, dimension
        boscia_vs_afw(seed, dimension, 1, mode="mixed", bo_mode="afw")
    end
end

