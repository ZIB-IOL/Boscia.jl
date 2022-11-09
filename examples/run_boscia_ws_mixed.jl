include("boscia_vs_afw.jl")


#for dimension in [70:5:80;]
#    for seed in 1:4
#        @show seed, dimension
#        boscia_vs_afw(seed, dimension, 1, mode="mixed", bo_mode="ss")
#    end
#end

#for dimension in [80:5:80;]
#    for seed in 3:4
#        @show seed, dimension
#        boscia_vs_afw(seed, dimension, 1, mode="mixed", bo_mode="as")
#    end
#end

for dimension in [80:5:80;]
    for seed in 1:4
        @show seed, dimension
        boscia_vs_afw(seed, dimension, 1, mode="mixed", bo_mode="as_ss")
    end
end

boscia_vs_afw(3, 60, 1, mode="mixed", bo_mode="as_ss")
boscia_vs_afw(4, 60, 1, mode="mixed", bo_mode="as_ss")
boscia_vs_afw(4, 75, 1, mode="mixed", bo_mode="as_ss")

boscia_vs_afw(4, 75, 1, mode="mixed", bo_mode="as")
boscia_vs_afw(2, 80, 1, mode="mixed", bo_mode="as")
boscia_vs_afw(4, 80, 1, mode="mixed", bo_mode="as")

boscia_vs_afw(4, 65, 1, mode="mixed", bo_mode="ss")
