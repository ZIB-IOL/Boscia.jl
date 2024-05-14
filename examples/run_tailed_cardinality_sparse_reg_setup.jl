include("tailed_cardinality.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
#mode = parse(String, ARGS[3])
mode = ARGS[3]
depth = parse(Int64, ARGS[4])
@show seed, dimension, mode

try 
    tailed_cardinality_sparse_reg_boscia(seed, dimension, bo_mode=mode, depth=depth)#, bo_mode="default")
catch e 
    println(e)
    file = "boscia_sparse_reg_" * mode * "_" * str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
