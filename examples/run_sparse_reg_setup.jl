include("sparse_reg.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
mode = ARGS[3]
depth = parse(Int64, ARGS[4])
@show seed, dimension, mode, depth

try 
    sparse_reg_boscia(seed, dimension, bo_mode=mode, depth=depth, full_callback=true)#, bo_mode="default")
catch e 
    println(e)
    file = "boscia_sparse_reg_" * mode * "_" * string(seed) * "_" * string(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
