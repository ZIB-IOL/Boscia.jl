include("sparse_reg.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
mode = ARGS[3]
depth = parse(Int64, ARGS[4])
factor = parse(Float64, ARGS[5])
epsilon = parse(Float64, ARGS[6])
@show seed, dimension, mode, depth

try 
    sparse_reg_boscia(seed, dimension, bo_mode=mode, depth=depth, full_callback=false, dual_gap_decay_factor=factor, fw_epsilon=epsilon)#, bo_mode="default")
catch e 
    println(e)
    file = "boscia_sparse_reg_" * mode * "_" * string(seed) * "_" * string(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
