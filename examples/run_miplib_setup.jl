include("mps-example.jl")

example = ARGS[1]
num_v = parse(Int64, ARGS[2])
seed = parse(Int64, ARGS[3])
mode = ARGS[4]
depth = parse(Int64, ARGS[5])
@show example, seed, num_v, mode, depth

try 
    miplib_boscia(example, seed, num_v, bo_mode=mode, depth=depth, full_callback=true)
    #miplib_ipopt(example, seed, num_v, full_callback=true)
catch e 
    println(e)
    file = "boscia_miplib_" * mode * "_" * str(seed) * "_" * str(num_v)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end