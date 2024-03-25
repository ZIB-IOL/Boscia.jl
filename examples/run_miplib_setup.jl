include("mps-example.jl")

example = ARGS[1]
num_v = parse(Int64, ARGS[2])
seed = parse(Int64, ARGS[3])
@show example, seed, num_v

try 
    miplib_boscia(seed, num_v, bo_mode="default", example=example)
catch e 
    println(e)
    file = "boscia_miplib_" * str(seed) * "_" * str(num_v)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
