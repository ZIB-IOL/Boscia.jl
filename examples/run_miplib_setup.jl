include("mip-examples.jl")

example = ARGS[1]
num_v = parse(Int64, ARGS[2])
seed = parse(Int64, ARGS[3])
@show example, num_v, seed

try 
    miplib_shot(example, num_v, seed)
catch e 
    println(e)
    file = "shot_miplib_" *example * "_" * str(seed) * "_" * str(num_v)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
