include("mps-example.jl")

example = ARGS[1]
num_v = parse(Int64, ARGS[2])
seed = parse(Int64, ARGS[3])
@show example, seed, num_v

try 
    miplib_ipopt(seed, num_v, example=example)
catch e 
    println(e)
    file = "ipopt_miplib_" * str(seed) * "_" * str(num_v)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
