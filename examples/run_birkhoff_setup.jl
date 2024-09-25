include("birkhoff.jl")

mode = ARGS[1]
dimension = parse(Int64, ARGS[2])
seed = parse(Int64, ARGS[3])

@show mode, dimension, seed

try 
    birkhoff_boscia(seed, dimension, mode=mode)
catch e 
    println(e)
    file = "Boscia_birkhoff_" * mode * "_" * string(seed) * "_" * string(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end