include("poisson_reg.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
ns = parse(Float64, ARGS[3])
@show seed, dimension, ns

try 
    poisson_boscia(seed, dimension, ns, bo_mode="default")
catch e 
    println(e)
    file = "boscia_poisson_" * str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
