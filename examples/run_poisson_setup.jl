include("poisson_reg.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
ns = parse(Float64, ARGS[3])

@show seed, dimension, ns

try 
    poisson_reg_pavito(seed, dimension, ns)
catch e 
    println(e)
    file = "pavito_poisson_reg_" * str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
