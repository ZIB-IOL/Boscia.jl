include("portfolio.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
set = ARGS[3]
mode = ARGS[4]
depth = parse(Int64, ARGS[5])
@show seed, dimension, mode

try 
    portfolio_pavito(seed, dimension, mode=set, bo_mode=mode, depth=depth)#, bo_mode="default")
catch e 
    println(e)
    file = "pavito_portfolio_" * string(mode) * "_" * str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
