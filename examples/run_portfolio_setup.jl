include("portfolio.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
mode = ARGS[3]
@show seed, dimension, mode

try 
    portfolio_shot(seed, dimension; mode=mode)
catch e 
    println(e)
    file = "shot_portfolio_" * str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end