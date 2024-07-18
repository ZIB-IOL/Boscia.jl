include("portfolio.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
set = ARGS[3]
mode = ARGS[4]
depth = parse(Int64, ARGS[5])
solver = ARGS[6]
@show seed, dimension, mode, solver

try 
    if solver == "Boscia"
        portfolio_boscia(seed, dimension, mode=set, bo_mode=mode, depth=depth)#, bo_mode="default")
    elseif solver == "Ipopt"
        portfolio_ipopt(seed, dimension, mode=set)
    elseif solver == "Pavito"
        portfolio_pavito(seed, dimension, mode=set)
    elseif solver == "SCIP"
        portfolio_scip(seed, dimension, mode=set)
    elseif solver == "SHOT"
        portfolio_shot(seed, dimension, mode=set)
    else 
        error("solver not known")
    end
catch e 
    println(e)
    file = solver * "_portfolio_" * string(set) * "_" * mode * "_" * str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
