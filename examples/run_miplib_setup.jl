include("mps-example.jl")

example = ARGS[1]
num_v = parse(Int64, ARGS[2])
seed = parse(Int64, ARGS[3])
mode = ARGS[4]
depth = parse(Int64, ARGS[5])
solver = ARGS[6]
@show example, seed, num_v, mode, depth, solver

try 
    if solver == "Boscia"
        miplib_boscia(example, seed, num_v, bo_mode=mode, depth=depth, full_callback=true)
    elseif solver == "Ipopt"
        miplib_ipopt(example, seed, num_v, full_callback=true)
    elseif solver == "Pavito"
        miplib_pavito(example,seed, num_v)
    elseif solver == "SCIP"
        miplib_scip(example, seed, num_v)
    elseif solver == "SHOT"
        miplib_shot(example, seed, num_v)
    else
        error("solver not known!")
    end
catch e 
    println(e)
    file = solver * "_miplib_" * mode * "_" * str(seed) * "_" * str(num_v)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
