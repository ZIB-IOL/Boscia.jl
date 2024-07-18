include("tailed_cardinality.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
#mode = parse(String, ARGS[3])
mode = ARGS[3]
depth = parse(Int64, ARGS[4])
solver = ARGS[5]
@show seed, dimension, mode, solver

try 
    if solver == "Boscia"
        tailed_cardinality_sparse_reg_boscia(seed, dimension, bo_mode=mode, depth=depth)#, bo_mode="default")
    elseif solver == "Ipopt"
        tailed_cardinality_sparse_reg_ipopt(seed, dimension)
    elseif solver == "Pavito"
        tailed_cardinality_sparse_reg_pavito(seed, dimension)
    elseif solver == "SHOT"
        tailed_cardinality_sparse_reg_shot(seed, dimension)
    elseif solver == "SCIP"
        tailed_cardinality_sparse_reg_scip(seed, dimension)
    else
        error("Solver not known")
    end
catch e 
    println(e)
    file = solver * "_sparse_reg_" * mode * "_" * str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
