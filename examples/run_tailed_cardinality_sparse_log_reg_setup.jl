include("tailed_cardinality_sparse_log_reg.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
M = parse(Float64, ARGS[3])
var_A = parse(Int64, ARGS[4])
mode = ARGS[5]
depth = parse(Int64, ARGS[6])
solver = ARSG[7]
@show seed, dimension, M, var_A, mode, solver

try 
    if solver == "Boscia"
        tailed_cardinality_sparse_log_reg_boscia(seed, dimension, M, var_A, bo_mode=mode, depth=depth)#, bo_mode="default")
    elseif solver == "Ipopt"
        tailed_cardinality_sparse_log_reg_ipopt(seed, dimension, M, var_A)
    elseif solver == "Pavito"
        tailed_cardinality_sparse_log_reg_pavito(seed, dimension, M, var_A)
    elseif solver == "SHOT"
        tailed_cardinality_sparse_log_reg_shot(seed, dimension, M, var_A)
    elseif solver == "SCIP"
        tailed_cardinality_sparse_log_reg_scip(seed, dimension, M, var_A)
    else
        error("Solver not known")
    end
catch e 
    println(e)
    file = solver * "_sparse_log_reg_" * mode * "_" * str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
