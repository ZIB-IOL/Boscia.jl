include("sparse_log_reg.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
M = parse(Float64, ARGS[3])
k = parse(Float64, ARGS[4])
var_A = parse(Int64, ARGS[5])
mode = ARGS[6]
depth = parse(Int64, ARGS[7])
solver = ARGS[8]
@show seed, dimension, M, k, var_A, mode, depth, solver

try 
    if solver == "Boscia"
        sparse_log_reg_boscia(seed, dimension, M, k, var_A, true, bo_mode=mode, depth=depth)#, bo_mode="default")
    elseif solver == "Ipopt"
        sparse_log_reg_ipopt(seed, dimensionm, M, k, var_A)
    elseif solver == "Pavito"
        sparse_log_reg_pavito(seed, dimension, M, k, var_A)
    elseif solver == "SHOT"
        sparse_log_reg_shot(seed, dimension, M, k, var_A)
    elseif solver == "SCIP"
        sparse_log_reg_scip(seed, dimension, M, k, var_A)
    else
        errror("solver not known")
    end
catch e 
    println(e)
    file = solver * "_sparse_log_reg_" * mode * "_" *str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
