include("tailed_cardinality_sparse_log_reg.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
M = parse(Float64, ARGS[3])
var_A = parse(Int64, ARGS[4])
mode = ARGS[5]
#mode = parse(String, ARGS[5])
@show seed, dimension, M, var_A, mode

try 
    tailed_cardinality_sparse_log_reg_boscia(seed, dimension, M, var_A, bo_mode=mode)#, bo_mode="default")
catch e 
    println(e)
    file = "boscia_sparse_log_reg_" * mode * "_" * str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
