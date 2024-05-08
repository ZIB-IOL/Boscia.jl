include("tailed_cardinality_sparse_log_reg.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
M = parse(Float64, ARGS[3])
var_A = parse(Int64, ARGS[4])
@show seed, dimension, M, var_A

try 
    tailed_cardinality_sparse_log_reg_boscia(seed, dimension, M, var_A)#, bo_mode="default")
catch e 
    println(e)
    file = "boscia_sparse_log_reg_" * str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
