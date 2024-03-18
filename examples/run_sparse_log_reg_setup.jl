include("sparse_log_reg.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
@show seed, dimension
M = parse(Float64, ARGS[3]) 
k = parse(Float64, ARGS[4]) 
var_A = parse(Float64, ARGS[5])

try 
    sparse_log_reg_pavito(seed, dimension, M, k, var_A)
catch e 
    println(e)
    file = "pavito_sparse_log_reg_" * str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
