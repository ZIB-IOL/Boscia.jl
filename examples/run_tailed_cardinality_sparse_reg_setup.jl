include("tailed_cardinality.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
@show seed, dimension

try 
    tailed_cardinality_sparse_reg_scip(seed, dimension)#, bo_mode="default")
catch e 
    println(e)
    file = "scip_sparse_reg_" * str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
