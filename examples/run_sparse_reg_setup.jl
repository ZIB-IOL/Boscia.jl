include("sparse_reg.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
mode = ARGS[3]
depth = parse(Int64, ARGS[4])
factor = parse(Float64, ARGS[5])
epsilon = parse(Float64, ARGS[6])
solver = ARGS[7]
@show seed, dimension, mode, depth, solver

try 
    if solver == "Boscia"
        sparse_reg_boscia(seed, dimension, bo_mode=mode, depth=depth, full_callback=false, dual_gap_decay_factor=factor, fw_epsilon=epsilon)#, bo_mode="default")
    elseif solver == "Ipopt"
        sparse_reg_ipopt(seed, dimension)
    elseif solver == "Pavito"
        sparse_reg_pavito(seed, dimension)
    elseif solver == "SHOT"
        sparse_reg_shot(seed, dimension)
    elseif solver == "SCIP"
        sparse_reg_shot(seed, dimension)
    else
        error("Solver is not known")
    end
catch e 
    println(e)
    file = solver * "_sparse_reg_" * mode * "_" * string(seed) * "_" * string(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
