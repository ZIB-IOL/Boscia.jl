include("poisson_reg.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
ns = parse(Float64, ARGS[3])
mode = ARGS[4]
depth = parse(Int64, ARGS[5])
solver = ARGS[6]
@show seed, dimension, ns, solver

try 
    if solver == "Boscia"
        poisson_reg_boscia(seed, dimension, ns, true, bo_mode=mode, depth=depth)#, bo_mode="default")
    elseif solver == "Ipopt"
        poisson_reg_ipopt(seed, dimension, ns)
    elseif solver == "SHOT"
        poisson_reg_shot(seed, dimension, ns)  # check job 2317211 (1, 90, 1.0)
    elseif solver == "SCIP"
        poisson_reg_scip(seed, dimension, ns)
    elseif solver == "Pavito"
        poisson_reg_pavito(seed, dimension, ns)
    else
        error("Solver not known")
    end
catch e 
    println(e)
    file =  solver * "_poisson_" * mode * "_" * str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end
