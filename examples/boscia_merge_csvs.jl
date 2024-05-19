using CSV
using DataFrames

function merge_csvs(;example="sparse_reg", mode="default", seeds=1:10, dimensions=15:30, Ns=[], k=[], var_A=[], M=[])
    # setup df
    name = mode in ["no_as","no_ss","no_as_no_ss"] ? "no_warm_start_" * mode : "boscia_" * mode
    folder = if mode == "default"
                ""
    elseif contains(mode, "hybrid_branching")
        "hybrid_branching"
    else
        mode
    end
    if example == "sparse_reg" 
        if mode == "default"
            df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * "1_15.csv")))
        else
            df =  DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(seeds[1]) * "_" * string(dimensions[1]) * ".csv")))
        end
    elseif occursin("mip_lib", example) 
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(dimensions[1]) * "_" * string(seeds[1]) * ".csv")))
    elseif example == "portfolio_mixed" 
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(dimensions[1]) * "_" * string(seeds[1]) * ".csv")))
    elseif example == "portfolio_integer"
        try 
            df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(dimensions[1]) * "_" * string(seeds[1]) * ".csv")))
        catch e 
            df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(25) * "_" * string(1) * ".csv")))
        end
    elseif example == "sparse_log_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(seeds[1]) * "_" * string(dimensions[1]) * "_" * string(var_A[1]) * "_" * string(dimensions[1]*5) * "_" * string(Ns[1]) * ".csv")))
    elseif example == "poisson_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(seeds[1]) * "_" * string(dimensions[1]) * "_" * string(float(dimensions[1]/2)) * "_" * string(Ns[1]) * ".csv")))
        @show df
    elseif example == "tailed_cardinality"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(dimensions[1]) * "_" * string(seeds[1]) * ".csv")))
    elseif example == "tailed_cardinality_sparse_log_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(seeds[1]) * "_" * string(dimensions[1]) * "_" * string(var_A[1]) * "_" * string(M[1]) * ".csv")))
    else 
        @error "not a valid example"
    end
    select!(df, Not(:termination))
    df[!, "termination"] = ["ALMOST_LOCALLY_SOLVED"]
    select!(df, Not(:time))
    df[!, "time"] = [1800.0]
    deleteat!(df, 1)

    # add results
    if example == "sparse_reg" 
        for seed in seeds 
            for dimension in dimensions 
                try 
                    df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(seed) * "_" * string(dimension) * ".csv")))
                    append!(df, df_temp)
                catch e 
                    println(e)
                end 
            end 
        end
    elseif occursin("mip_lib", example) 
        for seed in seeds 
            for dimension in dimensions 
                try 
                    df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(dimension) * "_" * string(seed) * ".csv")))
                    append!(df, df_temp)
                catch e 
                    println(e)
                end 
            end 
        end
    elseif example == "portfolio_mixed" || example == "portfolio_integer"
        for seed in seeds 
            for dimension in dimensions 
                try 
                    df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(dimension) * "_" * string(seed) * ".csv")))
                    append!(df, df_temp)
                catch e 
                    println(e)
                end
            end 
        end
      #  println("\n")
    elseif example == "sparse_log_reg"
        for dimension in dimensions
            for seed in seeds
                for ns in Ns
                    for var in var_A
                        k = Float64(dimension)
                        try 
                            df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(seed) * "_" * string(dimension) * "_" * string(var) * "_" * string(dimension*5) * "_" * string(ns) * ".csv")))
                            append!(df, df_temp)
                        catch e
                        print(e)
                        end
                    end 
                end 
            end 
        end
    elseif example == "poisson_reg"
        for dimension in dimensions
            p = dimension
            for seed in seeds
                for ns in Ns
                    try 
                        df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(seed) * "_" * string(dimension) * "_" * string(float(p/2)) * "_" * string(ns) * ".csv")))
                        append!(df, df_temp)
                    catch e 
                        println(e)
                    end
                end 
            end 
        end
    elseif example == "tailed_cardinality"
        for dimension in dimensions
            for seed in seeds
                try
                    df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(dimension) * "_" * string(seed) * ".csv")))
                    append!(df, df_temp)
                catch e
                    println(e)
                end
            end
        end
    elseif example == "tailed_cardinality_sparse_log_reg"
        for dimension in dimensions
            for seed in seeds 
                for m in M
                    for var in var_A
                        try
                            df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/Boscia/" * folder * "/" * name * "_" * example * "_" * string(seed) * "_" * string(dimension) * "_" * string(var) * "_" * string(m) * ".csv")))
                            append!(df, df_temp)
                        catch e
                            println(e)
                        end
                    end
                end
            end
        end
    else
        error("Unknown example")
    end

    if occursin("mip_lib", example) 
        example = replace(example, "mip_lib" => "miplib")
    end

    # save csv 
    file_name = joinpath(@__DIR__, "final_csvs/boscia_" * mode * "_" * example * ".csv")
    CSV.write(file_name, df, append=false)

end

modes = ["default", "no_tightening", "global_tightening", "local_tightening", "afw", "no_ss", "no_as", "no_as_no_ss", "strong_branching", "hybrid_branching_1", "hybrid_branching_2", "hybrid_branching_5", "hybrid_branching_10", "hybrid_branching_20"]
for mode in modes

merge_csvs(
    example="sparse_reg", 
    mode = mode,
    seeds=1:10, 
    dimensions=15:30
) 

#=merge_csvs(
    example = "portfolio_mixed", 
    mode= mode, 
    seeds = 1:10, 
    dimensions = 20:5:120
)=# 

merge_csvs(
    example = "portfolio_integer", 
    mode = mode, 
    seeds = 1:10, 
    dimensions = 20:5:120
) 

merge_csvs(
    example = "mip_lib_22433", 
    mode = mode,
    dimensions = 4:8, 
    seeds = 1:3
)

merge_csvs(
    example = "mip_lib_neos5", 
    mode = mode,
    dimensions = 4:8, 
    seeds = 1:3
)

merge_csvs(
    example = "mip_lib_ran14x18-disj-8", 
    mode = mode,
    dimensions = 4:8, 
    seeds = 1:3
)

merge_csvs(
    example = "mip_lib_pg5_34", 
    mode = mode,
    dimensions = 4:8, 
    seeds = 1:3
)

merge_csvs(
    example = "sparse_log_reg", 
    mode = mode, 
    dimensions = [5:5:20;], 
    seeds = 1:3,
    Ns = [0.1,1],
    var_A = [1,5]
) # seed dim Ns/m k var_A

merge_csvs(
    example = "poisson_reg", 
    mode = mode, 
    dimensions = [50:20:100;], 
    seeds = 1:10,
    Ns = [0.1,1,5,10]
) # seed dim p k Ns

merge_csvs(
    example = "tailed_cardinality",
    mode = mode,
    dimensions = 15:30,
    seeds = 1:10,
)

merge_csvs(
    example = "tailed_cardinality_sparse_log_reg",
    mode = mode,
    dimensions = [5:5:20;],
    seeds = 1:3,
    var_A = [1,5],
    M = [0.1,1],
) 
end

merge_csvs(
    example = "mip_lib_22433", 
    mode = "strong_convexity",
    dimensions = 4:8, 
    seeds = 1:3
)

merge_csvs(
    example = "mip_lib_neos5", 
    mode = "strong_convexity",
    dimensions = 4:8, 
    seeds = 1:3
)

merge_csvs(
    example = "mip_lib_ran14x18-disj-8", 
    mode = "strong_convexity",
    dimensions = 4:8, 
    seeds = 1:3
)

merge_csvs(
    example = "mip_lib_pg5_34", 
    mode = "strong_convexity",
    dimensions = 4:8, 
    seeds = 1:3
)
