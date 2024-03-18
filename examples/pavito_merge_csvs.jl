using CSV
using DataFrames

function merge_csvs(;example="sparse_reg", seeds=1:10, dimensions=15:30, Ns=[], k=[], var_A=[])
    # setup df
    if example == "sparse_reg" 
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_1_17.csv")))
    elseif occursin("miplib", example) 
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(dimensions[1]) * "_" * string(seeds[1]) * ".csv")))
    elseif example == "portfolio_mixed" 
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(dimensions[1]) * "_" * string(seeds[1]) * ".csv")))
    elseif example == "portfolio_integer"
        try 
            df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(dimensions[1]) * "_" * string(seeds[1]) * ".csv")))
        catch e 
            df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(25) * "_" * string(1) * ".csv")))
        end
    elseif example == "sparse_log_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(seeds[1]) * "_" * string(dimensions[1]) * "_" * string(var_A[1]) * "_" * string(dimensions[1]*5) * "_" * string(Ns[1]) * ".csv")))
    elseif example == "poisson_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(seeds[1]) * "_" * string(dimensions[1]) * "_" * string(dimensions[1]) * "_" * string(dimensions[1]/2) * "_" * string(Ns[1]) * ".csv")))
        @show df
    else @error "not a valid example"
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
                    df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(seed) * "_" * string(dimension) * ".csv")))
                    append!(df, df_temp)
                catch e 
                    println(e)
                end 
            end 
        end
    elseif occursin("miplib", example) 
        for seed in seeds 
            for dimension in dimensions 
                try 
                    df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(dimension) * "_" * string(seed) * ".csv")))
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
                    df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(dimension) * "_" * string(seed) * ".csv")))
                    append!(df, df_temp)
                catch e 
                    println(e)
                end
            end 
        end
    elseif example == "sparse_log_reg"
        for dimension in dimensions
            for seed in seeds
                for ns in Ns
                    for var in var_A
                        k = Float64(dimension)
                        try 
                            df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(seed) * "_" * string(dimension) * "_" * string(var) * "_" * string(dimension*5) * "_" * string(ns) * ".csv")))
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
            k = dimension/2
            p = dimension
            for seed in seeds
                for ns in Ns
                    try 
                        df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(seed) * "_" * string(dimension) * "_" * string(p) * "_" * string(k) * "_" * string(ns) * ".csv")))
                        append!(df, df_temp)
                    catch e 
                        println(e)
                    end
                end 
            end 
        end
    end

    # save csv 
    file_name = joinpath(@__DIR__, "csv/pavito_" * example * ".csv")
    CSV.write(file_name, df, append=false)

end

# merge_csvs()

merge_csvs(
    example = "portfolio_mixed", 
    seeds = 1:10, 
    dimensions = 20:5:120
)

merge_csvs(
    example = "portfolio_integer", 
    seeds = 1:10, 
    dimensions = 20:5:120
)

merge_csvs(
    example = "miplib_22433", 
    dimensions = 4:8, 
    seeds = 1:3
)

merge_csvs(
    example = "miplib_neos5", 
    dimensions = 4:8, 
    seeds = 1:3
)

merge_csvs(
    example = "miplib_ran14x18-disj-8", 
    dimensions = 4:8, 
    seeds = 1:3
)

merge_csvs(
    example = "miplib_pg5_34", 
    dimensions = 4:8, 
    seeds = 1:3
)

merge_csvs(
    example = "sparse_log_reg", 
    dimensions = [5:5:20;], 
    seeds = 1:10,
    Ns = [0.1,1],
    var_A = [1.0,5.0]
) # seed dim Ns/m k var_A

merge_csvs(
    example = "poisson_reg", 
    dimensions = [50:20:100;], 
    seeds = 1:10,
    Ns = [0.1,1,5,10]
) # seed dim p k Ns
