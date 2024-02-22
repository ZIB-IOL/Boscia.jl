using CSV
using DataFrames

function merge_csvs(example="sparse_reg", seeds=1:1, dimensions=15:30)
    # setup df
    if example == "sparse_reg" 
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_1_17.csv")))
    elseif occursin("miplib", example) 
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(seeds[1]) * "_" * string(dimensions[1]) * ".csv")))
    elseif example == "portfolio_mixed" 
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(dimensions[1]) * "_" * string(seeds[1]) * ".csv")))
    elseif example == "portfolio_integer"
        try 
            df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(dimensions[1]) * "_" * string(seeds[1]) * ".csv")))
        catch e 
            df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(25) * "_" * string(1) * ".csv")))
        end 
    else @error "not a valid example"
    end
    select!(df, Not(:termination))
    df[!, "termination"] = ["ALMOST_LOCALLY_SOLVED"]
    select!(df, Not(:time))
    df[!, "time"] = [1800.0]
    deleteat!(df, 1)

    # add results
    if example == "sparse_reg" || occursin("miplib", example) 
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
    end

    # save csv 
    file_name = joinpath(@__DIR__, "csv/pavito_" * example * ".csv")
    CSV.write(file_name, df, append=false)

end

merge_csvs()
merge_csvs("portfolio_mixed", 1:1, 20:5:120)
merge_csvs("portfolio_integer", 1:1, 20:5:120)
merge_csvs("miplib_22433", 4:8, 1:1)
merge_csvs("miplib_neos5", 4:8, 1:1)
merge_csvs("miplib_ran14x18-disj-8", 4:8, 1:1)