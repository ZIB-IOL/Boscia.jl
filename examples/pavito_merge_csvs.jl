using CSV
using DataFrames

function merge_csvs(example="sparse_reg", seeds=1:2, dimensions=16:30)
    # setup df
    if example == "sparse_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(seeds[1]) * "_" * string(dimensions[1]) * ".csv")))
        deleteat!(df, 1)

        # add results
        for seed in seeds 
            for dimension in dimensions 
                df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(seed) * "_" * string(dimension) * ".csv")))
                append!(df, df_temp)
            end 
        end
    elseif example == "portfolio_mixed" || example == "portfolio_integer"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(dimensions[1]) * "_" * string(seeds[1]) * ".csv")))
        deleteat!(df, 1)

        # add results
        for seed in seeds 
            for dimension in dimensions 
                try 
                    df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/pavito_" * example * "_" * string(dimension) * "_" * string(seed) * ".csv")))
                    append!(df, df_temp)
                catch e 
                end
            end 
        end
    end

    # save csv 
    file_name = joinpath(@__DIR__, "csv/pavito_" * example * ".csv")
    CSV.write(file_name, df, append=false)

end

# merge_csvs()
# merge_csvs("portfolio_mixed", 1:1, 20:5:120)
# merge_csvs("portfolio_integer", 1:1, 20:5:120)