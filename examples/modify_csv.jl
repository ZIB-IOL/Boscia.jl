using CSV
using DataFrames

file_name = joinpath(@__DIR__, "csv/afw_sparse_reg.csv")
df_afw = DataFrame(CSV.File(file_name))

push!(df_afw, [4, 15, 75, 3.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)
push!(df_afw, [5, 15, 75, 3.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)
push!(df_afw, [7, 15, 75, 3.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)
push!(df_afw, [8, 15, 75, 3.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)

push!(df_afw, [1, 16, 80, 4.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)
push!(df_afw, [8, 16, 80, 4.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)

push!(df_afw, [10, 17, 85, 4.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)

push!(df_afw, [1, 18, 90, 4.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)
push!(df_afw, [5, 18, 90, 4.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)

push!(df_afw, [5, 19, 95, 4.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)

push!(df_afw, [1, 20, 100, 4.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)
push!(df_afw, [2, 20, 100, 4.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)
push!(df_afw, [5, 20, 100, 4.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)
push!(df_afw, [7, 20, 100, 4.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)
push!(df_afw, [8, 20, 100, 4.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)

push!(df_afw, [1, 21, 105, 5.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)

push!(df_afw, [10, 22, 110, 5.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)

push!(df_afw, [10, 23, 115, 5.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)

push!(df_afw, [10, 24, 120, 5.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)

push!(df_afw, [10, 25, 125, 5.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)

push!(df_afw, [6, 26, 130, 6.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)

push!(df_afw, [6, 28, 140, 6.0, 1800, Inf, Inf, Inf, "AssertionError", Inf], promote=true)

CSV.write(file_name, df_afw, append=true)
