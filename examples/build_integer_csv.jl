using CSV
using DataFrames
using Statistics

df = DataFrame()

# load boscia and scip oa
df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_vs_scip_integer_50.csv")))
indices = [index for index in 1:nrow(df_bs) if isodd(index)]
delete!(df_bs, indices)

df_bs.termination_boscia .= replace.(df_bs.termination_boscia, "Optimal (tree empty)" => "OPTIMAL")
df_bs.termination_boscia .= replace.(df_bs.termination_boscia, "Time limit reached" => "TIME_LIMIT")
termination_boscia = [row == "OPTIMAL" ? 1 : 0 for row in df_bs[!,:termination_boscia]]
termination_scip = [row == "OPTIMAL" ? 1 : 0 for row in df_bs[!,:termination_scip]]

df[!,:dimension] = df_bs[!,:dimension]
df[!,:time_boscia] = df_bs[!,:time_boscia]
df[!,:termination_boscia] = termination_boscia
df[!,:time_scip] = df_bs[!,:time_scip]
df[!,:termination_scip] = termination_scip

# load afw
df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_integer_50.csv")))
delete!(df_afw, indices)
df_afw.termination_afw .= replace.(df_afw.termination_afw, "Optimal (tree empty)" => "OPTIMAL")
df_afw.termination_afw .= replace.(df_afw.termination_afw, "Time limit reached" => "TIME_LIMIT")
termination_afw = [row == "OPTIMAL" ? 1 : 0 for row in df_afw[!,:termination_afw]]

df[!,:time_afw] = df_afw[!,:time_afw]
df[!,:termination_afw] = termination_afw

# load without as, without ss
df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_integer_50.csv")))
df_no_ws.termination_afw .= replace.(df_no_ws.termination_afw, "Optimal (tree empty)" => "OPTIMAL")
df_no_ws.termination_afw .= replace.(df_no_ws.termination_afw, "Time limit reached" => "TIME_LIMIT")
termination_no_ws = [row == "OPTIMAL" ? 1 : 0 for row in df_no_ws[!,:termination_afw]]

df[!,:time_no_ws] = df_no_ws[!,:time_afw]
df[!,:termination_no_ws] = termination_no_ws

# load without ss
df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_ss_integer_50.csv")))
df_no_ss.termination_afw .= replace.(df_no_ss.termination_afw, "Optimal (tree empty)" => "OPTIMAL")
df_no_ss.termination_afw .= replace.(df_no_ss.termination_afw, "Time limit reached" => "TIME_LIMIT")
termination_no_ss = [row == "OPTIMAL" ? 1 : 0 for row in df_no_ss[!,:termination_afw]]

df[!,:time_no_ss] = df_no_ss[!,:time_afw]
df[!,:termination_no_ss] = termination_no_ss

# load without as
df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_integer_50.csv")))
df_no_as.termination_afw .= replace.(df_no_as.termination_afw, "Optimal (tree empty)" => "OPTIMAL")
df_no_as.termination_afw .= replace.(df_no_as.termination_afw, "Time limit reached" => "TIME_LIMIT")
termination_no_as = [row == "OPTIMAL" ? 1 : 0 for row in df_no_as[!,:termination_afw]]

df[!,:time_no_as] = df_no_as[!,:time_afw]
df[!,:termination_no_as] = termination_no_as

# group by dimension
gdf = combine(
    groupby(df, :dimension), 
    :time_boscia => mean, :termination_boscia => sum,
    :time_scip => mean, :termination_scip => sum,
    :time_no_ws => mean, :termination_no_ws => sum,
    :time_no_as => mean, :termination_no_as => sum,
    :time_no_ss => mean, :termination_no_ss => sum,
    :time_afw => mean, :termination_afw => sum,
    )

# display(gdf)
# boscia # no ws # no as # no ss # afw # scip
# dim # time # solved 

file_name = joinpath(@__DIR__, "csv/integer_50.csv")
CSV.write(file_name, gdf, append=false)