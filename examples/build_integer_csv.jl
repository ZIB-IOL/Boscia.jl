using CSV
using DataFrames

df = DataFrame()#dimension=Int64[], time_boscia=Float64[], termination_boscia=String[], time_scip=Float64[], termination_scip=String[], time_afw=Float64[], termination_afw=String[], time_no_ws=Float64[], termination_no_ws=String[], time_no_ss=Float64[], termination_no_ss=String[], time_no_as=Float64[], termination_no_as=String[])

# load boscia 
df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_vs_scip_integer_50.csv")))

indices = [index for index in 1:nrow(df_bs) if isodd(index)]
delete!(df_bs, indices)

df_bs.termination_boscia .= replace.(df_bs.termination_boscia, "Optimal (tree empty)" => "OPTIMAL")
df_bs.termination_boscia .= replace.(df_bs.termination_boscia, "Time limit reached" => "TIME_LIMIT")

df[!,:dimension] = df_bs[!,:dimension]
df[!,:time_boscia] = df_bs[!,:time_boscia]
df[!,:termination_boscia] = df_bs[!,:termination_boscia]
df[!,:time_scip] = df_bs[!,:time_scip]
df[!,:termination_scip] = df_bs[!,:termination_scip]

display(df)
# boscia # no ws # no as # no ss # afw # scip
# dim # time # solved 

file_name = joinpath(@__DIR__, "csv/integer_50.csv")
CSV.write(file_name, df, append=false)