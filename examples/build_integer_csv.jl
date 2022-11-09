using CSV
using DataFrames
using Statistics

function build_csv(mode)
    df = DataFrame()

    # load data
    if mode == "integer"
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
    
    elseif mode == "mixed"
        # load afw
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_mixed_50.csv")))
        df_afw.termination_afw .= replace.(df_afw.termination_afw, "Optimal (tree empty)" => "OPTIMAL")
        df_afw.termination_afw .= replace.(df_afw.termination_afw, "Time limit reached" => "TIME_LIMIT")
        termination_afw = [row == "OPTIMAL" ? 1 : 0 for row in df_afw[!,:termination_afw]]

        df[!,:time_afw] = df_afw[!,:time_afw]
        df[!,:termination_afw] = termination_afw

        println("length afw ", length(termination_afw))

        # load without ss
        df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_ss_mixed_50.csv")))
        df_no_ss.termination_afw .= replace.(df_no_ss.termination_afw, "Optimal (tree empty)" => "OPTIMAL")
        df_no_ss.termination_afw .= replace.(df_no_ss.termination_afw, "Time limit reached" => "TIME_LIMIT")
        termination_no_ss = [row == "OPTIMAL" ? 1 : 0 for row in df_no_ss[!,:termination_afw]]

        # should be 52
        println("length no ss ", length(termination_no_ss))

        df[!,:time_no_ss] = df_no_ss[!,:time_afw]
        df[!,:termination_no_ss] = termination_no_ss        
    end 

    function geo_mean(group)
        prod = 1
        n = length(group)
        for element in group
            prod = prod * element
        end
        return prod^(1/n)
    end

    # group by dimension
    gdf = combine(
        groupby(df, :dimension), 
        :time_boscia => geo_mean, :termination_boscia => sum,
        :time_scip => geo_mean, :termination_scip => sum,
        :time_no_ws => geo_mean, :termination_no_ws => sum,
        :time_no_as => geo_mean, :termination_no_as => sum,
        :time_no_ss => geo_mean, :termination_no_ss => sum,
        :time_afw => geo_mean, :termination_afw => sum,
        nrow, renamecols=false
        )

    # remove underscore in headers for LaTex
    rename!(gdf,
        :time_boscia => :timeBoscia, 
        :termination_boscia => :terminationBoscia,
        :time_scip => :timeScip, 
        :termination_scip => :terminationScip,
        :time_no_ws => :timeNoWs, 
        :termination_no_ws => :terminationNoWs,
        :time_no_as => :timeNoAs, 
        :termination_no_as => :terminationNoAs,
        :time_no_ss => :timeNoSs, 
        :termination_no_ss => :terminationNoSs,
        :time_afw => :timeAfw, 
        :termination_afw => :terminationAfw,
        )

    # parse to int
    gdf[!,:timeBoscia] = convert.(Int64,round.(gdf[!,:timeBoscia]))
    gdf[!,:timeScip] = convert.(Int64,round.(gdf[!,:timeScip]))
    gdf[!,:timeNoWs] = convert.(Int64,round.(gdf[!,:timeNoWs]))
    gdf[!,:timeNoAs] = convert.(Int64,round.(gdf[!,:timeNoAs]))
    gdf[!,:timeNoSs] = convert.(Int64,round.(gdf[!,:timeNoSs]))
    gdf[!,:timeAfw] = convert.(Int64,round.(gdf[!,:timeAfw]))

    # solved instances in percentage
    no_experiments = 7
    gdf[!,:terminationBoscia] = gdf[!,:terminationBoscia]/no_experiments*100
    gdf[!,:terminationScip] = gdf[!,:terminationScip]/no_experiments*100
    gdf[!,:terminationNoWs] = gdf[!,:terminationNoWs]/no_experiments*100
    gdf[!,:terminationNoAs] = gdf[!,:terminationNoAs]/no_experiments*100
    gdf[!,:terminationNoSs] = gdf[!,:terminationNoSs]/no_experiments*100
    gdf[!,:terminationAfw] = gdf[!,:terminationAfw]/no_experiments*100

    # parse to int
    gdf[!,:terminationBoscia] = convert.(Int64,round.(gdf[!,:terminationBoscia]))
    gdf[!,:terminationScip] = convert.(Int64,round.(gdf[!,:terminationScip]))
    gdf[!,:terminationNoWs] = convert.(Int64,round.(gdf[!,:terminationNoWs]))
    gdf[!,:terminationNoAs] = convert.(Int64,round.(gdf[!,:terminationNoAs]))
    gdf[!,:terminationNoSs] = convert.(Int64,round.(gdf[!,:terminationNoSs]))
    gdf[!,:terminationAfw] = convert.(Int64,round.(gdf[!,:terminationAfw]))

    if mode == "integer"
        file_name = joinpath(@__DIR__, "csv/integer_50.csv")
    elseif mode == "mixed" 
        file_name = joinpath(@__DIR__, "csv/mixed_50.csv")
    end        
    CSV.write(file_name, gdf, append=false)
end
