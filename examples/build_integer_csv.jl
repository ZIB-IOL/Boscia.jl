using CSV
using DataFrames
using Statistics

function build_csv(mode)
    df = DataFrame()

    # load data
    if mode == "integer"
        # load boscia and scip oa
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_vs_scip_integer_50.csv")))
        # indices = [index for index in 1:nrow(df_bs) if isodd(index)]
        # delete!(df_bs, indices)

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
        #delete!(df_afw, indices)
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
        nrow => :NumInstances, renamecols=false
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

    # absolute instances solved
    gdf[!,:terminationBoscia] .= gdf[!,:terminationBoscia]
    gdf[!,:terminationScip] .= gdf[!,:terminationScip]
    gdf[!,:terminationNoWs] .= gdf[!,:terminationNoWs]
    gdf[!,:terminationNoAs] .= gdf[!,:terminationNoAs]
    gdf[!,:terminationNoSs] .= gdf[!,:terminationNoSs]
    gdf[!,:terminationAfw] .= gdf[!,:terminationAfw]

    # relative instances solved
    gdf[!,:terminationBosciaRel] = gdf[!,:terminationBoscia]./gdf[!,:NumInstances]*100
    gdf[!,:terminationScipRel] = gdf[!,:terminationScip]./gdf[!,:NumInstances]*100
    gdf[!,:terminationNoWsRel] = gdf[!,:terminationNoWs]./gdf[!,:NumInstances]*100
    gdf[!,:terminationNoAsRel] = gdf[!,:terminationNoAs]./gdf[!,:NumInstances]*100
    gdf[!,:terminationNoSsRel] = gdf[!,:terminationNoSs]./gdf[!,:NumInstances]*100
    gdf[!,:terminationAfwRel] .= gdf[!,:terminationAfw]./gdf[!,:NumInstances]*100

    # parse to int
    gdf[!,:terminationBosciaRel] = convert.(Int64,round.(gdf[!,:terminationBosciaRel]))
    gdf[!,:terminationScipRel] = convert.(Int64,round.(gdf[!,:terminationScipRel]))
    gdf[!,:terminationNoWsRel] = convert.(Int64,round.(gdf[!,:terminationNoWsRel]))
    gdf[!,:terminationNoAsRel] = convert.(Int64,round.(gdf[!,:terminationNoAsRel]))
    gdf[!,:terminationNoSsRel] = convert.(Int64,round.(gdf[!,:terminationNoSsRel]))
    gdf[!,:terminationAfwRel] = convert.(Int64,round.(gdf[!,:terminationAfwRel]))


    # geo_mean of intersection with solved instances by all solvers
    df_intersection = select!(df, Not(:time_scip))
    df_intersection = select!(df_intersection, Not(:termination_scip))

    df_intersection = filter(row -> !(row.termination_boscia == 0 || row.termination_afw == 0 || row.termination_no_ws == 0 || row.termination_no_ss == 0 || row.termination_no_as == 0),  df_intersection)
    
    df_intersection = combine(
        groupby(df_intersection, :dimension), 
        :time_boscia => geo_mean => :BosciaGeoMeanIntersection,
        :time_no_ws => geo_mean => :NoWsGeoMeanIntersection,
        :time_no_as => geo_mean => :NoAsGeoMeanIntersection,
        :time_no_ss => geo_mean => :NoSsGeoMeanIntersection,
        :time_afw => geo_mean => :AfwGeoMeanIntersection,
        renamecols=false
        )
        
    # parse to int
    df_intersection[!,:BosciaGeoMeanIntersection] = convert.(Int64,round.(df_intersection[!,:BosciaGeoMeanIntersection]))
    df_intersection[!,:NoWsGeoMeanIntersection] = convert.(Int64,round.(df_intersection[!,:NoWsGeoMeanIntersection]))
    df_intersection[!,:NoAsGeoMeanIntersection] = convert.(Int64,round.(df_intersection[!,:NoAsGeoMeanIntersection]))
    df_intersection[!,:NoSsGeoMeanIntersection] = convert.(Int64,round.(df_intersection[!,:NoSsGeoMeanIntersection]))
    df_intersection[!,:AfwGeoMeanIntersection] = convert.(Int64,round.(df_intersection[!,:AfwGeoMeanIntersection]))

    # add geometric mean of intersected instances to main df
    gdf[!,:BosciaGeoMeanIntersection] = df_intersection[!,:BosciaGeoMeanIntersection]
    gdf[!,:NoWsGeoMeanIntersection] = df_intersection[!,:NoWsGeoMeanIntersection]
    gdf[!,:NoAsGeoMeanIntersection] = df_intersection[!,:NoAsGeoMeanIntersection]
    gdf[!,:NoSsGeoMeanIntersection] = df_intersection[!,:NoSsGeoMeanIntersection]
    gdf[!,:AfwGeoMeanIntersection] = df_intersection[!,:AfwGeoMeanIntersection] 
    
    # save csv
    if mode == "integer"
        file_name = joinpath(@__DIR__, "csv/integer_50.csv")
    elseif mode == "mixed" 
        file_name = joinpath(@__DIR__, "csv/mixed_50.csv")
    end        
    CSV.write(file_name, gdf, append=false)
end
