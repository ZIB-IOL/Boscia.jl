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
        # load boscia and scip oa
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_vs_scip_mixed_50.csv")))
        # indices = [index for index in 1:nrow(df_bs) if isodd(index)]
        # delete!(df_bs, indices)
        filter!(row -> !(row.seed == 6 && row.dimension == 70),  df_bs)
        filter!(row -> !(row.seed == 6 && row.dimension == 80),  df_bs)
        filter!(row -> !(row.seed == 4 && row.dimension == 100),  df_bs)
        filter!(row -> !(row.seed == 9 && row.dimension == 100),  df_bs)
        filter!(row -> !(row.dimension > 100),  df_bs)

        time_scip = [row == -Inf ? 1800.0 : row for row in df_bs[!,:time_scip]]
        df_bs.termination_boscia .= replace.(df_bs.termination_boscia, "Optimal (tree empty)" => "OPTIMAL")
        df_bs.termination_boscia .= replace.(df_bs.termination_boscia, "Time limit reached" => "TIME_LIMIT")
        termination_boscia = [row == "OPTIMAL" ? 1 : 0 for row in df_bs[!,:termination_boscia]]
        termination_scip = [row == "OPTIMAL" ? 1 : 0 for row in df_bs[!,:termination_scip]]

        df[!,:dimension] = df_bs[!,:dimension]
        df[!,:time_boscia] = df_bs[!,:time_boscia]
        df[!,:termination_boscia] = termination_boscia
        df[!,:time_scip] = time_scip #df_bs[!,:time_scip]
        df[!,:termination_scip] = termination_scip

        # load afw
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_mixed_50.csv")))
        #delete!(df_afw, indices)
        filter!(row -> !(row.seed == 6 && row.dimension == 70),  df_afw)
        filter!(row -> !(row.seed == 6 && row.dimension == 80),  df_afw)
        filter!(row -> !(row.seed == 4 && row.dimension == 100),  df_afw)

        df_afw.termination_afw .= replace.(df_afw.termination_afw, "Optimal (tree empty)" => "OPTIMAL")
        df_afw.termination_afw .= replace.(df_afw.termination_afw, "Time limit reached" => "TIME_LIMIT")
        termination_afw = [row == "OPTIMAL" ? 1 : 0 for row in df_afw[!,:termination_afw]]

        df[!,:time_afw] = df_afw[!,:time_afw]
        df[!,:termination_afw] = termination_afw

        # load without as, without ss
        df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mixed_50.csv")))
        
        filter!(row -> !(row.seed == 6 && row.dimension == 70),  df_no_ws)
        filter!(row -> !(row.seed == 6 && row.dimension == 80),  df_no_ws)
        filter!(row -> !(row.seed == 4 && row.dimension == 100),  df_no_ws)
        filter!(row -> !(row.seed == 9 && row.dimension == 100),  df_no_ws)
        
        df_no_ws.termination_afw .= replace.(df_no_ws.termination_afw, "Optimal (tree empty)" => "OPTIMAL")
        df_no_ws.termination_afw .= replace.(df_no_ws.termination_afw, "Time limit reached" => "TIME_LIMIT")
        termination_no_ws = [row == "OPTIMAL" ? 1 : 0 for row in df_no_ws[!,:termination_afw]]

        df[!,:time_no_ws] = df_no_ws[!,:time_afw]
        df[!,:termination_no_ws] = termination_no_ws

        # load without ss
        df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_ss_mixed_50.csv")))
        
        filter!(row -> !(row.seed == 6 && row.dimension == 70),  df_no_ss)
        filter!(row -> !(row.seed == 6 && row.dimension == 80),  df_no_ss)
        filter!(row -> !(row.seed == 4 && row.dimension == 100),  df_no_ss)
        filter!(row -> !(row.seed == 9 && row.dimension == 100),  df_no_ss)

        df_no_ss.termination_afw .= replace.(df_no_ss.termination_afw, "Optimal (tree empty)" => "OPTIMAL")
        df_no_ss.termination_afw .= replace.(df_no_ss.termination_afw, "Time limit reached" => "TIME_LIMIT")
        termination_no_ss = [row == "OPTIMAL" ? 1 : 0 for row in df_no_ss[!,:termination_afw]]

        df[!,:time_no_ss] = df_no_ss[!,:time_afw]
        df[!,:termination_no_ss] = termination_no_ss

        # load without as
        df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_mixed_50.csv")))
        
        filter!(row -> !(row.seed == 6 && row.dimension == 70),  df_no_as)
        filter!(row -> !(row.seed == 6 && row.dimension == 80),  df_no_as)
        filter!(row -> !(row.seed == 4 && row.dimension == 100),  df_no_as)
        filter!(row -> !(row.seed == 9 && row.dimension == 100),  df_no_as)

        df_no_as.termination_afw .= replace.(df_no_as.termination_afw, "Optimal (tree empty)" => "OPTIMAL")
        df_no_as.termination_afw .= replace.(df_no_as.termination_afw, "Time limit reached" => "TIME_LIMIT")
        termination_no_as = [row == "OPTIMAL" ? 1 : 0 for row in df_no_as[!,:termination_afw]]

        df[!,:time_no_as] = df_no_as[!,:time_afw]
        df[!,:termination_no_as] = termination_no_as
    
    elseif mode == "poisson"
        # load boscia 
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_poisson.csv")))
        # filter!(row -> !(row.seed == 7 && row.Ns == 10.0 && row.dimension == 70),  df_bs)

        df_bs.termination .= replace.(df_bs.termination, "Time limit reached" => "TIME_LIMIT")
        termination_boscia = [row == "OPTIMAL" ? 1 : 0 for row in df_bs[!,:termination]]

        df[!,:dimension] = df_bs[!,:dimension]
        df[!,:time_boscia] = df_bs[!,:time]
        df[!,:seed] = df_bs[!,:seed]
        df[!,:p] = df_bs[!,:p]
        df[!,:k] = df_bs[!,:k]
        df[!,:Ns] = df_bs[!,:Ns]

        df[!,:termination_boscia] = termination_boscia

        # load afw
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_poisson.csv")))
        df_afw.termination .= replace.(df_afw.termination, "Time limit reached" => "TIME_LIMIT")
        termination_afw = [row == "OPTIMAL" ? 1 : 0 for row in df_afw[!,:termination]]

        df_afw[!,:time_afw] = df_afw[!,:time]
        df_afw[!,:termination_afw] = termination_afw
        df_afw = select(df_afw, [:termination_afw, :time_afw, :seed, :dimension, :k, :Ns, :p])

        df = innerjoin(df, df_afw, on = [:seed, :dimension, :k, :Ns, :p])

        # load without as, without ss
        df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_poisson.csv")))
        df_no_ws.termination .= replace.(df_no_ws.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ws = [row == "OPTIMAL" ? 1 : 0 for row in df_no_ws[!,:termination]]

        df_no_ws[!,:time_no_ws] = df_no_ws[!,:time]
        df_no_ws[!,:termination_no_ws] = termination_no_ws
        df_no_ws = select(df_no_ws, [:termination_no_ws, :time_no_ws, :seed, :dimension, :k, :Ns, :p])

        df = innerjoin(df, df_no_ws, on = [:seed, :dimension, :k, :Ns, :p])
        # print(first(df,5))

        # load without as
        df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_poisson.csv")))
        df_no_as.termination .= replace.(df_no_as.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_as = [row == "OPTIMAL" ? 1 : 0 for row in df_no_as[!,:termination]]

        df_no_as[!,:time_no_as] = df_no_as[!,:time]
        df_no_as[!,:termination_no_as] = termination_no_as
        df_no_as = select(df_no_as, [:termination_no_as, :time_no_as, :seed, :dimension, :k, :Ns, :p])

        df = innerjoin(df, df_no_as, on = [:seed, :dimension, :k, :Ns, :p])

        # load without ss
        df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_ss_poisson.csv")))
        df_no_ss.termination .= replace.(df_no_ss.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ss = [row == "OPTIMAL" ? 1 : 0 for row in df_no_ss[!,:termination]]

        df_no_ss[!,:time_no_ss] = df_no_ss[!,:time]
        df_no_ss[!,:termination_no_ss] = termination_no_ss
        df_no_ss = select(df_no_ss, [:termination_no_ss, :time_no_ss, :seed, :dimension, :k, :Ns, :p])

        df = innerjoin(df, df_no_ss, on = [:seed, :dimension, :k, :Ns, :p])

        # load scip oa
        df_scip = DataFrame(CSV.File(joinpath(@__DIR__, "csv/scip_oa_poisson.csv")))
        # add killed instance 
        push!(df_scip,[7, 70, 70, 35.0, 10.0, 1800, Inf, "KILLED", 0])
        push!(df_scip,[1, 50, 50, 25.0, 5.0, 1800, Inf, "KILLED", 0])
        push!(df_scip,[2, 50, 50, 25.0, 5.0, 1800, Inf, "KILLED", 0])
        push!(df_scip,[3, 50, 50, 25.0, 5.0, 1800, Inf, "KILLED", 0])

        termination_scip = [row == "OPTIMAL" ? 1 : 0 for row in df_scip[!,:termination]]

        time_scip = []
        for row in eachrow(df_scip)
            if row.solution == Inf 
                append!(time_scip,1800) 
            else 
                append!(time_scip,row.time)
            end
        end

        df_scip[!,:time_scip] = time_scip
        df_scip[!,:termination_scip] = termination_scip
        df_scip[!,:solution_scip] = df_scip[!,:solution]
        df_scip = select(df_scip, [:termination_scip, :time_scip, :seed, :dimension, :k, :Ns, :p])

        # delete duplicates
        df_scip = unique(df_scip, [:dimension, :k, :Ns, :seed])

        # sort!(df, [:dimension, :k, :Ns, :p])
        # print(first(df,20))
        # sort!(df_scip, [:dimension, :k, :Ns, :p])
        # print(first(df_scip,20))
        df = innerjoin(df, df_scip, on = [:seed, :dimension, :k, :Ns, :p])
        # print(sort(df, [:dimension, :k, :Ns, :p]))
        # df_sol = df[!, [:time_scip, :termination_scip, :solution_scip, :time_boscia, :termination_boscia, :solution_boscia]]
        # print(filter(row -> (row.termination_scip == 1 && row.termination_boscia == 1),  df_sol))
        # df_temp = combine(
        #     groupby(df, [:dimension, :k, :Ns]), 
        #     nrow => :NumInstances, renamecols=false
        #     )
        # sort!(df, [:dimension, :k, :Ns])
        # print(df)
        # print(df_temp)

        # save csv 
        file_name = joinpath(@__DIR__, "csv/poisson_non_grouped.csv")
        CSV.write(file_name, df, append=false)

    elseif mode == "sparse_reg"
        # load boscia 
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_sparse_reg.csv")))
        # delete duplicates
        df_bs = unique(df_bs, [:dimension, :k, :p, :seed])

        df_bs.termination .= replace.(df_bs.termination, "Time limit reached" => "TIME_LIMIT")
        termination_boscia = [row == "OPTIMAL" ? 1 : 0 for row in df_bs[!,:termination]]

        df[!,:dimension] = df_bs[!,:dimension]
        df[!,:time_boscia] = df_bs[!,:time]
        df[!,:seed] = df_bs[!,:seed]
        df[!,:p] = df_bs[!,:p]
        df[!,:k] = df_bs[!,:k]

        df[!,:termination_boscia] = termination_boscia
        df[!, :solution_boscia] = df_bs[!, :solution]

        # load afw
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_sparse_reg.csv")))
        df_afw.termination .= replace.(df_afw.termination, "Time limit reached" => "TIME_LIMIT")
        termination_afw = [row == "OPTIMAL" ? 1 : 0 for row in df_afw[!,:termination]]

        df_afw[!,:time_afw] = df_afw[!,:time]
        df_afw[!,:termination_afw] = termination_afw
        df_afw = select(df_afw, [:termination_afw, :time_afw, :seed, :dimension, :k, :p])

        df = innerjoin(df, df_afw, on = [:seed, :dimension, :k, :p])

        # load without as, without ss
        df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_sparse_reg.csv")))
        df_no_ws.termination .= replace.(df_no_ws.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ws = [row == "OPTIMAL" ? 1 : 0 for row in df_no_ws[!,:termination]]

        df_no_ws[!,:time_no_ws] = df_no_ws[!,:time]
        df_no_ws[!,:termination_no_ws] = termination_no_ws
        df_no_ws = select(df_no_ws, [:termination_no_ws, :time_no_ws, :seed, :dimension, :k, :p])

        df = innerjoin(df, df_no_ws, on = [:seed, :dimension, :k, :p])
        # print(first(df,5))

        # load without as
        df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_sparse_reg.csv")))
        df_no_as.termination .= replace.(df_no_as.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_as = [row == "OPTIMAL" ? 1 : 0 for row in df_no_as[!,:termination]]

        df_no_as[!,:time_no_as] = df_no_as[!,:time]
        df_no_as[!,:termination_no_as] = termination_no_as
        df_no_as = select(df_no_as, [:termination_no_as, :time_no_as, :seed, :dimension, :k, :p])

        df = innerjoin(df, df_no_as, on = [:seed, :dimension, :k, :p])

        # load without ss
        df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_ss_sparse_reg.csv")))
        df_no_ss.termination .= replace.(df_no_ss.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ss = [row == "OPTIMAL" ? 1 : 0 for row in df_no_ss[!,:termination]]

        df_no_ss[!,:time_no_ss] = df_no_ss[!,:time]
        df_no_ss[!,:termination_no_ss] = termination_no_ss
        df_no_ss = select(df_no_ss, [:termination_no_ss, :time_no_ss, :seed, :dimension, :k, :p])

        df = innerjoin(df, df_no_ss, on = [:seed, :dimension, :k, :p])

        # load scip oa
        df_scip = DataFrame(CSV.File(joinpath(@__DIR__, "csv/scip_oa_sparse_reg.csv")))
        termination_scip = [row == "OPTIMAL" ? 1 : 0 for row in df_scip[!,:termination]]

        time_scip = []
        for row in eachrow(df_scip)
            if row.solution == Inf 
                append!(time_scip,1800) 
            else 
                append!(time_scip,row.time)
            end
        end

        df_scip[!,:time_scip] = time_scip
        df_scip[!,:termination_scip] = termination_scip
        df_scip[!,:solution_scip] = df_scip[!,:solution]
        df_scip = select(df_scip, [:solution_scip, :termination_scip, :time_scip, :seed, :dimension, :k, :p])

        # delete duplicates
        df_scip = unique(df_scip, [:dimension, :k, :seed])

        # print(first(df,20))
        # sort!(df_scip, [:dimension, :k, :Ns, :p])
        # print(first(df_scip,20))
        df = innerjoin(df, df_scip, on = [:seed, :dimension, :k, :p])
        print(sort(df, [:dimension, :p, :k]))
        df_sol = df[!, [:time_scip, :termination_scip, :solution_scip, :time_boscia, :termination_boscia, :solution_boscia]]
        print(filter(row -> (row.termination_scip == 1 && row.termination_boscia == 1),  df_sol))
        sort!(df, [:dimension, :p, :k])

        # save csv 
        file_name = joinpath(@__DIR__, "csv/sparse_reg_non_grouped.csv")
        CSV.write(file_name, df, append=false)
    
    elseif mode == "sparse_log_reg"  
        # load boscia 
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_sparse_log_regression.csv")))
        # delete duplicates
        df_bs = unique(df_bs, [:dimension, :k, :p, :seed, :M])

        df_bs.termination .= replace.(df_bs.termination, "Time limit reached" => "TIME_LIMIT")
        termination_boscia = [row == "OPTIMAL" ? 1 : 0 for row in df_bs[!,:termination]]

        df[!,:dimension] = df_bs[!,:dimension]
        df[!,:time_boscia] = df_bs[!,:time]
        df[!,:seed] = df_bs[!,:seed]
        df[!,:p] = df_bs[!,:p]
        df[!,:k] = df_bs[!,:k]
        df[!,:M] = df_bs[!,:M]

        df[!,:termination_boscia] = termination_boscia
        df[!, :solution_boscia] = df_bs[!, :solution]  

        # load scip oa
        df_scip = DataFrame(CSV.File(joinpath(@__DIR__, "csv/scip_oa_sparse_log_regression.csv")))
        termination_scip = [row == "OPTIMAL" ? 1 : 0 for row in df_scip[!,:termination]]

        df_scip[!,:time_scip] = df_scip[!,:time]
        df_scip[!,:termination_scip] = termination_scip
        df_scip[!,:solution_scip] = df_scip[!,:solution]
        df_scip = select(df_scip, [:solution_scip, :termination_scip, :time_scip, :seed, :dimension, :k, :p, :M])

        # delete duplicates
        df_scip = unique(df_scip, [:dimension, :p, :k, :seed, :M])

        # print(first(df,20))
        # sort!(df_scip, [:dimension, :k, :Ns, :p])
        # print(first(df_scip,20))
        df = innerjoin(df, df_scip, on = [:seed, :dimension, :k, :p, :M])
        # print(sort(df, [:dimension, :p, :k]))
        df_sol = df[!, [:time_scip, :termination_scip, :solution_scip, :time_boscia, :termination_boscia, :solution_boscia]]
        print(filter(row -> (row.termination_scip == 1 && row.termination_boscia == 1),  df_sol))
        # sort!(df, [:dimension, :p, :k])

        # save csv 
        file_name = joinpath(@__DIR__, "csv/sparse_log_reg_non_grouped.csv")
        CSV.write(file_name, df, append=false)

    elseif mode == "tailed_cardinality"
        # load boscia 
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_tailed_cardinality.csv")))
        # filter!(row -> !(row.seed == 7 && row.Ns == 10.0 && row.dimension == 70),  df_bs)

        df_bs.termination .= replace.(df_bs.termination, "Time limit reached" => "TIME_LIMIT")
        termination_boscia = df_bs[!,:termination] #[row == "OPTIMAL" ? 1 : 0 for row in df_bs[!,:termination]]

        df[!,:time_boscia] = df_bs[!,:time]
        df[!,:seed] = df_bs[!,:seed]
        df[!,:n0] = df_bs[!,:n0]
        df[!,:m0] = df_bs[!,:m0]
        df[!,:M] = df_bs[!,:M]
        df[!,:solution_boscia] = df_bs[!,:solution]

        df[!,:termination_boscia] = termination_boscia

        # load scip oa
        df_scip = DataFrame(CSV.File(joinpath(@__DIR__, "csv/scip_oa_tailed_cardinality.csv")))

        termination_scip = df_scip[!,:termination] #[row == "OPTIMAL" ? 1 : 0 for row in df_scip[!,:termination]]

        time_scip = []
        for row in eachrow(df_scip)
            if row.solution == Inf 
                append!(time_scip,1800) 
            else 
                append!(time_scip,row.time)
            end
        end

        df_scip[!,:time_scip] = time_scip
        df_scip[!,:termination_scip] = termination_scip
        df_scip[!,:solution_scip] = df_scip[!,:solution]
        df_scip = select(df_scip, [:termination_scip, :solution_scip, :time_scip, :seed, :n0, :m0, :M])

        # sort!(df, [:dimension, :k, :Ns, :p])
        # print(first(df,20))
        # sort!(df_scip, [:dimension, :k, :Ns, :p])
        # print(first(df_scip,20))
        df = innerjoin(df, df_scip, on = [:seed, :n0, :m0, :M])

        # save csv 
        file_name = joinpath(@__DIR__, "csv/tailed_cardinality_non_grouped.csv")
        CSV.write(file_name, df, append=false)
    end

    function geo_mean(group)
        prod = 1.0
        n = length(group)
        for element in group
            # @show element
            prod = prod * element
        end
        # @show prod, n
        return prod^(1/n)
    end

    # group by dimension
    if mode != "poisson" && mode != "sparse_reg"
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
    elseif mode == "poisson"
        gdf = combine(
            groupby(df, [:dimension, :k, :Ns]), 
            :time_boscia => geo_mean, :termination_boscia => sum,
            :time_scip => geo_mean, :termination_scip => sum,
            :time_no_ws => geo_mean, :termination_no_ws => sum,
            :time_no_as => geo_mean, :termination_no_as => sum,
            :time_no_ss => geo_mean, :termination_no_ss => sum,
            :time_afw => geo_mean, :termination_afw => sum,
            nrow => :NumInstances, renamecols=false
            )
    elseif mode == "sparse_reg"
        gdf = combine(
            groupby(df, [:dimension, :p, :k]), 
            :time_boscia => geo_mean, :termination_boscia => sum,
            :time_scip => geo_mean, :termination_scip => sum,
            :time_no_ws => geo_mean, :termination_no_ws => sum,
            :time_no_as => geo_mean, :termination_no_as => sum,
            :time_no_ss => geo_mean, :termination_no_ss => sum,
            :time_afw => geo_mean, :termination_afw => sum,
            nrow => :NumInstances, renamecols=false
            )
    end

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

    # deletes entire row if scip solves solution but boscia does not
    df_intersection = filter(row -> !(row.termination_boscia == 0 || row.termination_afw == 0 || row.termination_no_ws == 0 || row.termination_no_ss == 0 || row.termination_no_as == 0),  df_intersection)
    
    if mode != "poisson" && mode != "sparse_reg"
        df_intersection = combine(
            groupby(df_intersection, :dimension), 
            :time_boscia => geo_mean => :BosciaGeoMeanIntersection,
            :time_no_ws => geo_mean => :NoWsGeoMeanIntersection,
            :time_no_as => geo_mean => :NoAsGeoMeanIntersection,
            :time_no_ss => geo_mean => :NoSsGeoMeanIntersection,
            :time_afw => geo_mean => :AfwGeoMeanIntersection,
            renamecols=false
            )
    elseif mode == "poisson"
        df_intersection = combine(
            groupby(df_intersection, [:dimension, :k, :Ns]), 
            :time_boscia => geo_mean => :BosciaGeoMeanIntersection,
            :time_no_ws => geo_mean => :NoWsGeoMeanIntersection,
            :time_no_as => geo_mean => :NoAsGeoMeanIntersection,
            :time_no_ss => geo_mean => :NoSsGeoMeanIntersection,
            :time_afw => geo_mean => :AfwGeoMeanIntersection,
            renamecols=false
            )
    elseif mode == "sparse_reg"
        df_intersection = combine(
            groupby(df_intersection, [:dimension, :p, :k]), 
            :time_boscia => geo_mean => :BosciaGeoMeanIntersection,
            :time_no_ws => geo_mean => :NoWsGeoMeanIntersection,
            :time_no_as => geo_mean => :NoAsGeoMeanIntersection,
            :time_no_ss => geo_mean => :NoSsGeoMeanIntersection,
            :time_afw => geo_mean => :AfwGeoMeanIntersection,
            renamecols=false
            )
    end
        
    # parse to int
    df_intersection[!,:BosciaGeoMeanIntersection] = convert.(Int64,round.(df_intersection[!,:BosciaGeoMeanIntersection]))
    df_intersection[!,:NoWsGeoMeanIntersection] = convert.(Int64,round.(df_intersection[!,:NoWsGeoMeanIntersection]))
    df_intersection[!,:NoAsGeoMeanIntersection] = convert.(Int64,round.(df_intersection[!,:NoAsGeoMeanIntersection]))
    df_intersection[!,:NoSsGeoMeanIntersection] = convert.(Int64,round.(df_intersection[!,:NoSsGeoMeanIntersection]))
    df_intersection[!,:AfwGeoMeanIntersection] = convert.(Int64,round.(df_intersection[!,:AfwGeoMeanIntersection]))

    if mode != "poisson" && mode != "sparse_reg"
        gdf = innerjoin(gdf, df_intersection, on =[:dimension])
    elseif mode == "poisson"
        gdf = innerjoin(gdf, df_intersection, on =[:dimension, :k, :Ns])
    elseif mode == "sparse_reg"
        gdf = innerjoin(gdf, df_intersection, on =[:dimension, :p, :k])
    end
    # add geometric mean of intersected instances to main df
    # gdf[!,:BosciaGeoMeanIntersection] = df_intersection[!,:BosciaGeoMeanIntersection]
    # gdf[!,:NoWsGeoMeanIntersection] = df_intersection[!,:NoWsGeoMeanIntersection]
    # gdf[!,:NoAsGeoMeanIntersection] = df_intersection[!,:NoAsGeoMeanIntersection]
    # gdf[!,:NoSsGeoMeanIntersection] = df_intersection[!,:NoSsGeoMeanIntersection]
    # gdf[!,:AfwGeoMeanIntersection] = df_intersection[!,:AfwGeoMeanIntersection] 
    
    # save csv
    if mode == "integer"
        file_name = joinpath(@__DIR__, "csv/integer_50.csv")
    elseif mode == "mixed" 
        file_name = joinpath(@__DIR__, "csv/mixed_50.csv")
    elseif mode == "poisson" 
        file_name = joinpath(@__DIR__, "csv/poisson.csv")
    elseif mode == "sparse_reg" 
        file_name = joinpath(@__DIR__, "csv/sparse_reg.csv")   
    end        
    CSV.write(file_name, gdf, append=false)
end
