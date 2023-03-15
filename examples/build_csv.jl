using CSV
using DataFrames
using Statistics

function build_non_grouped_csv(mode)
    df = DataFrame()

    # load data
    if mode == "integer"
        # load boscia 
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_integer_portfolio.csv")))

        df_bs.termination .= replace.(df_bs.termination, "Time limit reached" => "TIME_LIMIT")
        for row in eachrow(df_bs)
            if row.time > 1800
                row.termination = "TIME_LIMIT" 
            end
        end

        termination_boscia = [row == "OPTIMAL" || row == "tree.lb>primal-dual_gap" || row == "primal>=tree.incumbent" ? 1 : 0 for row in df_bs[!,:termination]]

        df[!,:dimension] = df_bs[!,:dimension]
        df[!,:time_boscia] = df_bs[!,:time]
        df[!,:seed] = df_bs[!,:seed]
        df[!,:solution_boscia] = df_bs[!,:solution]
        df[!,:termination_boscia] = termination_boscia
        lowerBounds = df_bs[!, :solution] - df_bs[!, :dual_gap]
        df[!,:lb_boscia] = lowerBounds
        # incumbent - lb /min (incumbent ,lb )
        df[!, :rel_gap_boscia] = df_bs[!, :dual_gap] ./ min.(abs.(lowerBounds), abs.(df_bs[!, :solution]))

        # load afw
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_integer_portfolio.csv")))
        df_afw.termination .= replace.(df_afw.termination, "Time limit reached" => "TIME_LIMIT")
        for row in eachrow(df_afw)
            if row.time > 1800
                row.termination = "TIME_LIMIT" 
            end
        end

        termination_afw = [row == "OPTIMAL" || row == "tree.lb>primal-dual_gap" || row == "primal>=tree.incumbent" ? 1 : 0 for row in df_afw[!,:termination]]

        df_afw[!,:time_afw] = df_afw[!,:time]
        df_afw[!,:termination_afw] = termination_afw
        lowerBounds = df_afw[!, :solution] - df_afw[!, :dual_gap]
        df_afw[!, :rel_gap_afw] = (df_afw[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_afw[!, :solution]))
        df_afw = select(df_afw, [:termination_afw, :time_afw, :rel_gap_afw, :seed, :dimension])

        df = outerjoin(df, df_afw, on = [:seed, :dimension])

        # load without as, without ss
        df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_integer_portfolio.csv")))
        df_no_ws.termination .= replace.(df_no_ws.termination, "Time limit reached" => "TIME_LIMIT")
        for row in eachrow(df_no_ws)
            if row.time > 1800
                row.termination = "TIME_LIMIT" 
            end
        end

        termination_no_ws = [row == "OPTIMAL" || row == "tree.lb>primal-dual_gap" || row == "primal>=tree.incumbent" ? 1 : 0 for row in df_no_ws[!,:termination]]

        df_no_ws[!,:time_no_ws] = df_no_ws[!,:time]
        df_no_ws[!,:termination_no_ws] = termination_no_ws
        lowerBounds = df_no_ws[!, :solution] - df_no_ws[!, :dual_gap]
        df_no_ws[!, :rel_gap_no_ws] = df_no_ws[!, :dual_gap] ./ min.(abs.(lowerBounds), abs.(df_no_ws[!, :solution]))
        df_no_ws = select(df_no_ws, [:termination_no_ws, :time_no_ws, :rel_gap_no_ws, :seed, :dimension])

        df = innerjoin(df, df_no_ws, on = [:seed, :dimension])

        # load ipopt 
        df_ipopt = DataFrame(CSV.File(joinpath(@__DIR__, "csv/ipopt_portfolio_integer.csv")))
        df_ipopt.termination .= replace.(df_ipopt.termination, "Time limit reached" => "TIME_LIMIT")
        termination_ipopt = [row == "Optimal" ? 1 : 0 for row in df_ipopt[!, :termination]]

        df_ipopt[!, :time_ipopt] = df_ipopt[!, :time]/1000
        df_ipopt[!, :termination_ipopt] = termination_ipopt
        df_ipopt[!,:solution_ipopt] = df_ipopt[!,:solution]
        df_ipopt = select(df_ipopt, [:termination_ipopt, :time_ipopt, :solution_ipopt, :seed, :dimension])

        df = innerjoin(df, df_ipopt, on = [:seed, :dimension])   

        # load without as
        df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_integer_portfolio.csv")))
        df_no_as.termination .= replace.(df_no_as.termination, "Time limit reached" => "TIME_LIMIT")
        for row in eachrow(df_no_as)
            if row.time > 1800
                row.termination = "TIME_LIMIT" 
            end
        end
        termination_no_as = [row == "OPTIMAL" || row == "tree.lb>primal-dual_gap" || row == "primal>=tree.incumbent" ? 1 : 0 for row in df_no_as[!,:termination]]

        df_no_as[!,:time_no_as] = df_no_as[!,:time]
        df_no_as[!,:termination_no_as] = termination_no_as
        lowerBounds = df_no_as[!, :solution] - df_no_as[!, :dual_gap]
        df_no_as[!, :rel_gap_no_as] = (df_no_as[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_as[!, :solution]))
        df_no_as = select(df_no_as, [:termination_no_as, :time_no_as, :rel_gap_no_as, :seed, :dimension])

        df = innerjoin(df, df_no_as, on = [:seed, :dimension])

        # load without ss
        df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_ss_integer_portfolio.csv")))
        df_no_ss.termination .= replace.(df_no_ss.termination, "Time limit reached" => "TIME_LIMIT")
        for row in eachrow(df_no_ss)
            if row.time > 1800
                row.termination = "TIME_LIMIT" 
            end
        end
        termination_no_ss = [row == "OPTIMAL" || row == "tree.lb>primal-dual_gap" || row == "primal>=tree.incumbent" ? 1 : 0 for row in df_no_ss[!,:termination]]

        df_no_ss[!,:time_no_ss] = df_no_ss[!,:time]
        df_no_ss[!,:termination_no_ss] = termination_no_ss
        lowerBounds = df_no_ss[!, :solution] - df_no_ss[!, :dual_gap]
        df_no_ss[!, :rel_gap_no_ss] = (df_no_ss[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_ss[!, :solution]))
        df_no_ss = select(df_no_ss, [:termination_no_ss, :time_no_ss, :rel_gap_no_ss, :seed, :dimension])

        df = innerjoin(df, df_no_ss, on = [:seed, :dimension])

        # load scip oa
        df_scip = DataFrame(CSV.File(joinpath(@__DIR__, "csv/scip_oa_portfolio_integer.csv"))) 
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
        df_scip = select(df_scip, [:termination_scip, :time_scip, :seed, :solution_scip, :dimension])

        df = innerjoin(df, df_scip, on = [:seed, :dimension])

        sort!(df, [:dimension])

        # check if solution optimal
        optimal_scip = []
        optimal_ipopt = []
        optimal_boscia = []
        for row in eachrow(df)
            if isapprox(row.lb_boscia, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_boscia, 1)
            else 
                append!(optimal_boscia, 0)
            end
            if isapprox(row.solution_ipopt, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_ipopt, 1)
            else 
                append!(optimal_ipopt, 0)
            end
            if isapprox(row.solution_scip, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_scip, 1)
            else 
                append!(optimal_scip, 0)
            end
        end
        df[!,:optimal_scip] = optimal_scip
        df[!,:optimal_ipopt] = optimal_ipopt
        df[!,:optimal_boscia] = optimal_boscia

        # compute relative gap
        rel_gap_scip = []
        rel_gap_ipopt = []
        for row in eachrow(df)
            if min(abs(row.solution_scip), abs(row.lb_boscia)) == 0
                push!(rel_gap_scip, row.solution_scip - row.lb_boscia)
            elseif sign(row.lb_boscia) != sign(row.solution_scip)
                push!(rel_gap_scip, Inf)
            else
                push!(rel_gap_scip, (row.solution_scip - row.lb_boscia)/min(abs(row.solution_scip), abs(row.lb_boscia)))
            end
            if min(abs(row.solution_ipopt), abs(row.lb_boscia)) == 0
                push!(rel_gap_ipopt, row.solution_ipopt - row.lb_boscia)
            elseif sign(row.lb_boscia) != sign(row.solution_ipopt)
                push!(rel_gap_ipopt, Inf)
            else
                push!(rel_gap_ipopt, (row.solution_ipopt - row.lb_boscia)/min(abs(row.solution_ipopt), abs(row.lb_boscia)))
            end
        end
        df[!, :rel_gap_scip] = round.(rel_gap_scip,digits=3)
        df[!, :rel_gap_ipopt] = round.(rel_gap_ipopt,digits=3)

        # save csv 
        file_name = joinpath(@__DIR__, "csv/portfolio_integer_non_grouped.csv")
        CSV.write(file_name, df, append=false)
    
    # elseif mode == "mixed_obsolete"
        #     # load boscia and scip oa
        #     df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_vs_scip_mixed_50.csv")))
        #     # indices = [index for index in 1:nrow(df_bs) if isodd(index)]
        #     # delete!(df_bs, indices)
        #     filter!(row -> !(row.seed == 6 && row.dimension == 70),  df_bs)
        #     filter!(row -> !(row.seed == 6 && row.dimension == 80),  df_bs)
        #     filter!(row -> !(row.seed == 4 && row.dimension == 100),  df_bs)
        #     filter!(row -> !(row.seed == 9 && row.dimension == 100),  df_bs)
        #     filter!(row -> !(row.dimension > 100),  df_bs)

        #     time_scip = [row == -Inf ? 1800.0 : row for row in df_bs[!,:time_scip]]
        #     df_bs.termination_boscia .= replace.(df_bs.termination_boscia, "Optimal (tree empty)" => "OPTIMAL")
        #     df_bs.termination_boscia .= replace.(df_bs.termination_boscia, "Time limit reached" => "TIME_LIMIT")
        #     termination_boscia = [row == "OPTIMAL" ? 1 : 0 for row in df_bs[!,:termination_boscia]]
        #     termination_scip = [row == "OPTIMAL" ? 1 : 0 for row in df_bs[!,:termination_scip]]

        #     df[!,:dimension] = df_bs[!,:dimension]
        #     df[!,:time_boscia] = df_bs[!,:time_boscia]
        #     df[!,:termination_boscia] = termination_boscia
        #     df[!,:time_scip] = time_scip #df_bs[!,:time_scip]
        #     df[!,:termination_scip] = termination_scip

        #     # load afw
        #     df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_mixed_50.csv")))
        #     #delete!(df_afw, indices)
        #     filter!(row -> !(row.seed == 6 && row.dimension == 70),  df_afw)
        #     filter!(row -> !(row.seed == 6 && row.dimension == 80),  df_afw)
        #     filter!(row -> !(row.seed == 4 && row.dimension == 100),  df_afw)

        #     df_afw.termination_afw .= replace.(df_afw.termination_afw, "Optimal (tree empty)" => "OPTIMAL")
        #     df_afw.termination_afw .= replace.(df_afw.termination_afw, "Time limit reached" => "TIME_LIMIT")
        #     termination_afw = [row == "OPTIMAL" ? 1 : 0 for row in df_afw[!,:termination_afw]]

        #     df[!,:time_afw] = df_afw[!,:time_afw]
        #     df[!,:termination_afw] = termination_afw

        #     # load without as, without ss
        #     df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mixed_50.csv")))
            
        #     filter!(row -> !(row.seed == 6 && row.dimension == 70),  df_no_ws)
        #     filter!(row -> !(row.seed == 6 && row.dimension == 80),  df_no_ws)
        #     filter!(row -> !(row.seed == 4 && row.dimension == 100),  df_no_ws)
        #     filter!(row -> !(row.seed == 9 && row.dimension == 100),  df_no_ws)
            
        #     df_no_ws.termination_afw .= replace.(df_no_ws.termination_afw, "Optimal (tree empty)" => "OPTIMAL")
        #     df_no_ws.termination_afw .= replace.(df_no_ws.termination_afw, "Time limit reached" => "TIME_LIMIT")
        #     termination_no_ws = [row == "OPTIMAL" ? 1 : 0 for row in df_no_ws[!,:termination_afw]]

        #     df[!,:time_no_ws] = df_no_ws[!,:time_afw]
        #     df[!,:termination_no_ws] = termination_no_ws

        #     # load without ss
        #     df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_ss_mixed_50.csv")))
            
        #     filter!(row -> !(row.seed == 6 && row.dimension == 70),  df_no_ss)
        #     filter!(row -> !(row.seed == 6 && row.dimension == 80),  df_no_ss)
        #     filter!(row -> !(row.seed == 4 && row.dimension == 100),  df_no_ss)
        #     filter!(row -> !(row.seed == 9 && row.dimension == 100),  df_no_ss)

        #     df_no_ss.termination_afw .= replace.(df_no_ss.termination_afw, "Optimal (tree empty)" => "OPTIMAL")
        #     df_no_ss.termination_afw .= replace.(df_no_ss.termination_afw, "Time limit reached" => "TIME_LIMIT")
        #     termination_no_ss = [row == "OPTIMAL" ? 1 : 0 for row in df_no_ss[!,:termination_afw]]

        #     df[!,:time_no_ss] = df_no_ss[!,:time_afw]
        #     df[!,:termination_no_ss] = termination_no_ss

        #     # load without as
        #     df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_mixed_50.csv")))
            
        #     filter!(row -> !(row.seed == 6 && row.dimension == 70),  df_no_as)
        #     filter!(row -> !(row.seed == 6 && row.dimension == 80),  df_no_as)
        #     filter!(row -> !(row.seed == 4 && row.dimension == 100),  df_no_as)
        #     filter!(row -> !(row.seed == 9 && row.dimension == 100),  df_no_as)f(x_t)^T(a_t - v_t) 

        #     df_no_as.termination_afw .= replace.(df_no_as.termination_afw, "Optimal (tree empty)" => "OPTIMAL")
        #     df_no_as.termination_afw .= replace.(df_no_as.termination_afw, "Time limit reached" => "TIME_LIMIT")
        #     termination_no_as = [row == "OPTIMAL" ? 1 : 0 for row in df_no_as[!,:termination_afw]]

        #     df[!,:time_no_as] = df_no_as[!,:time_afw]
        #     df[!,:termination_no_as] = termination_no_as
        
    elseif mode == "mixed_portfolio"
        # load boscia 
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_mixed_portfolio.csv")))

        df_bs.termination .= replace.(df_bs.termination, "Time limit reached" => "TIME_LIMIT")
        for row in eachrow(df_bs)
            if row.time > 1800
                row.termination = "TIME_LIMIT" 
            end
        end
        
        termination_boscia = [row == "OPTIMAL" || row == "tree.lb>primal-dual_gap" || row == "primal>tree.incumbent+1e-2" ? 1 : 0 for row in df_bs[!,:termination]]

        df[!,:dimension] = df_bs[!,:dimension]
        df[!,:time_boscia] = df_bs[!,:time]
        df[!,:seed] = df_bs[!,:seed]
        df[!,:solution_boscia] = df_bs[!,:solution]
        lowerBounds = df_bs[!, :solution] - df_bs[!, :dual_gap]
        df[!,:dual_gap] = df_bs[!,:dual_gap]
        df[!,:lb_boscia] = lowerBounds
        @show df[df.dimension .== 75, :]
        df[!,:rel_gap_boscia] = df_bs[!, :dual_gap] ./ min.(abs.(lowerBounds), abs.(df_bs[!, :solution]))
        df[!,:termination_boscia] = termination_boscia

        # load afw
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_mixed_portfolio.csv")))
        df_afw.termination .= replace.(df_afw.termination, "Time limit reached" => "TIME_LIMIT")
        for row in eachrow(df_afw)
            if row.time > 1800
                row.termination = "TIME_LIMIT" 
            end
        end
        
        termination_afw = [row == "OPTIMAL" || row == "tree.lb>primal-dual_gap" || row == "primal>tree.incumbent+1e-2" ? 1 : 0 for row in df_afw[!,:termination]]

        df_afw[!,:time_afw] = df_afw[!,:time]
        df_afw[!,:termination_afw] = termination_afw
        lowerBounds = df_afw[!, :solution] - df_afw[!, :dual_gap]
        df_afw[!, :rel_gap_afw] = (df_afw[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_afw[!, :solution]))
        df_afw = select(df_afw, [:termination_afw, :time_afw, :rel_gap_afw, :seed, :dimension])

        df = innerjoin(df, df_afw, on = [:seed, :dimension])

        # load without as, without ss
        df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mixed_portfolio.csv")))
        df_no_ws.termination .= replace.(df_no_ws.termination, "Time limit reached" => "TIME_LIMIT")
        for row in eachrow(df_no_ws)
            if row.time > 1800
                row.termination = "TIME_LIMIT" 
            end
        end

        termination_no_ws = [row == "OPTIMAL" || row == "tree.lb>primal-dual_gap" || row == "primal>tree.incumbent+1e-2" ? 1 : 0 for row in df_no_ws[!,:termination]]

        df_no_ws[!,:time_no_ws] = df_no_ws[!,:time]
        df_no_ws[!,:termination_no_ws] = termination_no_ws
        lowerBounds = df_no_ws[!, :solution] - df_no_ws[!, :dual_gap]
        df_no_ws[!, :rel_gap_no_ws] = (df_no_ws[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_ws[!, :solution]))
        df_no_ws = select(df_no_ws, [:termination_no_ws, :time_no_ws, :rel_gap_no_ws, :seed, :dimension])

        df = innerjoin(df, df_no_ws, on = [:seed, :dimension])

        # load ipopt 
        df_ipopt = DataFrame(CSV.File(joinpath(@__DIR__, "csv/ipopt_portfolio_mixed.csv")))
        df_ipopt.termination .= replace.(df_ipopt.termination, "Time limit reached" => "TIME_LIMIT")
        termination_ipopt = [row == "Optimal" ? 1 : 0 for row in df_ipopt[!, :termination]]
        
        df_ipopt[!, :time_ipopt] = df_ipopt[!, :time]/1000
        df_ipopt[!, :termination_ipopt] = termination_ipopt
        df_ipopt[!,:solution_ipopt] = df_ipopt[!,:solution]
        df_ipopt = select(df_ipopt, [:termination_ipopt, :time_ipopt, :solution_ipopt, :seed, :dimension])
        
        df = innerjoin(df, df_ipopt, on = [:seed, :dimension])   

        # load without as
        df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_mixed_portfolio.csv")))
        df_no_as.termination .= replace.(df_no_as.termination, "Time limit reached" => "TIME_LIMIT")
        for row in eachrow(df_no_as)
            if row.time > 1800
                row.termination = "TIME_LIMIT" 
            end
        end
        termination_no_as = [row == "OPTIMAL" || row == "tree.lb>primal-dual_gap" || row == "primal>tree.incumbent+1e-2" ? 1 : 0 for row in df_no_as[!,:termination]]

        df_no_as[!,:time_no_as] = df_no_as[!,:time]
        df_no_as[!,:termination_no_as] = termination_no_as
        lowerBounds = df_no_as[!, :solution] - df_no_as[!, :dual_gap]
        df_no_as[!, :rel_gap_no_as] = (df_no_as[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_as[!, :solution]))
        df_no_as = select(df_no_as, [:termination_no_as, :time_no_as, :rel_gap_no_as, :seed, :dimension])

        df = innerjoin(df, df_no_as, on = [:seed, :dimension])

        # load without ss
        df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_ss_mixed_portfolio.csv")))
        df_no_ss.termination .= replace.(df_no_ss.termination, "Time limit reached" => "TIME_LIMIT")
        for row in eachrow(df_no_ss)
            if row.time > 1800
                row.termination = "TIME_LIMIT" 
            end
        end
        termination_no_ss = [row == "OPTIMAL" || row == "tree.lb>primal-dual_gap" || row == "primal>tree.incumbent+1e-2" ? 1 : 0 for row in df_no_ss[!,:termination]]

        df_no_ss[!,:time_no_ss] = df_no_ss[!,:time]
        df_no_ss[!,:termination_no_ss] = termination_no_ss
        lowerBounds = df_no_ss[!, :solution] - df_no_ss[!, :dual_gap]
        df_no_ss[!, :rel_gap_no_ss] = (df_no_ss[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_ss[!, :solution]))
        df_no_ss = select(df_no_ss, [:termination_no_ss, :time_no_ss, :rel_gap_no_ss, :seed, :dimension])

        df = innerjoin(df, df_no_ss, on = [:seed, :dimension])

        # load scip oa
        df_scip = DataFrame(CSV.File(joinpath(@__DIR__, "csv/scip_oa_mixed_portfolio.csv"))) 
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
        df_scip = select(df_scip, [:termination_scip, :time_scip, :seed, :solution_scip, :dimension])

        df = innerjoin(df, df_scip, on = [:seed, :dimension])

        sort!(df, [:dimension])

        # check if solution optimal
        optimal_scip = []
        optimal_ipopt = []
        optimal_boscia = []
        for row in eachrow(df)
            if isapprox(row.lb_boscia, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_boscia, 1)
            else 
                append!(optimal_boscia, 0)
            end
            if isapprox(row.solution_ipopt, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_ipopt, 1)
            else 
                append!(optimal_ipopt, 0)
            end
            if isapprox(row.solution_scip, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_scip, 1)
            else 
                append!(optimal_scip, 0)
            end
        end
        df[!,:optimal_scip] = optimal_scip
        df[!,:optimal_ipopt] = optimal_ipopt
        df[!,:optimal_boscia] = optimal_boscia

        # compute relative gap
        rel_gap_scip = []
        rel_gap_ipopt = []
        for row in eachrow(df)
            if min(abs(row.solution_scip), abs(row.lb_boscia)) == 0
                push!(rel_gap_scip, row.solution_scip - row.lb_boscia)
            elseif sign(row.lb_boscia) != sign(row.solution_scip)
                push!(rel_gap_scip, Inf)
            else
                push!(rel_gap_scip, (row.solution_scip - row.lb_boscia)/min(abs(row.solution_scip), abs(row.lb_boscia)))
            end
            if min(abs(row.solution_ipopt), abs(row.lb_boscia)) == 0
                push!(rel_gap_ipopt, row.solution_ipopt - row.lb_boscia)
            elseif sign(row.lb_boscia) != sign(row.solution_ipopt)
                push!(rel_gap_ipopt, Inf)
            else
                push!(rel_gap_ipopt, (row.solution_ipopt - row.lb_boscia)/min(abs(row.solution_ipopt), abs(row.lb_boscia)))
            end
        end
        df[!, :rel_gap_scip] = round.(rel_gap_scip,digits=3)
        df[!, :rel_gap_ipopt] = round.(rel_gap_ipopt,digits=3)

        @show select(df[df.dimension .== 75, :], [:solution_boscia, :lb_boscia, :rel_gap_boscia])

        # save csv 
        file_name = joinpath(@__DIR__, "csv/mixed_portfolio_non_grouped.csv")
        CSV.write(file_name, df, append=false)
    
    elseif mode == "poisson"
        # load boscia 
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_poisson.csv")))
        # filter!(row -> !(row.seed == 7 && row.Ns == 10.0 && row.dimension == 70),  df_bs)

        df_bs.termination .= replace.(df_bs.termination, "Time limit reached" => "TIME_LIMIT")
        termination_boscia = [row == "TIME_LIMIT" ? 0 : 1 for row in df_bs[!,:termination]]

        df[!,:dimension] = df_bs[!,:dimension]
        df[!,:time_boscia] = df_bs[!,:time]
        df[!,:solution_boscia] = df_bs[!,:solution]
        df[!,:seed] = df_bs[!,:seed]
        df[!,:p] = df_bs[!,:p]
        df[!,:k] = df_bs[!,:k]
        df[!,:Ns] = df_bs[!,:Ns]
        lowerBounds = df_bs[!, :solution] - df_bs[!, :dual_gap]
        df[!,:lb_boscia] = lowerBounds
        df[!,:rel_gap_boscia] = df_bs[!, :dual_gap] ./ min.(abs.(lowerBounds), abs.(df_bs[!, :solution]))
        df[!,:termination_boscia] = termination_boscia

        # load afw
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_poisson.csv")))
        df_afw.termination .= replace.(df_afw.termination, "Time limit reached" => "TIME_LIMIT")
        termination_afw = [row == "TIME_LIMIT" ? 0 : 1 for row in df_afw[!,:termination]]

        df_afw[!,:time_afw] = df_afw[!,:time]
        df_afw[!,:termination_afw] = termination_afw
        lowerBounds = df_afw[!, :solution] - df_afw[!, :dual_gap]
        df_afw[!, :rel_gap_afw] = (df_afw[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_afw[!, :solution]))
        df_afw = select(df_afw, [:termination_afw, :time_afw, :rel_gap_afw, :seed, :dimension, :k, :Ns, :p])

        df = innerjoin(df, df_afw, on = [:seed, :dimension, :k, :Ns, :p])

        # load without as, without ss
        df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_poisson.csv")))
        df_no_ws.termination .= replace.(df_no_ws.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ws = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ws[!,:termination]]

        df_no_ws[!,:time_no_ws] = df_no_ws[!,:time]
        df_no_ws[!,:termination_no_ws] = termination_no_ws
        lowerBounds = df_no_ws[!, :solution] - df_no_ws[!, :dual_gap]
        df_no_ws[!, :rel_gap_no_ws] = (df_no_ws[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_ws[!, :solution]))
        df_no_ws = select(df_no_ws, [:termination_no_ws, :time_no_ws, :rel_gap_no_ws, :seed, :dimension, :k, :Ns, :p])

        df = innerjoin(df, df_no_ws, on = [:seed, :dimension, :k, :Ns, :p])
        # print(first(df,5))

        # load without as
        df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_poisson.csv")))
        df_no_as.termination .= replace.(df_no_as.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_as = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_as[!,:termination]]

        df_no_as[!,:time_no_as] = df_no_as[!,:time]
        df_no_as[!,:termination_no_as] = termination_no_as
        lowerBounds = df_no_as[!, :solution] - df_no_as[!, :dual_gap]
        df_no_as[!, :rel_gap_no_as] = (df_no_as[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_as[!, :solution]))
        df_no_as = select(df_no_as, [:termination_no_as, :time_no_as, :rel_gap_no_as, :seed, :dimension, :k, :Ns, :p])

        df = innerjoin(df, df_no_as, on = [:seed, :dimension, :k, :Ns, :p])

        # load without ss
        df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_ss_poisson.csv")))
        df_no_ss.termination .= replace.(df_no_ss.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ss = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ss[!,:termination]]

        df_no_ss[!,:time_no_ss] = df_no_ss[!,:time]
        df_no_ss[!,:termination_no_ss] = termination_no_ss
        lowerBounds = df_no_ss[!, :solution] - df_no_ss[!, :dual_gap]
        df_no_ss[!, :rel_gap_no_ss] = (df_no_ss[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_ss[!, :solution]))
        df_no_ss = select(df_no_ss, [:termination_no_ss, :time_no_ss, :rel_gap_no_ss, :seed, :dimension, :k, :Ns, :p])

        df = innerjoin(df, df_no_ss, on = [:seed, :dimension, :k, :Ns, :p])

        # load ipopt 
        df_ipopt = DataFrame(CSV.File(joinpath(@__DIR__, "csv/ipopt_poisson_reg.csv")))
        df_ipopt.termination .= replace.(df_ipopt.termination, "Time limit reached" => "TIME_LIMIT")
        termination_ipopt = [row == "TIME_LIMIT" ? 0 : 1 for row in df_ipopt[!, :termination]]
        
        df_ipopt[!, :time_ipopt] = df_ipopt[!, :time]/1000
        df_ipopt[!, :termination_ipopt] = termination_ipopt
        df_ipopt[!,:solution_ipopt] = df_ipopt[!,:solution]
        df_ipopt = select(df_ipopt, [:termination_ipopt, :time_ipopt, :solution_ipopt, :seed, :dimension, :k, :Ns, :p])
        
        df = innerjoin(df, df_ipopt, on = [:seed, :dimension, :k, :Ns, :p])

        # load scip oa
        df_scip = DataFrame(CSV.File(joinpath(@__DIR__, "csv/scip_oa_poisson.csv")))

        termination_scip = [row == "TIME_LIMIT" || row == "KILLED" || row == "INFEASIBLE" ? 0 : 1 for row in df_scip[!,:termination]]

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
        df_scip = select(df_scip, [:termination_scip, :time_scip, :seed, :solution_scip, :dimension, :k, :Ns, :p])

        # # delete duplicates
        # df_scip = unique(df_scip, [:dimension, :k, :Ns, :seed])

        df = innerjoin(df, df_scip, on = [:seed, :dimension, :k, :Ns, :p])

        sort!(df, [:dimension, :k, :Ns])


        # check if solution optimal
        optimal_scip = []
        optimal_ipopt = []
        optimal_boscia = []
        for row in eachrow(df)
            if isapprox(row.lb_boscia, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_boscia, 1)
            else 
                append!(optimal_boscia, 0)
            end
            if isapprox(row.solution_ipopt, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_ipopt, 1)
            else 
                append!(optimal_ipopt, 0)
            end
            if isapprox(row.solution_scip, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_scip, 1)
            else 
                append!(optimal_scip, 0)
            end
        end
        df[!,:optimal_scip] = optimal_scip
        df[!,:optimal_ipopt] = optimal_ipopt
        df[!,:optimal_boscia] = optimal_boscia

        # compute lower bound 
        rel_gap_scip = []
        rel_gap_ipopt = []
        for row in eachrow(df)
            if min(abs(row.solution_scip), abs(row.lb_boscia)) == 0
                push!(rel_gap_scip, row.solution_scip - row.lb_boscia)
            elseif sign(row.lb_boscia) != sign(row.solution_scip)
                push!(rel_gap_scip, Inf)
            else
                push!(rel_gap_scip, (row.solution_scip - row.lb_boscia)/min(abs(row.solution_scip), abs(row.lb_boscia)))
            end
            if min(abs(row.solution_ipopt), abs(row.lb_boscia)) == 0
                push!(rel_gap_ipopt, row.solution_ipopt - row.lb_boscia)
            elseif sign(row.lb_boscia) != sign(row.solution_ipopt)
                push!(rel_gap_ipopt, Inf)
            else
                push!(rel_gap_ipopt, (row.solution_ipopt - row.lb_boscia)/min(abs(row.solution_ipopt), abs(row.lb_boscia)))
            end
        end
        df[!, :rel_gap_scip] = round.(rel_gap_scip,digits=3)
        df[!, :rel_gap_ipopt] = round.(rel_gap_ipopt,digits=3)

        filter!(row -> (isapprox(row.solution_ipopt,row.solution_boscia,atol=1.0)) || row.solution_ipopt==Inf, df)

        # save csv 
        file_name = joinpath(@__DIR__, "csv/poisson_non_grouped.csv")
        CSV.write(file_name, df, append=false)

    elseif mode == "sparse_reg"
        # load boscia 
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_sparse_reg.csv")))
        # delete duplicates
        df_bs = unique(df_bs, [:dimension, :k, :p, :seed])

        df_bs.termination .= replace.(df_bs.termination, "Time limit reached" => "TIME_LIMIT")
        termination_boscia = [row == "TIME_LIMIT" ? 0 : 1 for row in df_bs[!,:termination]]

        df[!,:dimension] = df_bs[!,:dimension]
        df[!,:time_boscia] = df_bs[!,:time]
        df[!,:seed] = df_bs[!,:seed]
        df[!,:p] = df_bs[!,:p]
        df[!,:k] = df_bs[!,:k]

        df[!,:termination_boscia] = termination_boscia
        df[!, :solution_boscia] = df_bs[!, :solution]
        lowerBounds = df_bs[!, :solution] - df_bs[!, :dual_gap]
        df[!, :lb_boscia] = lowerBounds
        df[!, :rel_gap_boscia] = df_bs[!, :dual_gap] ./ min.(abs.(lowerBounds), abs.(df_bs[!, :solution]))

        # load afw
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_sparse_reg.csv")))
        df_afw.termination .= replace.(df_afw.termination, "Time limit reached" => "TIME_LIMIT")
        termination_afw = [row == "TIME_LIMIT" ? 0 : 1 for row in df_afw[!,:termination]]

        df_afw[!,:time_afw] = df_afw[!,:time]
        df_afw[!,:termination_afw] = termination_afw
        lowerBounds = df_afw[!, :solution] - df_afw[!, :dual_gap]
        df_afw[!, :rel_gap_afw] = (df_afw[!, :solution] - lowerBounds)./ min.(abs.(lowerBounds), abs.(df_afw[!, :solution]))
        df_afw = select(df_afw, [:termination_afw, :time_afw, :rel_gap_afw, :seed, :dimension, :k, :p])

        df = innerjoin(df, df_afw, on = [:seed, :dimension, :k, :p])

        # load without as, without ss
        df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_sparse_reg.csv")))
        df_no_ws.termination .= replace.(df_no_ws.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ws = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ws[!,:termination]]

        df_no_ws[!,:time_no_ws] = df_no_ws[!,:time]
        df_no_ws[!,:termination_no_ws] = termination_no_ws
        lowerBounds = df_no_ws[!, :solution] - df_no_ws[!, :dual_gap]
        df_no_ws[!, :rel_gap_no_ws] = (df_no_ws[!, :solution] - lowerBounds)./ min.(abs.(lowerBounds), abs.(df_no_ws[!, :solution]))
        df_no_ws = select(df_no_ws, [:termination_no_ws, :time_no_ws, :rel_gap_no_ws, :seed, :dimension, :k, :p])

        df = innerjoin(df, df_no_ws, on = [:seed, :dimension, :k, :p])
        # print(first(df,5))

        # load without as
        df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_sparse_reg.csv")))
        df_no_as.termination .= replace.(df_no_as.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_as = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_as[!,:termination]]

        df_no_as[!,:time_no_as] = df_no_as[!,:time]
        df_no_as[!,:termination_no_as] = termination_no_as
        lowerBounds = df_no_as[!, :solution] - df_no_as[!, :dual_gap]
        df_no_as[!, :rel_gap_no_as] = (df_no_as[!, :solution] - lowerBounds)./ min.(abs.(lowerBounds), abs.(df_no_as[!, :solution]))
        df_no_as = select(df_no_as, [:termination_no_as, :time_no_as, :rel_gap_no_as, :seed, :dimension, :k, :p])

        df = innerjoin(df, df_no_as, on = [:seed, :dimension, :k, :p])

        # load without ss
        df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_ss_sparse_reg.csv")))
        df_no_ss.termination .= replace.(df_no_ss.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ss = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ss[!,:termination]]

        df_no_ss[!,:time_no_ss] = df_no_ss[!,:time]
        df_no_ss[!,:termination_no_ss] = termination_no_ss
        lowerBounds = df_no_ss[!, :solution] - df_no_ss[!, :dual_gap]
        df_no_ss[!, :rel_gap_no_ss] = (df_no_ss[!, :solution] - lowerBounds)./ min.(abs.(lowerBounds), abs.(df_no_ss[!, :solution]))
        df_no_ss = select(df_no_ss, [:termination_no_ss, :time_no_ss, :rel_gap_no_ss, :seed, :dimension, :k, :p])

        df = innerjoin(df, df_no_ss, on = [:seed, :dimension, :k, :p])

        # load local tightening
        df_loc_ti = DataFrame(CSV.File(joinpath(@__DIR__, "csv/local_tightening_sparse_reg.csv")))
        df_loc_ti.termination .= replace.(df_loc_ti.termination, "Time limit reached" => "TIME_LIMIT")
        termination_loc_ti = [row == "TIME_LIMIT" ? 0 : 1 for row in df_loc_ti[!, :termination]]

        df_loc_ti[!, :time_loc_ti] = df_loc_ti[!, :time]
        df_loc_ti[!, :termination_loc_ti] = termination_loc_ti
        df_loc_ti[!, :lowerBounds] = df_loc_ti[!, :solution] - df_loc_ti[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_loc_ti)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_loc_ti[!, :rel_gap_loc_ti] = rel_gap
        df_loc_ti = select(df_loc_ti, [:termination_loc_ti, :time_loc_ti, :rel_gap_loc_ti, :seed, :dimension, :k, :p])

        df = innerjoin(df, df_loc_ti, on = [:seed, :dimension, :k, :p])

        # load local tightening
        df_gl_ti = DataFrame(CSV.File(joinpath(@__DIR__, "csv/global_tightening_sparse_reg.csv")))
        df_gl_ti.termination .= replace.(df_gl_ti.termination, "Time limit reached" => "TIME_LIMIT")
        termination_gl_ti = [row == "TIME_LIMIT" ? 0 : 1 for row in df_gl_ti[!, :termination]]

        df_gl_ti[!, :time_gl_ti] = df_gl_ti[!, :time]
        df_gl_ti[!, :termination_gl_ti] = termination_gl_ti
        df_gl_ti[!, :lowerBounds] = df_gl_ti[!, :solution] - df_gl_ti[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_gl_ti)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_gl_ti[!, :rel_gap_gl_ti] = rel_gap
        df_gl_ti = select(df_gl_ti, [:termination_gl_ti, :time_gl_ti, :rel_gap_gl_ti, :seed, :dimension, :k, :p])

        df = innerjoin(df, df_gl_ti, on = [:seed, :dimension, :k, :p])

        # load local tightening
        df_no_ti = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_tightening_sparse_reg.csv")))
        df_no_ti.termination .= replace.(df_no_ti.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ti = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ti[!, :termination]]

        df_no_ti[!, :time_no_ti] = df_no_ti[!, :time]
        df_no_ti[!, :termination_no_ti] = termination_no_ti
        df_no_ti[!, :lowerBounds] = df_no_ti[!, :solution] - df_no_ti[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_no_ti)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_no_ti[!, :rel_gap_no_ti] = rel_gap
        df_no_ti = select(df_no_ti, [:termination_no_ti, :time_no_ti, :rel_gap_no_ti, :seed, :dimension, :k, :p])

        df = innerjoin(df, df_no_ti, on = [:seed, :dimension, :k, :p])

        # load ipopt 
        df_ipopt = DataFrame(CSV.File(joinpath(@__DIR__, "csv/ipopt_sparse_reg.csv")))
        df_ipopt.termination .= replace.(df_ipopt.termination, "Time limit reached" => "TIME_LIMIT")
        termination_ipopt = [row == "Optimal" ? 1 : 0 for row in df_ipopt[!, :termination]]

        df_ipopt[!, :time_ipopt] = df_ipopt[!, :time]/1000
        df_ipopt[!, :termination_ipopt] = termination_ipopt
        df_ipopt[!, :solution_ipopt] = df_ipopt[!, :solution]
        df_ipopt = select(df_ipopt, [:termination_ipopt, :time_ipopt, :solution_ipopt, :seed, :dimension, :k, :p])

        df = innerjoin(df, df_ipopt, on = [:seed, :dimension, :k, :p])

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

        df = innerjoin(df, df_scip, on = [:seed, :dimension, :k, :p])

        # load scip oa tol 1e-9
        df_scip_tol = DataFrame(CSV.File(joinpath(@__DIR__, "csv/scip_oa_sparse_reg_ 1.0e-9.csv")))

        termination_scip_tol = [row == "OPTIMAL" ? 1 : 0 for row in df_scip_tol[!,:termination]]

        time_scip_tol = []
        for row in eachrow(df_scip_tol)
            if row.solution == Inf 
                append!(time_scip_tol,1800) 
            else 
                append!(time_scip_tol,row.time)
            end
        end

        df_scip_tol[!,:time_scip_tol] = time_scip_tol
        df_scip_tol[!,:termination_scip_tol] = termination_scip_tol
        df_scip_tol[!,:solution_scip_tol] = df_scip_tol[!,:solution]
        df_scip_tol = select(df_scip_tol, [:solution_scip_tol, :termination_scip_tol, :time_scip_tol, :seed, :dimension, :k, :p])

        df = innerjoin(df, df_scip_tol, on = [:seed, :dimension, :k, :p])

        # check if solution optimal
        optimal_scip = []
        optimal_ipopt = []
        optimal_boscia = []
        for row in eachrow(df)
            if isapprox(row.lb_boscia, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_boscia, 1)
            else 
                append!(optimal_boscia, 0)
            end
            if isapprox(row.solution_ipopt, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_ipopt, 1)
            else 
                append!(optimal_ipopt, 0)
            end
            if isapprox(row.solution_scip, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_scip, 1)
            else 
                append!(optimal_scip, 0)
            end
        end
        df[!,:optimal_scip] = optimal_scip
        df[!,:optimal_ipopt] = optimal_ipopt
        df[!,:optimal_boscia] = optimal_boscia

        # compute lower bound 
        rel_gap_scip = []
        rel_gap_ipopt = []
        for row in eachrow(df)
            if min(abs(row.solution_scip), abs(row.lb_boscia)) == 0
                push!(rel_gap_scip, row.solution_scip - row.lb_boscia)
            elseif sign(row.lb_boscia) != sign(row.solution_scip)
                push!(rel_gap_scip, Inf)
            else
                push!(rel_gap_scip, (row.solution_scip - row.lb_boscia)/min(abs(row.solution_scip), abs(row.lb_boscia)))
            end
            if min(abs(row.solution_ipopt), abs(row.lb_boscia)) == 0
                push!(rel_gap_ipopt, row.solution_ipopt - row.lb_boscia)
            elseif sign(row.lb_boscia) != sign(row.solution_ipopt)
                push!(rel_gap_ipopt, Inf)
            else
                push!(rel_gap_ipopt, (row.solution_ipopt - row.lb_boscia)/min(abs(row.solution_ipopt), abs(row.lb_boscia)))
            end
        end
        df[!, :rel_gap_scip] = round.(rel_gap_scip,digits=3)
        df[!, :rel_gap_ipopt] = round.(rel_gap_ipopt,digits=3)

        # save csv 
        file_name = joinpath(@__DIR__, "csv/sparse_reg_non_grouped.csv")
        CSV.write(file_name, df, append=false)
    
    elseif mode == "sparse_log_reg"  
        # load boscia 
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_sparse_log_regression.csv")))
        # delete duplicates
        df_bs = unique(df_bs, [:dimension, :k, :p, :seed, :M, :var_A])

        df_bs.termination .= replace.(df_bs.termination, "Time limit reached" => "TIME_LIMIT")
        termination_boscia = [row == "TIME_LIMIT" ? 0 : 1 for row in df_bs[!,:termination]]

        df[!,:dimension] = df_bs[!,:dimension]
        df[!,:time_boscia] = df_bs[!,:time]
        df[!,:seed] = df_bs[!,:seed]
        df[!,:p] = df_bs[!,:p]
        df[!,:k] = df_bs[!,:k]
        df[!,:M] = df_bs[!,:M]
        df[!,:var_A] = df_bs[!,:var_A]

        df[!,:termination_boscia] = termination_boscia
        df[!, :solution_boscia] = df_bs[!, :solution]  
        lowerBounds = df_bs[!, :solution] - df_bs[!, :dual_gap]
        df[!, :lb_boscia] = lowerBounds
        df[!, :rel_gap_boscia] = df_bs[!, :dual_gap] ./ min.(abs.(lowerBounds), abs.(df_bs[!, :solution]))

        # load afw
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_sparse_log_regression.csv")))
        df_afw.termination .= replace.(df_afw.termination, "Time limit reached" => "TIME_LIMIT")
        termination_afw = [row == "TIME_LIMIT" ? 0 : 1 for row in df_afw[!,:termination]]

        df_afw[!,:time_afw] = df_afw[!,:time]
        df_afw[!,:termination_afw] = termination_afw
        lowerBounds = df_afw[!, :solution] - df_afw[!, :dual_gap]
        df_afw[!, :rel_gap_afw] = (df_afw[!, :solution]- lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_afw[!, :solution]))
        df_afw = select(df_afw, [:termination_afw, :time_afw, :rel_gap_afw, :dimension, :k, :p, :seed, :M, :var_A])

        df = innerjoin(df, df_afw, on = [:dimension, :k, :p, :seed, :M, :var_A])

        # load without as, without ss
        df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_sparse_log_regression.csv")))
        df_no_ws.termination .= replace.(df_no_ws.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ws = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ws[!,:termination]]

        df_no_ws[!,:time_no_ws] = df_no_ws[!,:time]
        df_no_ws[!,:termination_no_ws] = termination_no_ws
        lowerBounds = df_no_ws[!, :solution] - df_no_ws[!, :dual_gap]
        df_no_ws[!, :rel_gap_no_ws] = (df_no_ws[!, :solution]- lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_ws[!, :solution]))
        df_no_ws = select(df_no_ws, [:termination_no_ws, :time_no_ws, :rel_gap_no_ws, :dimension, :k, :p, :seed, :M, :var_A])

        df = innerjoin(df, df_no_ws, on = [:dimension, :k, :p, :seed, :M, :var_A])
        # print(first(df,5))

        # load without as
        df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_sparse_log_regression.csv")))
        df_no_as.termination .= replace.(df_no_as.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_as = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_as[!,:termination]]

        df_no_as[!,:time_no_as] = df_no_as[!,:time]
        df_no_as[!,:termination_no_as] = termination_no_as
        lowerBounds = df_no_as[!, :solution] - df_no_as[!, :dual_gap]
        df_no_as[!, :rel_gap_no_as] = (df_no_as[!, :solution]- lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_as[!, :solution]))
        df_no_as = select(df_no_as, [:termination_no_as, :time_no_as, :rel_gap_no_as, :dimension, :k, :p, :seed, :M, :var_A])

        df = innerjoin(df, df_no_as, on = [:dimension, :k, :p, :seed, :M, :var_A])

        # load without ss
        df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_ss_sparse_log_regression.csv")))
        df_no_ss.termination .= replace.(df_no_ss.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ss = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ss[!,:termination]]
        lowerBounds = df_no_ss[!, :solution] - df_no_ss[!, :dual_gap]
        df_no_ss[!,:time_no_ss] = df_no_ss[!,:time]
        df_no_ss[!,:termination_no_ss] = termination_no_ss
        
        df_no_ss[!, :rel_gap_no_ss] = (df_no_ss[!, :solution]- lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_ss[!, :solution]))
        df_no_ss = select(df_no_ss, [:termination_no_ss, :time_no_ss, :rel_gap_no_ss, :dimension, :k, :p, :seed, :M, :var_A])

        df = innerjoin(df, df_no_ss, on = [:dimension, :k, :p, :seed, :M, :var_A])

        # load local tightening
        df_loc_ti = DataFrame(CSV.File(joinpath(@__DIR__, "csv/local_tightening_sparse_log_regression.csv")))
        df_loc_ti.termination .= replace.(df_loc_ti.termination, "Time limit reached" => "TIME_LIMIT")
        termination_loc_ti = [row == "TIME_LIMIT" ? 0 : 1 for row in df_loc_ti[!, :termination]]

        df_loc_ti[!, :time_loc_ti] = df_loc_ti[!, :time]
        df_loc_ti[!, :termination_loc_ti] = termination_loc_ti
        df_loc_ti[!, :lowerBounds] = df_loc_ti[!, :solution] - df_loc_ti[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_loc_ti)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_loc_ti[!, :rel_gap_loc_ti] = rel_gap
        df_loc_ti = select(df_loc_ti, [:termination_loc_ti, :time_loc_ti, :rel_gap_loc_ti, :dimension, :k, :p, :seed, :M, :var_A])

        df = innerjoin(df, df_loc_ti, on = [:seed, :dimension, :k, :p, :M, :var_A])

        # load local tightening
        df_gl_ti = DataFrame(CSV.File(joinpath(@__DIR__, "csv/global_tightening_sparse_log_regression.csv")))
        df_gl_ti.termination .= replace.(df_gl_ti.termination, "Time limit reached" => "TIME_LIMIT")
        termination_gl_ti = [row == "TIME_LIMIT" ? 0 : 1 for row in df_gl_ti[!, :termination]]

        df_gl_ti[!, :time_gl_ti] = df_gl_ti[!, :time]
        df_gl_ti[!, :termination_gl_ti] = termination_gl_ti
        df_gl_ti[!, :lowerBounds] = df_gl_ti[!, :solution] - df_gl_ti[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_gl_ti)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_gl_ti[!, :rel_gap_gl_ti] = rel_gap
        df_gl_ti = select(df_gl_ti, [:termination_gl_ti, :time_gl_ti, :rel_gap_gl_ti, :seed, :dimension, :k, :p, :M, :var_A])

        df = innerjoin(df, df_gl_ti, on = [:seed, :dimension, :k, :p, :M, :var_A])

        # load local tightening
        df_no_ti = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_tightening_sparse_log_regression.csv")))
        df_no_ti.termination .= replace.(df_no_ti.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ti = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ti[!, :termination]]

        df_no_ti[!, :time_no_ti] = df_no_ti[!, :time]
        df_no_ti[!, :termination_no_ti] = termination_no_ti
        df_no_ti[!, :lowerBounds] = df_no_ti[!, :solution] - df_no_ti[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_no_ti)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_no_ti[!, :rel_gap_no_ti] = rel_gap
        df_no_ti = select(df_no_ti, [:termination_no_ti, :time_no_ti, :rel_gap_no_ti, :seed, :dimension, :k, :p, :M, :var_A])

        df = innerjoin(df, df_no_ti, on = [:seed, :dimension, :k, :p, :M, :var_A])

        # load ipopt 
        df_ipopt = DataFrame(CSV.File(joinpath(@__DIR__, "csv/ipopt_sparse_log_reg.csv")))
        df_ipopt.termination .= replace.(df_ipopt.termination, "Time limit reached" => "TIME_LIMIT")
        termination_ipopt = [row == "TIME_LIMIT" ? 0 : 1 for row in df_ipopt[!, :termination]]

        df_ipopt[!, :time_ipopt] = df_ipopt[!, :time]/1000
        df_ipopt[!, :termination_ipopt] = termination_ipopt
        df_ipopt[!, :solution_ipopt] = df_ipopt[!, :solution]
        df_ipopt = select(df_ipopt, [:termination_ipopt, :time_ipopt, :solution_ipopt, :dimension, :k, :p, :seed, :M, :var_A])

        df = innerjoin(df, df_ipopt, on = [:dimension, :k, :p, :seed, :M, :var_A])

        # load scip oa
        df_scip = DataFrame(CSV.File(joinpath(@__DIR__, "csv/scip_oa_sparse_log_regression.csv")))
        termination_scip = [row == "OPTIMAL" ? 1 : 0 for row in df_scip[!,:termination]]

        df_scip[!,:time_scip] = df_scip[!,:time]
        df_scip[!,:termination_scip] = termination_scip
        df_scip[!,:solution_scip] = df_scip[!,:solution]
        df_scip = select(df_scip, [:solution_scip, :termination_scip, :time_scip, :seed, :dimension, :k, :p, :M, :var_A])

        df = innerjoin(df, df_scip, on = [:seed, :dimension, :k, :p, :M, :var_A])

        # check if solution optimal
        optimal_scip = []
        optimal_ipopt = []
        optimal_boscia = []
        for row in eachrow(df)
            if isapprox(row.lb_boscia, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_boscia, 1)
            else 
                append!(optimal_boscia, 0)
            end
            if isapprox(row.solution_ipopt, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_ipopt, 1)
            else 
                append!(optimal_ipopt, 0)
            end
            if isapprox(row.solution_scip, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_scip, 1)
            else 
                append!(optimal_scip, 0)
            end
        end
        df[!,:optimal_scip] = optimal_scip
        df[!,:optimal_ipopt] = optimal_ipopt
        df[!,:optimal_boscia] = optimal_boscia

        rename!(df, :var_A => :varA)

        # compute lower bound 
        rel_gap_scip = []
        rel_gap_ipopt = []
        for row in eachrow(df)
            if min(abs(row.solution_scip), abs(row.lb_boscia)) == 0
                push!(rel_gap_scip, row.solution_scip - row.lb_boscia)
            elseif sign(row.lb_boscia) != sign(row.solution_scip)
                push!(rel_gap_scip, Inf)
            else
                push!(rel_gap_scip, (row.solution_scip - row.lb_boscia)/min(abs(row.solution_scip), abs(row.lb_boscia)))
            end
            if min(abs(row.solution_ipopt), abs(row.lb_boscia)) == 0
                push!(rel_gap_ipopt, row.solution_ipopt - row.lb_boscia)
            elseif sign(row.lb_boscia) != sign(row.solution_ipopt)
                push!(rel_gap_ipopt, Inf)
            else
                push!(rel_gap_ipopt, (row.solution_ipopt - row.lb_boscia)/min(abs(row.solution_ipopt), abs(row.lb_boscia)))
            end
        end
        df[!, :rel_gap_scip] = round.(rel_gap_scip,digits=3)
        df[!, :rel_gap_ipopt] = round.(rel_gap_ipopt,digits=3)

        # save csv 
        file_name = joinpath(@__DIR__, "csv/sparse_log_reg_non_grouped.csv")
        CSV.write(file_name, df, append=false)

    elseif mode == "tailed_cardinality"
        # load boscia 
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_tailed_cardinality.csv")))
        # filter!(row -> !(row.seed == 7 && row.Ns == 10.0 && row.dimension == 70),  df_bs)

        df_bs.termination .= replace.(df_bs.termination, "Time limit reached" => "TIME_LIMIT")
        termination_boscia = [row == "TIME_LIMIT" ? 0 : 1 for row in df_bs[!,:termination]]

        df[!,:time_boscia] = df_bs[!,:time]
        df[!,:seed] = df_bs[!,:seed]
        df[!,:n0] = df_bs[!,:n0]
        df[!,:m0] = df_bs[!,:m0]
        df[!,:M] = df_bs[!,:M]
        df[!,:solution_boscia] = df_bs[!,:solution]

        df[!,:termination_boscia] = termination_boscia
        lowerBounds = df_bs[!, :solution] - df_bs[!, :dual_gap]
        df[!, :lb_boscia] = lowerBounds
        df[!, :rel_gap_boscia] = df_bs[!, :dual_gap] ./ min.(abs.(lowerBounds), abs.(df_bs[!, :solution]))

        # load afw
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_tailed_cardinality.csv")))
        df_afw.termination .= replace.(df_afw.termination, "Time limit reached" => "TIME_LIMIT")
        termination_afw = [row == "TIME_LIMIT" ? 0 : 1 for row in df_afw[!,:termination]]

        df_afw[!,:time_afw] = df_afw[!,:time]
        df_afw[!,:termination_afw] = termination_afw
        lowerBounds = df_afw[!, :solution] - df_afw[!, :dual_gap]
        df_afw[!, :rel_gap_afw] = (df_afw[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_afw[!, :solution]))
        df_afw = select(df_afw, [:termination_afw, :time_afw, :rel_gap_afw, :seed, :n0, :m0, :M])

        df = innerjoin(df, df_afw, on = [:seed, :n0, :m0, :M])

        # load without as, without ss
        df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_tailed_cardinality.csv")))
        df_no_ws.termination .= replace.(df_no_ws.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ws = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ws[!,:termination]]

        df_no_ws[!,:time_no_ws] = df_no_ws[!,:time]
        df_no_ws[!,:termination_no_ws] = termination_no_ws
        lowerBounds = df_no_ws[!, :solution] - df_no_ws[!, :dual_gap]
        df_no_ws[!, :rel_gap_no_ws] = (df_no_ws[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_ws[!, :solution]))
        df_no_ws = select(df_no_ws, [:termination_no_ws, :time_no_ws, :rel_gap_no_ws, :seed, :n0, :m0, :M])

        df = innerjoin(df, df_no_ws, on = [:seed, :n0, :m0, :M])
        # print(first(df,5))

        # load without as
        df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_tailed_cardinality.csv")))
        df_no_as.termination .= replace.(df_no_as.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_as = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_as[!,:termination]]

        df_no_as[!,:time_no_as] = df_no_as[!,:time]
        df_no_as[!,:termination_no_as] = termination_no_as
        lowerBounds = df_no_as[!, :solution] - df_no_as[!, :dual_gap]
        df_no_as[!, :rel_gap_no_as] = (df_no_as[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_as[!, :solution]))
        df_no_as = select(df_no_as, [:termination_no_as, :time_no_as, :rel_gap_no_as, :seed, :n0, :m0, :M])

        df = innerjoin(df, df_no_as, on = [:seed, :n0, :m0, :M])

        # load without ss
        df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_ss_tailed_cardinality.csv")))
        df_no_ss.termination .= replace.(df_no_ss.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ss = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ss[!,:termination]]

        df_no_ss[!,:time_no_ss] = df_no_ss[!,:time]
        df_no_ss[!,:termination_no_ss] = termination_no_ss
        lowerBounds = df_no_ss[!, :solution] - df_no_ss[!, :dual_gap]
        df_no_ss[!, :rel_gap_no_ss] = (df_no_ss[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_ss[!, :solution]))
        df_no_ss = select(df_no_ss, [:termination_no_ss, :time_no_ss, :rel_gap_no_ss, :seed, :n0, :m0, :M])

        df = innerjoin(df, df_no_ss, on = [:seed, :n0, :m0, :M])

        # load local tightening
        df_loc_ti = DataFrame(CSV.File(joinpath(@__DIR__, "csv/local_tightening_tailed_cardinality.csv")))
        df_loc_ti.termination .= replace.(df_loc_ti.termination, "Time limit reached" => "TIME_LIMIT")
        termination_loc_ti = [row == "TIME_LIMIT" ? 0 : 1 for row in df_loc_ti[!, :termination]]

        df_loc_ti[!, :time_loc_ti] = df_loc_ti[!, :time]
        df_loc_ti[!, :termination_loc_ti] = termination_loc_ti
        df_loc_ti[!, :lowerBounds] = df_loc_ti[!, :solution] - df_loc_ti[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_loc_ti)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_loc_ti[!, :rel_gap_loc_ti] = rel_gap
        df_loc_ti = select(df_loc_ti, [:termination_loc_ti, :time_loc_ti, :rel_gap_loc_ti, :seed, :n0, :m0, :M])

        df = innerjoin(df, df_loc_ti, on = [:seed, :n0, :m0, :M])

        # load local tightening
        df_gl_ti = DataFrame(CSV.File(joinpath(@__DIR__, "csv/global_tightening_tailed_cardinality.csv")))
        df_gl_ti.termination .= replace.(df_gl_ti.termination, "Time limit reached" => "TIME_LIMIT")
        termination_gl_ti = [row == "TIME_LIMIT" ? 0 : 1 for row in df_gl_ti[!, :termination]]

        df_gl_ti[!, :time_gl_ti] = df_gl_ti[!, :time]
        df_gl_ti[!, :termination_gl_ti] = termination_gl_ti
        df_gl_ti[!, :lowerBounds] = df_gl_ti[!, :solution] - df_gl_ti[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_gl_ti)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_gl_ti[!, :rel_gap_gl_ti] = rel_gap
        df_gl_ti = select(df_gl_ti, [:termination_gl_ti, :time_gl_ti, :rel_gap_gl_ti, :seed, :n0, :m0, :M])

        df = innerjoin(df, df_gl_ti, on = [:seed, :n0, :m0, :M])

        # load local tightening
        df_no_ti = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_tightening_tailed_cardinality.csv")))
        df_no_ti.termination .= replace.(df_no_ti.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ti = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ti[!, :termination]]

        df_no_ti[!, :time_no_ti] = df_no_ti[!, :time]
        df_no_ti[!, :termination_no_ti] = termination_no_ti
        df_no_ti[!, :lowerBounds] = df_no_ti[!, :solution] - df_no_ti[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_no_ti)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_no_ti[!, :rel_gap_no_ti] = rel_gap
        df_no_ti = select(df_no_ti, [:termination_no_ti, :time_no_ti, :rel_gap_no_ti, :seed, :n0, :m0, :M])

        df = innerjoin(df, df_no_ti, on = [:seed, :n0, :m0, :M])

        # load scip oa
        df_scip = DataFrame(CSV.File(joinpath(@__DIR__, "csv/scip_oa_tailed_cardinality.csv")))

        termination_scip = [row == "TIME_LIMIT" ? 0 : 1 for row in df_scip[!,:termination]]

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

        # check if solution optimal
        optimal_scip = []
        optimal_boscia = []
        for row in eachrow(df)
            if isapprox(row.lb_boscia, min(row.lb_boscia, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_boscia, 1)
            else 
                append!(optimal_boscia, 0)
            end
            if isapprox(row.solution_scip, min(row.lb_boscia, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_scip, 1)
            else 
                append!(optimal_scip, 0)
            end
        end
        df[!,:optimal_scip] = optimal_scip
        df[!,:optimal_boscia] = optimal_boscia
                
        # compute lower bound 
        rel_gap_scip = []
        for row in eachrow(df)
            if min(abs(row.solution_scip), abs(row.lb_boscia)) == 0
                push!(rel_gap_scip, row.solution_scip - row.lb_boscia)
            elseif sign(row.lb_boscia) != sign(row.solution_scip)
                push!(rel_gap_scip, Inf)
            else
                push!(rel_gap_scip, (row.solution_scip - row.lb_boscia)/min(abs(row.solution_scip), abs(row.lb_boscia)))
            end
        end
        df[!, :rel_gap_scip] = round.(rel_gap_scip,digits=3)

        # save csv 
        file_name = joinpath(@__DIR__, "csv/tailed_cardinality_non_grouped.csv")
        CSV.write(file_name, df, append=false)
    
    elseif mode == "tailed_cardinality_sparse_log_reg"
        # load boscia 
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_tailed_cardinality_sparse_log_reg.csv")))
        # filter!(row -> !(row.seed == 7 && row.Ns == 10.0 && row.dimension == 70),  df_bs)

        df_bs.termination .= replace.(df_bs.termination, "Time limit reached" => "TIME_LIMIT")
        termination_boscia = [row == "TIME_LIMIT" ? 0 : 1 for row in df_bs[!,:termination]]

        df[!,:dimension] = df_bs[!,:dimension]
        df[!,:time_boscia] = df_bs[!,:time]
        df[!,:seed] = df_bs[!,:seed]
        df[!,:M] = df_bs[!,:M]
        df[!,:var_A] = df_bs[!,:var_A]
        df[!,:termination_boscia] = termination_boscia
        df[!, :solution_boscia] = df_bs[!, :solution]  
        lowerBounds = df_bs[!, :solution] - df_bs[!, :dual_gap]
        df[!, :lb_boscia] = lowerBounds
        df[!, :rel_gap_boscia] = df_bs[!, :dual_gap] ./ min.(abs.(lowerBounds), abs.(df_bs[!, :solution]))
        
        # load afw
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_tailed_cardinality_sparse_log_reg.csv")))
        df_afw.termination .= replace.(df_afw.termination, "Time limit reached" => "TIME_LIMIT")
        termination_afw = [row == "TIME_LIMIT" ? 0 : 1 for row in df_afw[!,:termination]]

        df_afw[!,:time_afw] = df_afw[!,:time]
        df_afw[!,:termination_afw] = termination_afw
        lowerBounds = df_afw[!, :solution] - df_afw[!, :dual_gap]
        df_afw[!, :rel_gap_afw] = (df_afw[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_afw[!, :solution]))
        df_afw = select(df_afw, [:termination_afw, :time_afw, :seed, :rel_gap_afw, :dimension, :M, :var_A])

        df = innerjoin(df, df_afw, on = [:seed, :dimension, :M, :var_A])

        # load without as, without ss
        df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_tailed_cardinality_sparse_log_reg.csv")))
        df_no_ws.termination .= replace.(df_no_ws.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ws = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ws[!,:termination]]

        df_no_ws[!,:time_no_ws] = df_no_ws[!,:time]
        df_no_ws[!,:termination_no_ws] = termination_no_ws
        lowerBounds = df_no_ws[!, :solution] - df_no_ws[!, :dual_gap]
        df_no_ws[!, :rel_gap_no_ws] = (df_no_ws[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_ws[!, :solution]))
        df_no_ws = select(df_no_ws, [:termination_no_ws, :time_no_ws, :rel_gap_no_ws, :seed, :dimension, :M, :var_A])

        df = innerjoin(df, df_no_ws, on = [:seed, :dimension, :M, :var_A])
        # print(first(df,5))

        # load without as
        df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_tailed_cardinality_sparse_log_reg.csv")))
        df_no_as.termination .= replace.(df_no_as.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_as = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_as[!,:termination]]

        df_no_as[!,:time_no_as] = df_no_as[!,:time]
        df_no_as[!,:termination_no_as] = termination_no_as
        lowerBounds = df_no_as[!, :solution] - df_no_as[!, :dual_gap]
        df_no_as[!, :rel_gap_no_as] = (df_no_as[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_as[!, :solution]))
        df_no_as = select(df_no_as, [:termination_no_as, :time_no_as, :rel_gap_no_as, :seed, :dimension, :M, :var_A])

        df = innerjoin(df, df_no_as, on = [:seed, :dimension, :M, :var_A])

        # load without ss
        df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_ss_tailed_cardinality_sparse_log_reg.csv")))
        df_no_ss.termination .= replace.(df_no_ss.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ss = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ss[!,:termination]]

        df_no_ss[!,:time_no_ss] = df_no_ss[!,:time]
        df_no_ss[!,:termination_no_ss] = termination_no_ss
        lowerBounds = df_no_ss[!, :solution] - df_no_ss[!, :dual_gap]
        df_no_ss[!, :rel_gap_no_ss] = (df_no_ss[!, :solution] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_no_ss[!, :solution]))
        df_no_ss = select(df_no_ss, [:termination_no_ss, :time_no_ss, :rel_gap_no_ss, :seed, :dimension, :M, :var_A])

        df = innerjoin(df, df_no_ss, on = [:seed, :dimension, :M, :var_A])

        # load scip oa
        df_scip = DataFrame(CSV.File(joinpath(@__DIR__, "csv/scip_oa_tailed_cardinality_sparse_log_reg.csv")))

        termination_scip = [row == "TIME_LIMIT" ? 0 : 1 for row in df_scip[!,:termination]]

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
        df_scip = select(df_scip, [:termination_scip, :solution_scip, :time_scip, :seed, :dimension, :M, :var_A])

        # sort!(df, [:dimension, :k, :Ns, :p])
        # print(first(df,20))
        # sort!(df_scip, [:dimension, :k, :Ns, :p])
        # print(first(df_scip,20))
        df = innerjoin(df, df_scip, on = [:seed, :dimension, :M, :var_A])

        # check if solution optimal
        optimal_scip = []
        optimal_boscia = []
        for row in eachrow(df)
            if isapprox(row.lb_boscia, min(row.lb_boscia, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_boscia, 1)
            else 
                append!(optimal_boscia, 0)
            end
            if isapprox(row.solution_scip, min(row.lb_boscia, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_scip, 1)
            else 
                append!(optimal_scip, 0)
            end
        end
        df[!,:optimal_scip] = optimal_scip
        df[!,:optimal_boscia] = optimal_boscia

        rename!(df, :var_A => :varA)

        # compute lower bound 
        rel_gap_scip = []
        for row in eachrow(df)
            if min(abs(row.solution_scip), abs(row.lb_boscia)) == 0
                push!(rel_gap_scip, row.solution_scip - row.lb_boscia)
            elseif sign(row.lb_boscia) != sign(row.solution_scip)
                push!(rel_gap_scip, Inf)
            else
                push!(rel_gap_scip, (row.solution_scip - row.lb_boscia)/min(abs(row.solution_scip), abs(row.lb_boscia)))
            end
        end
        df[!, :rel_gap_scip] = round.(rel_gap_scip,digits=3)

        # save csv 
        file_name = joinpath(@__DIR__, "csv/tailed_cardinality_sparse_log_reg_non_grouped.csv")
        CSV.write(file_name, df, append=false)

    elseif mode == "22433"
        # load boscia 
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_mip_lib_22433.csv")))
        # delete duplicates
        df_bs = unique(df_bs, [:seed, :num_v])

        df_bs.termination .= replace.(df_bs.termination, "Time limit reached" => "TIME_LIMIT")
        termination_boscia = [row == "TIME_LIMIT" ? 0 : 1 for row in df_bs[!,:termination]]

        df[!,:time_boscia] = df_bs[!,:time]
        df[!,:seed] = df_bs[!,:seed]
        df[!,:num_v] = df_bs[!,:num_v]

        df[!,:termination_boscia] = termination_boscia
        df[!, :solution_boscia] = df_bs[!, :solution]  
        df_bs[!, :lowerBounds] = df_bs[!, :solution] - df_bs[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_bs)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df[!, :rel_gap_boscia] = rel_gap

        # load afw
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_mip_lib_22433.csv")))
        df_afw.termination .= replace.(df_afw.termination, "Time limit reached" => "TIME_LIMIT")
        termination_afw = [row == "TIME_LIMIT" ? 0 : 1 for row in df_afw[!,:termination]]

        df_afw[!,:time_afw] = df_afw[!,:time]
        df_afw[!,:termination_afw] = termination_afw
        df_afw[!, :afw_solution] = df_afw[!, :solution]
        df_afw[!, :lowerBounds] = df_afw[!, :solution] - df_afw[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_afw)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_afw[!, :rel_gap_afw] = rel_gap
        df_afw = select(df_afw, [:termination_afw, :time_afw, :afw_solution, :rel_gap_afw, :seed, :num_v])

        df = innerjoin(df, df_afw, on = [:seed, :num_v])

        # load without as, without ss
        df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mip_lib_22433.csv")))
        df_no_ws.termination .= replace.(df_no_ws.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ws = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ws[!,:termination]]

        df_no_ws[!,:time_no_ws] = df_no_ws[!,:time]
        df_no_ws[!,:termination_no_ws] = termination_no_ws
        df_no_ws[!, :solution_no_ws] = df_no_ws[!, :solution]
        df_no_ws[!, :lowerBounds] = df_no_ws[!, :solution] - df_no_ws[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_no_ws)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_no_ws[!, :rel_gap_no_ws] = rel_gap
        df_no_ws = select(df_no_ws, [:termination_no_ws, :time_no_ws, :solution_no_ws, :rel_gap_no_ws, :seed, :num_v])

        df = innerjoin(df, df_no_ws, on = [:seed, :num_v])
        # print(first(df,5))

        # load without as
        df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mip_lib_22433.csv")))
        df_no_as.termination .= replace.(df_no_as.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_as = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_as[!,:termination]]

        df_no_as[!,:time_no_as] = df_no_as[!,:time]
        df_no_as[!,:termination_no_as] = termination_no_as
        df_no_as[!, :solution_no_as] = df_no_as[!, :solution]
        df_no_as[!, :lowerBounds] = df_no_as[!, :solution] - df_no_as[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_no_as)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_no_as[!, :rel_gap_no_as] = rel_gap
        df_no_as = select(df_no_as, [:termination_no_as, :time_no_as, :solution_no_as, :rel_gap_no_as, :seed, :num_v])

        df = innerjoin(df, df_no_as, on = [:seed, :num_v])

        # load without ss
        df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mip_lib_22433.csv")))
        df_no_ss.termination .= replace.(df_no_ss.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ss = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ss[!,:termination]]

        df_no_ss[!,:time_no_ss] = df_no_ss[!,:time]
        df_no_ss[!,:termination_no_ss] = termination_no_ss
        df_no_ss[!, :solution_no_ss] = df_no_ss[!, :solution]
        df_no_ss[!, :lowerBounds] = df_no_ss[!, :solution] - df_no_ss[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_no_ss)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_no_ss[!, :rel_gap_no_ss] = rel_gap
        df_no_ss = select(df_no_ss, [:termination_no_ss, :time_no_ss, :solution_no_ss, :rel_gap_no_ss, :seed, :num_v])

        df = innerjoin(df, df_no_ss, on = [:seed, :num_v])

        # load ipopt 
        df_ipopt = DataFrame(CSV.File(joinpath(@__DIR__, "csv/ipopt_22433.csv")))
        df_ipopt.termination .= replace.(df_ipopt.termination, "Time limit reached" => "TIME_LIMIT")
        termination_ipopt = [row == "TIME_LIMIT" ? 0 : 1 for row in df_ipopt[!, :termination]]

        df_ipopt[!, :time_ipopt] = df_ipopt[!, :time]/1000
        df_ipopt[!, :termination_ipopt] = termination_ipopt
        df_ipopt[!, :solution_ipopt] = df_ipopt[!, :solution]
        df_ipopt[!, :lowerBounds] = df_bs[!, :lowerBounds]
        rel_gap = []
        for row in eachrow(df_ipopt)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, (row.solution-row.lowerBounds)/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_ipopt[!, :rel_gap_ipopt] = rel_gap

        df_ipopt = select(df_ipopt, [:termination_ipopt, :time_ipopt, :solution_ipopt, :seed, :num_v])

        df = innerjoin(df, df_ipopt, on = [:seed, :num_v])
        df[!, :rel_gap_ipopt] = (df[!, :solution_ipopt] - df[!,:lb_boscia]) ./ min.(abs.(df[!,:lb_boscia]), abs.(df[!, :solution_ipopt]))

        # load scip oa
        df_scip = DataFrame(CSV.File(joinpath(@__DIR__, "csv/scip_oa_mip_lib_22433.csv")))
        termination_scip = [row == "OPTIMAL" ? 1 : 0 for row in df_scip[!,:termination]]

        df_scip[!,:time_scip] = df_scip[!,:time]
        df_scip[!,:termination_scip] = termination_scip
        df_scip[!,:solution_scip] = df_scip[!,:solution]
        df_scip[!, :lowerBounds] = df_bs[!, :lowerBounds]
        rel_gap = []
        for row in eachrow(df_scip)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, (row.solution-row.lowerBounds)/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_scip[!, :rel_gap_scip] = rel_gap
        df_scip = select(df_scip, [:solution_scip, :termination_scip, :time_scip, :rel_gap_scip, :seed, :num_v])

        df = innerjoin(df, df_scip, on = [:seed, :num_v])
        df[!, :rel_gap_scip] = (df[!, :solution_scip] - df[!,:lb_boscia]) ./ min.(abs.(df[!,:lb_boscia]), abs.(df[!, :solution_scip]))

        # check if solution optimal
        optimal_scip = []
        optimal_ipopt = []
        optimal_boscia = []
        for row in eachrow(df)
            if isapprox(row.lb_boscia, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_boscia, 1)
            else 
                append!(optimal_boscia, 0)
            end
            if isapprox(row.solution_ipopt, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_ipopt, 1)
            else 
                append!(optimal_ipopt, 0)
            end
            if isapprox(row.solution_scip, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_scip, 1)
            else 
                append!(optimal_scip, 0)
            end
        end
        df[!,:optimal_scip] = optimal_scip
        df[!,:optimal_ipopt] = optimal_ipopt
        df[!,:optimal_boscia] = optimal_boscia

        # save csv 
        file_name = joinpath(@__DIR__, "csv/mip_lib_22433_non_grouped.csv")
        CSV.write(file_name, df, append=false)
    elseif mode == "neos5"
        # load boscia 
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_mip_lib_neos5.csv")))
        # delete duplicates
        df_bs = unique(df_bs, [:seed, :num_v])

        df_bs.termination .= replace.(df_bs.termination, "Time limit reached" => "TIME_LIMIT")
        termination_boscia = [row == "TIME_LIMIT" ? 0 : 1 for row in df_bs[!,:termination]]

        df[!,:time_boscia] = df_bs[!,:time]
        df[!,:seed] = df_bs[!,:seed]
        df[!,:num_v] = df_bs[!,:num_v]

        df[!,:termination_boscia] = termination_boscia
        df[!, :solution_boscia] = df_bs[!, :solution]  
        df_bs[!, :lowerBounds] = df_bs[!, :solution] - df_bs[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_bs)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df[!, :rel_gap_boscia] = rel_gap

        # load afw
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_mip_lib_neos5.csv")))
        df_afw.termination .= replace.(df_afw.termination, "Time limit reached" => "TIME_LIMIT")
        termination_afw = [row == "TIME_LIMIT" ? 0 : 1 for row in df_afw[!,:termination]]

        df_afw[!,:time_afw] = df_afw[!,:time]
        df_afw[!,:termination_afw] = termination_afw
        df_afw[!, :afw_solution] = df_afw[!, :solution]
        df_afw[!, :lowerBounds] = df_afw[!, :solution] - df_afw[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_afw)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_afw[!, :rel_gap_afw] = rel_gap
        df_afw = select(df_afw, [:termination_afw, :time_afw, :afw_solution, :rel_gap_afw, :seed, :num_v])

        df = innerjoin(df, df_afw, on = [:seed, :num_v])

        # load without as, without ss
        df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mip_lib_neos5.csv")))
        df_no_ws.termination .= replace.(df_no_ws.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ws = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ws[!,:termination]]

        df_no_ws[!,:time_no_ws] = df_no_ws[!,:time]
        df_no_ws[!,:termination_no_ws] = termination_no_ws
        df_no_ws[!, :solution_no_ws] = df_no_ws[!, :solution]
        df_no_ws[!, :lowerBounds] = df_no_ws[!, :solution] - df_no_ws[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_no_ws)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_no_ws[!, :rel_gap_no_ws] = rel_gap
        df_no_ws = select(df_no_ws, [:termination_no_ws, :time_no_ws, :solution_no_ws, :rel_gap_no_ws, :seed, :num_v])

        df = innerjoin(df, df_no_ws, on = [:seed, :num_v])
        # print(first(df,5))

        # load without as
        df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mip_lib_neos5.csv")))
        df_no_as.termination .= replace.(df_no_as.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_as = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_as[!,:termination]]

        df_no_as[!,:time_no_as] = df_no_as[!,:time]
        df_no_as[!,:termination_no_as] = termination_no_as
        df_no_as[!, :solution_no_as] = df_no_as[!, :solution]
        df_no_as[!, :lowerBounds] = df_no_as[!, :solution] - df_no_as[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_no_as)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_no_as[!, :rel_gap_no_as] = rel_gap
        df_no_as = select(df_no_as, [:termination_no_as, :time_no_as, :solution_no_as, :rel_gap_no_as, :seed, :num_v])

        df = innerjoin(df, df_no_as, on = [:seed, :num_v])

        # load without ss
        df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mip_lib_neos5.csv")))
        df_no_ss.termination .= replace.(df_no_ss.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ss = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ss[!,:termination]]

        df_no_ss[!,:time_no_ss] = df_no_ss[!,:time]
        df_no_ss[!,:termination_no_ss] = termination_no_ss
        df_no_ss[!, :solution_no_ss] = df_no_ss[!, :solution]
        df_no_ss[!, :lowerBounds] = df_no_ss[!, :solution] - df_no_ss[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_no_ss)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_no_ss[!, :rel_gap_no_ss] = rel_gap
        df_no_ss = select(df_no_ss, [:termination_no_ss, :time_no_ss, :solution_no_ss, :rel_gap_no_ss, :seed, :num_v])

        df = innerjoin(df, df_no_ss, on = [:seed, :num_v])

        # load ipopt 
        df_ipopt = DataFrame(CSV.File(joinpath(@__DIR__, "csv/ipopt_neos5.csv")))
        df_ipopt.termination .= replace.(df_ipopt.termination, "Time limit reached" => "TIME_LIMIT")
        termination_ipopt = [row == "TIME_LIMIT" ? 0 : 1 for row in df_ipopt[!, :termination]]

        df_ipopt[!, :time_ipopt] = df_ipopt[!, :time]/1000
        df_ipopt[!, :termination_ipopt] = termination_ipopt
        df_ipopt[!, :solution_ipopt] = df_ipopt[!, :solution]
        df_ipopt[!, :lowerBounds] = df_bs[!, :lowerBounds]
        rel_gap = []
        for row in eachrow(df_ipopt)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, (row.solution-row.lowerBounds)/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_ipopt[!, :rel_gap_ipopt] = rel_gap

        df_ipopt = select(df_ipopt, [:termination_ipopt, :time_ipopt, :solution_ipopt, :seed, :num_v])

        df = innerjoin(df, df_ipopt, on = [:seed, :num_v])
        df[!, :rel_gap_ipopt] = (df[!, :solution_ipopt] - df[!,:lb_boscia]) ./ min.(abs.(df[!,:lb_boscia]), abs.(df[!, :solution_ipopt]))

        # load scip oa
        df_scip = DataFrame(CSV.File(joinpath(@__DIR__, "csv/scip_oa_mip_lib_neos5.csv")))
        termination_scip = [row == "OPTIMAL" ? 1 : 0 for row in df_scip[!,:termination]]

        df_scip[!,:time_scip] = df_scip[!,:time]
        df_scip[!,:termination_scip] = termination_scip
        df_scip[!,:solution_scip] = df_scip[!,:solution]
        df_scip[!, :lowerBounds] = df_bs[!, :lowerBounds] 
        rel_gap = []
        for row in eachrow(df_scip)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, (row.solution-row.lowerBounds)/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_scip[!, :rel_gap_scip] = rel_gap
        df_scip = select(df_scip, [:solution_scip, :termination_scip, :time_scip, :rel_gap_scip, :seed, :num_v])

        df = innerjoin(df, df_scip, on = [:seed, :num_v])
        df[!, :rel_gap_scip] = (df[!, :solution_scip] - df[!,:lb_boscia]) ./ min.(abs.(df[!,:lb_boscia]), abs.(df[!, :solution_scip]))

        # load strong convexity 
        df_sc = DataFrame(CSV.File(joinpath(@__DIR__, "csv/strong_convexity_mip_lib_neos5.csv")))
        df_sc.termination .= replace.(df_sc.termination, "Time limit reached" => "TIME_LIMIT")
        termination_sc = [row == "TIME_LIMIT" ? 0 : 1 for row in df_sc[!,:termination]]

        df_sc[!,:time_sc] = df_sc[!,:time]
        df_sc[!,:seed] = df_sc[!,:seed]
        df_sc[!,:num_v] = df_sc[!,:num_v]

        df_sc[!,:solution_sc] = df_sc[!,:solution]
        df_sc[!,:termination_sc] = termination_sc
        lowerBounds = df_sc[!, :solution] - df_sc[!, :dual_gap]
        df_sc[!,:lb_sc] = lowerBounds
        df = innerjoin(df, df_sc, on = [:seed, :num_v])

        df[!, :rel_gap_sc] = (df[!, :solution_sc] - df[!,:lb_boscia]) ./ min.(abs.(df[!,:lb_boscia]), abs.(df[!, :solution_sc]))
        df[!, :rel_gap_boscia_sc] = (df[!, :solution_boscia] - df[!,:lb_sc]) ./ min.(abs.(df[!,:lb_sc]), abs.(df[!, :solution_boscia])) # used to compare strong convexity

        # check if solution optimal
        optimal_scip = []
        optimal_ipopt = []
        optimal_boscia = []
        for row in eachrow(df)
            if isapprox(row.lb_boscia, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_boscia, 1)
            else 
                append!(optimal_boscia, 0)
            end
            if isapprox(row.solution_ipopt, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_ipopt, 1)
            else 
                append!(optimal_ipopt, 0)
            end
            if isapprox(row.solution_scip, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_scip, 1)
            else 
                append!(optimal_scip, 0)
            end
        end
        df[!,:optimal_scip] = optimal_scip
        df[!,:optimal_ipopt] = optimal_ipopt
        df[!,:optimal_boscia] = optimal_boscia

        print(select(df,[:solution_boscia, :time_boscia, :solution_sc, :time_sc]))
        # save csv 
        file_name = joinpath(@__DIR__, "csv/mip_lib_neos5_non_grouped.csv")
        CSV.write(file_name, df, append=false)
    elseif mode == "pg5_34"
        # load boscia 
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_mip_lib_pg5_34.csv")))
        # delete duplicates
        df_bs = unique(df_bs, [:seed, :num_v])

        df_bs.termination .= replace.(df_bs.termination, "Time limit reached" => "TIME_LIMIT")
        termination_boscia = [row == "TIME_LIMIT" ? 0 : 1 for row in df_bs[!,:termination]]

        df[!,:time_boscia] = df_bs[!,:time]
        df[!,:seed] = df_bs[!,:seed]
        df[!,:num_v] = df_bs[!,:num_v]

        df[!,:termination_boscia] = termination_boscia
        df[!, :solution_boscia] = df_bs[!, :solution]  
        df_bs[!, :lowerBounds] = df_bs[!, :solution] - df_bs[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_bs)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df[!, :rel_gap_boscia] = rel_gap

        # load afw
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_mip_lib_pg5_34.csv")))
        df_afw.termination .= replace.(df_afw.termination, "Time limit reached" => "TIME_LIMIT")
        termination_afw = [row == "TIME_LIMIT" ? 0 : 1 for row in df_afw[!,:termination]]

        df_afw[!,:time_afw] = df_afw[!,:time]
        df_afw[!,:termination_afw] = termination_afw
        df_afw[!, :afw_solution] = df_afw[!, :solution]
        df_afw[!, :lowerBounds] = df_afw[!, :solution] - df_afw[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_afw)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_afw[!, :rel_gap_afw] = rel_gap
        df_afw = select(df_afw, [:termination_afw, :time_afw, :afw_solution, :rel_gap_afw, :seed, :num_v])

        df = innerjoin(df, df_afw, on = [:seed, :num_v])

        # load without as, without ss
        df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mip_lib_pg5_34.csv")))
        df_no_ws.termination .= replace.(df_no_ws.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ws = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ws[!,:termination]]

        df_no_ws[!,:time_no_ws] = df_no_ws[!,:time]
        df_no_ws[!,:termination_no_ws] = termination_no_ws
        df_no_ws[!, :solution_no_ws] = df_no_ws[!, :solution]
        df_no_ws[!, :lowerBounds] = df_no_ws[!, :solution] - df_no_ws[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_no_ws)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_no_ws[!, :rel_gap_no_ws] = rel_gap
        df_no_ws = select(df_no_ws, [:termination_no_ws, :time_no_ws, :solution_no_ws, :rel_gap_no_ws, :seed, :num_v])

        df = innerjoin(df, df_no_ws, on = [:seed, :num_v])
        # print(first(df,5))

        # load without as
        df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mip_lib_pg5_34.csv")))
        df_no_as.termination .= replace.(df_no_as.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_as = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_as[!,:termination]]

        df_no_as[!,:time_no_as] = df_no_as[!,:time]
        df_no_as[!,:termination_no_as] = termination_no_as
        df_no_as[!, :solution_no_as] = df_no_as[!, :solution]
        df_no_as[!, :lowerBounds] = df_no_as[!, :solution] - df_no_as[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_no_as)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_no_as[!, :rel_gap_no_as] = rel_gap
        df_no_as = select(df_no_as, [:termination_no_as, :time_no_as, :solution_no_as, :rel_gap_no_as, :seed, :num_v])

        df = innerjoin(df, df_no_as, on = [:seed, :num_v])

        # load without ss
        df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mip_lib_pg5_34.csv")))
        df_no_ss.termination .= replace.(df_no_ss.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ss = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ss[!,:termination]]

        df_no_ss[!,:time_no_ss] = df_no_ss[!,:time]
        df_no_ss[!,:termination_no_ss] = termination_no_ss
        df_no_ss[!, :solution_no_ss] = df_no_ss[!, :solution]
        df_no_ss[!, :lowerBounds] = df_no_ss[!, :solution] - df_no_ss[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_no_ss)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_no_ss[!, :rel_gap_no_ss] = rel_gap
        df_no_ss = select(df_no_ss, [:termination_no_ss, :time_no_ss, :solution_no_ss, :rel_gap_no_ss, :seed, :num_v])

        df = innerjoin(df, df_no_ss, on = [:seed, :num_v])

        # load ipopt 
        df_ipopt = DataFrame(CSV.File(joinpath(@__DIR__, "csv/ipopt_pg5_34.csv")))
        df_ipopt.termination .= replace.(df_ipopt.termination, "Time limit reached" => "TIME_LIMIT")
        termination_ipopt = [row == "TIME_LIMIT" ? 0 : 1 for row in df_ipopt[!, :termination]]

        df_ipopt[!, :time_ipopt] = df_ipopt[!, :time]/1000
        df_ipopt[!, :termination_ipopt] = termination_ipopt
        df_ipopt[!, :solution_ipopt] = df_ipopt[!, :solution]
        df_ipopt[!, :lowerBounds] = df_bs[!, :lowerBounds] 
        rel_gap = []
        for row in eachrow(df_ipopt)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, (row.solution-row.lowerBounds)/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_ipopt[!, :rel_gap_ipopt] = rel_gap

        df_ipopt = select(df_ipopt, [:termination_ipopt, :time_ipopt, :solution_ipopt, :seed, :num_v])

        df = innerjoin(df, df_ipopt, on = [:seed, :num_v])
        df[!, :rel_gap_ipopt] = (df[!, :solution_ipopt] - df[!,:lb_boscia]) ./ min.(abs.(df[!,:lb_boscia]), abs.(df[!, :solution_ipopt]))

        # load scip oa
        df_scip = DataFrame(CSV.File(joinpath(@__DIR__, "csv/scip_oa_mip_lib_pg5_34.csv")))
        termination_scip = [row == "OPTIMAL" ? 1 : 0 for row in df_scip[!,:termination]]

        df_scip[!,:time_scip] = df_scip[!,:time]
        df_scip[!,:termination_scip] = termination_scip
        df_scip[!,:solution_scip] = df_scip[!,:solution]
        df_scip[!, :lowerBounds] = df_bs[!, :lowerBounds] 
        rel_gap = []
        for row in eachrow(df_scip)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, (row.solution-row.lowerBounds)/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_scip[!, :rel_gap_scip] = rel_gap
        df_scip = select(df_scip, [:solution_scip, :termination_scip, :time_scip, :rel_gap_scip, :seed, :num_v])

        df = innerjoin(df, df_scip, on = [:seed, :num_v])
        df[!, :rel_gap_scip] = (df[!, :solution_scip] - df[!,:lb_boscia]) ./ min.(abs.(df[!,:lb_boscia]), abs.(df[!, :solution_scip]))

        # load strong convexity 
        df_sc = DataFrame(CSV.File(joinpath(@__DIR__, "csv/strong_convexity_mip_lib_pg5_34.csv")))
        df_sc.termination .= replace.(df_sc.termination, "Time limit reached" => "TIME_LIMIT")
        termination_sc = [row == "TIME_LIMIT" ? 0 : 1 for row in df_sc[!,:termination]]

        df_sc[!,:time_sc] = df_sc[!,:time]
        df_sc[!,:seed] = df_sc[!,:seed]
        df_sc[!,:num_v] = df_sc[!,:num_v]

        df_sc[!,:termination_sc] = termination_sc
        df = innerjoin(df, df_sc, on = [:seed, :num_v])

        # check if solution optimal
        optimal_scip = []
        optimal_ipopt = []
        optimal_boscia = []
        for row in eachrow(df)
            if isapprox(row.lb_boscia, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_boscia, 1)
            else 
                append!(optimal_boscia, 0)
            end
            if isapprox(row.solution_ipopt, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_ipopt, 1)
            else 
                append!(optimal_ipopt, 0)
            end
            if isapprox(row.solution_scip, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_scip, 1)
            else 
                append!(optimal_scip, 0)
            end
        end
        df[!,:optimal_scip] = optimal_scip
        df[!,:optimal_ipopt] = optimal_ipopt
        df[!,:optimal_boscia] = optimal_boscia
        # save csv 
        file_name = joinpath(@__DIR__, "csv/mip_lib_pg5_34_non_grouped.csv")
        CSV.write(file_name, df, append=false)
    elseif mode == "ran14x18"
        # load boscia 
        df_bs = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_mip_lib_ran14x18-disj-8.csv")))
        # delete duplicates
        df_bs = unique(df_bs, [:seed, :num_v])

        df_bs.termination .= replace.(df_bs.termination, "Time limit reached" => "TIME_LIMIT")
        termination_boscia = [row == "TIME_LIMIT" ? 0 : 1 for row in df_bs[!,:termination]]

        df[!,:time_boscia] = df_bs[!,:time]
        df[!,:seed] = df_bs[!,:seed]
        df[!,:num_v] = df_bs[!,:num_v]

        df[!,:termination_boscia] = termination_boscia
        df[!, :solution_boscia] = df_bs[!, :solution]  
        df_bs[!, :lowerBounds] = df_bs[!, :solution] - df_bs[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_bs)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df[!, :rel_gap_boscia] = rel_gap
        
        # load afw
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_mip_lib_ran14x18-disj-8.csv")))
        df_afw.termination .= replace.(df_afw.termination, "Time limit reached" => "TIME_LIMIT")
        termination_afw = [row == "TIME_LIMIT" ? 0 : 1 for row in df_afw[!,:termination]]

        df_afw[!,:time_afw] = df_afw[!,:time]
        df_afw[!,:termination_afw] = termination_afw
        df_afw[!, :afw_solution] = df_afw[!, :solution]
        df_afw[!, :lowerBounds] = df_afw[!, :solution] - df_afw[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_afw)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_afw[!, :rel_gap_afw] = rel_gap
        df_afw = select(df_afw, [:termination_afw, :time_afw, :afw_solution, :rel_gap_afw, :seed, :num_v])

        df = innerjoin(df, df_afw, on = [:seed, :num_v])

        # load without as, without ss
        df_no_ws = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mip_lib_ran14x18-disj-8.csv")))
        df_no_ws.termination .= replace.(df_no_ws.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ws = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ws[!,:termination]]

        df_no_ws[!,:time_no_ws] = df_no_ws[!,:time]
        df_no_ws[!,:termination_no_ws] = termination_no_ws
        df_no_ws[!, :solution_no_ws] = df_no_ws[!, :solution]
        df_no_ws[!, :lowerBounds] = df_no_ws[!, :solution] - df_no_ws[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_no_ws)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_no_ws[!, :rel_gap_no_ws] = rel_gap
        df_no_ws = select(df_no_ws, [:termination_no_ws, :time_no_ws, :solution_no_ws, :rel_gap_no_ws, :seed, :num_v])

        df = innerjoin(df, df_no_ws, on = [:seed, :num_v])
        # print(first(df,5))

        # load without as
        df_no_as = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mip_lib_ran14x18-disj-8.csv")))
        df_no_as.termination .= replace.(df_no_as.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_as = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_as[!,:termination]]

        df_no_as[!,:time_no_as] = df_no_as[!,:time]
        df_no_as[!,:termination_no_as] = termination_no_as
        df_no_as[!, :solution_no_as] = df_no_as[!, :solution]
        df_no_as[!, :lowerBounds] = df_no_as[!, :solution] - df_no_as[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_no_as)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_no_as[!, :rel_gap_no_as] = rel_gap
        df_no_as = select(df_no_as, [:termination_no_as, :time_no_as, :solution_no_as, :rel_gap_no_as, :seed, :num_v])

        df = innerjoin(df, df_no_as, on = [:seed, :num_v])

        # load without ss
        df_no_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mip_lib_ran14x18-disj-8.csv")))
        df_no_ss.termination .= replace.(df_no_ss.termination, "Time limit reached" => "TIME_LIMIT")
        termination_no_ss = [row == "TIME_LIMIT" ? 0 : 1 for row in df_no_ss[!,:termination]]

        df_no_ss[!,:time_no_ss] = df_no_ss[!,:time]
        df_no_ss[!,:termination_no_ss] = termination_no_ss
        df_no_ss[!, :solution_no_ss] = df_no_ss[!, :solution]
        df_no_ss[!, :lowerBounds] = df_no_ss[!, :solution] - df_no_ss[!, :dual_gap]
        rel_gap = []
        for row in eachrow(df_no_ss)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, row.dual_gap/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_no_ss[!, :rel_gap_no_ss] = rel_gap
        df_no_ss = select(df_no_ss, [:termination_no_ss, :time_no_ss, :solution_no_ss, :rel_gap_no_ss, :seed, :num_v])

        df = innerjoin(df, df_no_ss, on = [:seed, :num_v]) 

        # load ipopt 
        df_ipopt = DataFrame(CSV.File(joinpath(@__DIR__, "csv/ipopt_ran14x18-disj-8.csv")))
        df_ipopt.termination .= replace.(df_ipopt.termination, "Time limit reached" => "TIME_LIMIT")
        termination_ipopt = [row == "TIME_LIMIT" ? 0 : 1 for row in df_ipopt[!, :termination]]

        df_ipopt[!, :time_ipopt] = df_ipopt[!, :time]/1000
        df_ipopt[!, :termination_ipopt] = termination_ipopt
        df_ipopt[!, :solution_ipopt] = df_ipopt[!, :solution]
        df_ipopt[!, :lowerBounds] = df_bs[!, :lowerBounds] 
        rel_gap = []
        for row in eachrow(df_ipopt)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, (row.solution-row.lowerBounds)/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_ipopt[!, :rel_gap_ipopt] = rel_gap

        df_ipopt = select(df_ipopt, [:termination_ipopt, :time_ipopt, :solution_ipopt, :seed, :num_v])

        df = innerjoin(df, df_ipopt, on = [:seed, :num_v])
        df[!, :rel_gap_ipopt] = (df[!, :solution_ipopt] - df[!,:lb_boscia]) ./ min.(abs.(df[!,:lb_boscia]), abs.(df[!, :solution_ipopt]))

        # load scip oa
        df_scip = DataFrame(CSV.File(joinpath(@__DIR__, "csv/scip_oa_mip_lib_ran14x18-disj-8.csv")))
        termination_scip = [row == "OPTIMAL" ? 1 : 0 for row in df_scip[!,:termination]]

        df_scip[!,:time_scip] = df_scip[!,:time]
        df_scip[!,:termination_scip] = termination_scip
        df_scip[!,:solution_scip] = df_scip[!,:solution]
        df_scip[!, :lowerBounds] = df_bs[!, :lowerBounds]
        rel_gap = []
        for row in eachrow(df_scip)
            if min(abs(row.solution), abs(row.lowerBounds)) == 0.0
                push!(rel_gap, row.dual_gap)
            elseif sign(row.lowerBounds) != sign(row.solution)
                push!(rel_gap, Inf)
            else
                push!(rel_gap, (row.solution-row.lowerBounds)/min(abs(row.solution), abs(row.lowerBounds)))
            end
        end
        df_scip[!, :rel_gap_scip] = rel_gap
        df_scip = select(df_scip, [:solution_scip, :termination_scip, :time_scip, :rel_gap_scip, :seed, :num_v])

        df = innerjoin(df, df_scip, on = [:seed, :num_v])
        df[!, :rel_gap_scip] = (df[!, :solution_scip] - df[!,:lb_boscia]) ./ min.(abs.(df[!,:lb_boscia]), abs.(df[!, :solution_scip]))

        # load strong convexity 
        df_sc = DataFrame(CSV.File(joinpath(@__DIR__, "csv/strong_convexity_mip_lib_ran14x18-disj-8.csv")))
        df_sc.termination .= replace.(df_sc.termination, "Time limit reached" => "TIME_LIMIT")
        termination_sc = [row == "TIME_LIMIT" ? 0 : 1 for row in df_sc[!,:termination]]

        df_sc[!,:time_sc] = df_sc[!,:time]
        df_sc[!,:seed] = df_sc[!,:seed]
        df_sc[!,:num_v] = df_sc[!,:num_v]

        df_sc[!,:termination_sc] = termination_sc
        df_sc[!,:solution_sc] = df_sc[!,:solution]
        lowerBounds = df_sc[!, :solution] - df_sc[!, :dual_gap]
        df_sc[!,:lb_sc] = lowerBounds
        df_sc[!, :rel_gap_sc] = (df_sc[!, :solution_sc] - lowerBounds) ./ min.(abs.(lowerBounds), abs.(df_sc[!, :solution_sc]))
        select!(df_sc, [:solution_sc, :termination_sc, :time_sc, :rel_gap_sc, :seed, :num_v, :lb_sc])

        df = innerjoin(df, df_sc, on = [:seed, :num_v])

        df[!, :rel_gap_boscia_sc] = (df[!, :solution_boscia] - df[!,:lb_sc]) ./ min.(abs.(df[!,:lb_sc]), abs.(df[!, :solution_boscia])) # used to compare strong convexity

        # check if solution optimal
        optimal_scip = []
        optimal_ipopt = []
        optimal_boscia = []
        for row in eachrow(df)
            if isapprox(row.lb_boscia, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_boscia, 1)
            else 
                append!(optimal_boscia, 0)
            end
            if isapprox(row.solution_ipopt, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_ipopt, 1)
            else 
                append!(optimal_ipopt, 0)
            end
            if isapprox(row.solution_scip, min(row.lb_boscia, row.solution_ipopt, row.solution_scip), atol=1e-4, rtol=1e-2) 
                append!(optimal_scip, 1)
            else 
                append!(optimal_scip, 0)
            end
        end
        df[!,:optimal_scip] = optimal_scip
        df[!,:optimal_ipopt] = optimal_ipopt
        df[!,:optimal_boscia] = optimal_boscia

        print(select(df,[:solution_boscia, :time_boscia, :rel_gap_boscia, :lb_boscia, :rel_gap_boscia_sc, :solution_sc, :time_sc, :rel_gap_sc, :lb_sc]))

        # save csv 
        file_name = joinpath(@__DIR__, "csv/mip_lib_ran14x18-disj-8_non_grouped.csv")
        CSV.write(file_name, df, append=false)
    end
    return file_name
end

function build_grouped_csv(file_name, mode)
    df = CSV.read(file_name, DataFrame)

    # set time below 1800
    df[df.time_boscia.>1800, :time_boscia] .= 1800
    df[df.time_scip.>1800, :time_scip] .= 1800

    df[df.time_afw.>1800, :time_afw] .= 1800
    df[df.time_no_as.>1800, :time_no_as] .= 1800
    df[df.time_no_ss.>1800, :time_no_ss] .= 1800
    df[df.time_no_ws.>1800, :time_no_ws] .= 1800

    if mode == "sparse_reg" || mode == "sparse_log_reg" || mode == "tailed_cardinality"
        df[df.time_loc_ti.>1800, :time_loc_ti] .= 1800
        df[df.time_gl_ti.>1800, :time_gl_ti] .= 1800
        df[df.time_no_ti.>1800, :time_no_ti] .= 1800
    end

    if mode == "neos5" || mode == "ran14x18"
        df[df.time_sc.>1800, :time_sc] .= 1800
    end

    if mode != "tailed_cardinality" && mode != "tailed_cardinality_sparse_log_reg"
        df[df.time_ipopt.>1800, :time_ipopt] .= 1800
    end

    function geo_mean(group)
        prod = 1.0
        n = length(group)
        count = 0
        for element in group
            # @show element
            if element != Inf && element != -Inf && element != NaN  #&& element != missing 
                prod = prod * element
                count += 1
            end
        end
        # @show prod, n
        return prod^(1/count)
    end

    function custom_mean(group)
        sum = 0.0
        n = length(group)
        count = 0
        for element in group
            # @show element
            if element != Inf && element != -Inf && element != NaN  #&& element != missing
                sum = sum + element
                count += 1
            end
        end
        # @show prod, n
        return count > 0 ? (sum/count) : Inf
    end

    function custom_sum(group)
        sum = 0.0
        count = 0
        for element in group
            # @show element
            if element != Inf && element != -Inf && element != NaN # && element != missing
                sum = sum + element
                count += 1
            end
        end
        # @show prod, n
        return count > 0 ? sum : Inf
    end

    # group by dimension
    if mode != "poisson" && mode != "sparse_reg" && mode != "sparse_log_reg" && mode != "tailed_cardinality" && mode != "tailed_cardinality_sparse_log_reg" && mode != "22433" && mode != "neos5" && mode != "pg5_34" && mode != "ran14x18" 
        gdf = combine(
            groupby(df, :dimension), 
            :time_boscia => geo_mean, :termination_boscia => custom_sum,
            :rel_gap_boscia => custom_mean,
            :time_scip => geo_mean, :termination_scip => custom_sum,
            :rel_gap_scip => custom_mean,
            :time_ipopt => geo_mean, :termination_ipopt => custom_sum,
            :rel_gap_ipopt => custom_mean,
            :time_no_ws => geo_mean, :termination_no_ws => custom_sum,
            :rel_gap_no_ws => custom_mean,
            :time_no_as => geo_mean, :termination_no_as => custom_sum,
            :rel_gap_no_as => custom_mean,
            :time_no_ss => geo_mean, :termination_no_ss => custom_sum,
            :rel_gap_no_ss => custom_mean,
            :time_afw => geo_mean, :termination_afw => custom_sum,
            :rel_gap_afw => custom_mean,
            :optimal_boscia => custom_sum, :optimal_ipopt => custom_sum,
            :optimal_scip => custom_sum,
            nrow => :NumInstances, renamecols=false
            )
    elseif mode == "poisson"
        gdf = combine(
            groupby(df, [:dimension, :k, :Ns]), 
            :time_boscia => geo_mean, :termination_boscia => sum,
            :rel_gap_boscia => custom_mean,
            :time_scip => geo_mean, :termination_scip => custom_sum,
            :rel_gap_scip => custom_mean,
            :time_ipopt => geo_mean, :termination_ipopt => custom_sum,
            :rel_gap_ipopt => custom_mean,
            :time_no_ws => geo_mean, :termination_no_ws => custom_sum,
            :rel_gap_no_ws => custom_mean,
            :time_no_as => geo_mean, :termination_no_as => custom_sum,
            :rel_gap_no_as => custom_mean,
            :time_no_ss => geo_mean, :termination_no_ss => custom_sum,
            :rel_gap_no_ss => custom_mean,
            :time_afw => geo_mean, :termination_afw => custom_sum,
            :rel_gap_afw => custom_mean,
            :optimal_boscia => custom_sum, :optimal_ipopt => custom_sum,
            :optimal_scip => custom_sum,
            nrow => :NumInstances, renamecols=false
            )
    elseif mode == "sparse_reg"
        gdf = combine(
            groupby(df, [:dimension, :p, :k]), 
            :time_boscia => geo_mean, :termination_boscia => sum,
            :rel_gap_boscia => custom_mean,
            :time_scip => geo_mean, :termination_scip => custom_sum,
            :rel_gap_scip => custom_mean,
            :time_scip_tol => geo_mean, :termination_scip_tol => custom_sum,
            :time_ipopt => geo_mean, :termination_ipopt => custom_sum,
            :rel_gap_ipopt => custom_mean,
            :time_no_ws => geo_mean, :termination_no_ws => custom_sum,
            :rel_gap_no_ws => custom_mean,
            :time_no_as => geo_mean, :termination_no_as => custom_sum,
            :rel_gap_no_as => custom_mean,
            :time_no_ss => geo_mean, :termination_no_ss => custom_sum,
            :rel_gap_no_ss => custom_mean,
            :time_afw => geo_mean, :termination_afw => custom_sum,
            :rel_gap_afw => custom_mean,
            :optimal_boscia => custom_sum, :optimal_ipopt => custom_sum,
            :optimal_scip => custom_sum,
            nrow => :NumInstances, renamecols=false
            )
    elseif mode == "sparse_log_reg"
        gdf = combine(
            groupby(df, [:dimension, :p, :k, :M, :varA]), 
            :time_boscia => geo_mean, :termination_boscia => custom_sum,
            :rel_gap_boscia => custom_mean,
            :time_scip => geo_mean, :termination_scip => custom_sum,
            :rel_gap_scip => custom_mean,
            :time_ipopt => geo_mean, :termination_ipopt => custom_sum,
            :rel_gap_ipopt => custom_mean,
            :time_no_ws => geo_mean, :termination_no_ws => custom_sum,
            :rel_gap_no_ws => custom_mean,
            :time_no_as => geo_mean, :termination_no_as => custom_sum,
            :rel_gap_no_as => custom_mean,
            :time_no_ss => geo_mean, :termination_no_ss => custom_sum,
            :rel_gap_no_ss => custom_mean,
            :time_afw => geo_mean, :termination_afw => custom_sum,
            :rel_gap_afw => custom_mean,
            :optimal_boscia => custom_sum, :optimal_ipopt => custom_sum,
            :optimal_scip => custom_sum,
            nrow => :NumInstances, renamecols=false
            )
    elseif mode == "tailed_cardinality"
        gdf = combine(
            groupby(df, [:n0, :m0, :M]), 
            :time_boscia => geo_mean, :termination_boscia => custom_sum,
            :rel_gap_boscia => custom_mean,
            :time_scip => geo_mean, :termination_scip => custom_sum,
            :rel_gap_scip => custom_mean,
            :time_no_ws => geo_mean, :termination_no_ws => custom_sum,
            :rel_gap_no_ws => custom_mean,
            :time_no_as => geo_mean, :termination_no_as => custom_sum,
            :rel_gap_no_as => custom_mean,
            :time_no_ss => geo_mean, :termination_no_ss => custom_sum,
            :rel_gap_no_ss => custom_mean,
            :time_afw => geo_mean, :termination_afw => custom_sum,
            :rel_gap_afw => custom_mean,
            :optimal_boscia => custom_sum,
            :optimal_scip => custom_sum,
            nrow => :NumInstances, renamecols=false
            )
    elseif mode == "tailed_cardinality_sparse_log_reg"
        gdf = combine(
            groupby(df, [:dimension, :M, :varA]), 
            :time_boscia => geo_mean, :termination_boscia => custom_sum,
            :rel_gap_boscia => custom_mean,
            :time_scip => geo_mean, :termination_scip => custom_sum,
            :rel_gap_scip => custom_mean,
            :time_no_ws => geo_mean, :termination_no_ws => custom_sum,
            :rel_gap_no_ws => custom_mean,
            :time_no_as => geo_mean, :termination_no_as => custom_sum,
            :rel_gap_no_as => custom_mean,
            :time_no_ss => geo_mean, :termination_no_ss => custom_sum,
            :rel_gap_no_ss => custom_mean,
            :time_afw => geo_mean, :termination_afw => custom_sum,
            :rel_gap_afw => custom_mean,
            :optimal_boscia => custom_sum,
            :optimal_scip => custom_sum,
            nrow => :NumInstances, renamecols=false
            )
    elseif mode == "neos5" || mode == "ran14x18"
        gdf = combine(
            groupby(df, :num_v), 
            :time_boscia => geo_mean, :termination_boscia => custom_sum,
            :rel_gap_boscia => custom_mean,
            :time_scip => geo_mean, :termination_scip => custom_sum,
            :rel_gap_scip => custom_mean,
            :time_ipopt => geo_mean, :termination_ipopt => custom_sum,
            :rel_gap_ipopt => custom_mean,
            :time_no_ws => geo_mean, :termination_no_ws => custom_sum,
            :rel_gap_no_ws => custom_mean,
            :time_no_as => geo_mean, :termination_no_as => custom_sum,
            :rel_gap_no_as => custom_mean,
            :time_no_ss => geo_mean, :termination_no_ss => custom_sum,
            :rel_gap_no_ss => custom_mean,
            :time_afw => geo_mean, :termination_afw => custom_sum,
            :rel_gap_afw => custom_mean,
            :optimal_boscia => custom_sum, :optimal_ipopt => custom_sum,
            :optimal_scip => custom_sum,
            :termination_sc => custom_sum,
            :time_sc => geo_mean,
            :rel_gap_sc => custom_mean,
            :rel_gap_boscia_sc => custom_mean,
            nrow => :NumInstances, renamecols=false
            )
    elseif mode == "22433" || mode == "pg5_34"
        gdf = combine(
            groupby(df, :num_v), 
            :time_boscia => geo_mean, :termination_boscia => custom_sum,
            :rel_gap_boscia => custom_mean,
            :time_scip => geo_mean, :termination_scip => custom_sum,
            :rel_gap_scip => custom_mean,
            :time_ipopt => geo_mean, :termination_ipopt => custom_sum,
            :rel_gap_ipopt => custom_mean,
            :time_no_ws => geo_mean, :termination_no_ws => custom_sum,
            :rel_gap_no_ws => custom_mean,
            :time_no_as => geo_mean, :termination_no_as => custom_sum,
            :rel_gap_no_as => custom_mean,
            :time_no_ss => geo_mean, :termination_no_ss => custom_sum,
            :rel_gap_no_ss => custom_mean,
            :time_afw => geo_mean, :termination_afw => custom_sum,
            :rel_gap_afw => custom_mean,
            :optimal_boscia => custom_sum, :optimal_ipopt => custom_sum,
            :optimal_scip => custom_sum,
            nrow => :NumInstances, renamecols=false
            )
    end

    # remove underscore in headers for LaTex
    rename!(gdf,
        :time_boscia => :timeBoscia, 
        :termination_boscia => :terminationBoscia,
        :rel_gap_boscia => :relGapBoscia,
        :time_scip => :timeScip, 
        :termination_scip => :terminationScip,
        :rel_gap_scip => :relGapScip,
        :optimal_boscia => :optimalBoscia,
        :optimal_scip => :optimalScip,
        :time_no_ws => :timeNoWs, 
        :termination_no_ws => :terminationNoWs,
        :rel_gap_no_ws => :relGapNoWs,
        :time_no_as => :timeNoAs, 
        :termination_no_as => :terminationNoAs,
        :rel_gap_no_as => :relGapNoAs,
        :time_no_ss => :timeNoSs, 
        :termination_no_ss => :terminationNoSs,
        :rel_gap_no_ss => :relGapNoSs,
        :time_afw => :timeAfw, 
        :termination_afw => :terminationAfw,
        :rel_gap_afw => :relGapAfw,
    )


    if mode != "tailed_cardinality" && mode != "tailed_cardinality_sparse_log_reg"
        rename!(gdf, 
        :time_ipopt => :timeIpopt,
        :termination_ipopt => :terminationIpopt,
        :rel_gap_ipopt => :relGapIpopt,
        :optimal_ipopt => :optimalIpopt,
        )
    end
    
    # parse to int
    gdf[!,:timeBoscia] = convert.(Int64,round.(gdf[!,:timeBoscia]))
    gdf[!,:timeScip] = convert.(Int64,round.(gdf[!,:timeScip]))

    non_inf_entries = findall(isfinite, gdf[!, :relGapBoscia])
    gdf[non_inf_entries, :relGapBoscia] = convert.(Int64, round.(gdf[non_inf_entries, :relGapBoscia]*100))
    non_inf_entries = findall(isfinite, gdf[!, :relGapScip])
    gdf[non_inf_entries, :relGapScip] = convert.(Int64, round.(gdf[non_inf_entries, :relGapScip]*100))
    non_inf_entries = findall(isfinite, gdf[!, :relGapAfw])
    gdf[non_inf_entries, :relGapAfw] = convert.(Int64, round.(gdf[non_inf_entries, :relGapAfw]*100))
    non_inf_entries = findall(isfinite, gdf[!, :relGapNoWs])
    gdf[non_inf_entries, :relGapNoWs] = convert.(Int64, round.(gdf[non_inf_entries, :relGapNoWs]*100))
    non_inf_entries = findall(isfinite, gdf[!, :relGapNoSs])
    gdf[non_inf_entries, :relGapNoSs] = convert.(Int64, round.(gdf[non_inf_entries, :relGapNoSs]*100))
    non_inf_entries = findall(isfinite, gdf[!, :relGapNoAs])
    gdf[non_inf_entries, :relGapNoAs] = convert.(Int64, round.(gdf[non_inf_entries, :relGapNoAs]*100))

    if mode != "tailed_cardinality" && mode != "tailed_cardinality_sparse_log_reg"
        non_inf_entries = findall(isfinite, gdf[!, :relGapIpopt])
        gdf[non_inf_entries, :relGapIpopt] = convert.(Int64, round.(gdf[non_inf_entries, :relGapIpopt]*100))
    end 

    gdf[!,:timeNoWs] = convert.(Int64,round.(gdf[!,:timeNoWs]))
    gdf[!,:timeNoAs] = convert.(Int64,round.(gdf[!,:timeNoAs]))
    gdf[!,:timeNoSs] = convert.(Int64,round.(gdf[!,:timeNoSs]))
    gdf[!,:timeAfw] = convert.(Int64,round.(gdf[!,:timeAfw]))
    
    if mode == "sparse_reg" 
        rename!(gdf, :time_scip_tol => :timeScipTol)
        rename!(gdf, :termination_scip_tol => :terminationScipTol)
        gdf[!,:timeScipTol] = convert.(Int64,round.(gdf[!,:timeScipTol]))
    end

    if mode != "tailed_cardinality" && mode != "tailed_cardinality_sparse_log_reg"
        gdf[!,:timeIpopt] = convert.(Int64,round.(gdf[!,:timeIpopt]))
    end

    # relative instances solved
    gdf[!,:terminationBosciaRel] = gdf[!,:terminationBoscia]./gdf[!,:NumInstances]*100
    gdf[!,:terminationScipRel] = gdf[!,:terminationScip]./gdf[!,:NumInstances]*100
    gdf[!,:terminationNoWsRel] = gdf[!,:terminationNoWs]./gdf[!,:NumInstances]*100
    gdf[!,:terminationNoAsRel] = gdf[!,:terminationNoAs]./gdf[!,:NumInstances]*100
    gdf[!,:terminationNoSsRel] = gdf[!,:terminationNoSs]./gdf[!,:NumInstances]*100
    gdf[!,:terminationAfwRel] .= gdf[!,:terminationAfw]./gdf[!,:NumInstances]*100

    if mode != "tailed_cardinality" && mode != "tailed_cardinality_sparse_log_reg"
        gdf[!,:terminationIpoptRel] = gdf[!,:terminationIpopt]./gdf[!,:NumInstances]*100
    end

    if mode == "sparse_reg"
        gdf[!,:terminationScipTolRel] = gdf[!,:terminationScipTol]./gdf[!,:NumInstances]*100
    end

    # parse to int
    gdf[!,:terminationBosciaRel] = convert.(Int64,round.(gdf[!,:terminationBosciaRel]))
    gdf[!,:terminationScipRel] = convert.(Int64,round.(gdf[!,:terminationScipRel]))
    gdf[!,:terminationNoWsRel] = convert.(Int64,round.(gdf[!,:terminationNoWsRel]))
    gdf[!,:terminationNoAsRel] = convert.(Int64,round.(gdf[!,:terminationNoAsRel]))
    gdf[!,:terminationNoSsRel] = convert.(Int64,round.(gdf[!,:terminationNoSsRel]))
    gdf[!,:terminationAfwRel] = convert.(Int64,round.(gdf[!,:terminationAfwRel]))

    if mode != "tailed_cardinality" && mode != "tailed_cardinality_sparse_log_reg"
        gdf[!,:terminationIpoptRel] = convert.(Int64, round.(gdf[!,:terminationIpoptRel]))
    end

    if mode == "sparse_reg"
        gdf[!,:terminationScipTolRel] = convert.(Int64,round.(gdf[!,:terminationScipTolRel]))
    end

    # geo_mean of intersection with solved instances by all solvers except for scip oa and ipopt
    if mode != "tailed_cardinality" && mode != "tailed_cardinality_sparse_log_reg"
        df_intersection = select(df, Not([:time_scip, :time_ipopt, :termination_scip, :termination_ipopt, :solution_boscia, :solution_ipopt, :solution_scip]))
    else 
        df_intersection = select(df, Not([:time_scip, :termination_scip, :solution_boscia, :solution_scip]))
    end

    size_df = (size(gdf))

    # print(first, gdf,5)

    # deletes entire row if scip solves solution but boscia does not
    df_intersection = filter(row -> !(row.termination_boscia == 0 || row.termination_afw == 0 || row.termination_no_ws == 0 || row.termination_no_ss == 0 || row.termination_no_as == 0),  df_intersection)

    if mode != "poisson" && mode != "sparse_reg" && mode != "sparse_log_reg" &&  mode != "tailed_cardinality" && mode != "tailed_cardinality_sparse_log_reg" && mode != "22433" && mode != "neos5" && mode != "pg5_34" && mode != "ran14x18"
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
    elseif mode == "sparse_log_reg"
        df_intersection = combine(
            groupby(df_intersection, [:dimension, :p, :k, :M, :varA]), 
            :time_boscia => geo_mean => :BosciaGeoMeanIntersection,
            :time_no_ws => geo_mean => :NoWsGeoMeanIntersection,
            :time_no_as => geo_mean => :NoAsGeoMeanIntersection,
            :time_no_ss => geo_mean => :NoSsGeoMeanIntersection,
            :time_afw => geo_mean => :AfwGeoMeanIntersection,
            renamecols=false
            )
    elseif mode == "tailed_cardinality"
        df_intersection = combine(
            groupby(df_intersection, [:n0, :m0, :M]), 
            :time_boscia => geo_mean => :BosciaGeoMeanIntersection,
            :time_no_ws => geo_mean => :NoWsGeoMeanIntersection,
            :time_no_as => geo_mean => :NoAsGeoMeanIntersection,
            :time_no_ss => geo_mean => :NoSsGeoMeanIntersection,
            :time_afw => geo_mean => :AfwGeoMeanIntersection,
            renamecols=false
            )
    elseif mode == "tailed_cardinality_sparse_log_reg"
        df_intersection = combine(
            groupby(df_intersection, [:dimension, :M,:varA]), 
            :time_boscia => geo_mean => :BosciaGeoMeanIntersection,
            :time_no_ws => geo_mean => :NoWsGeoMeanIntersection,
            :time_no_as => geo_mean => :NoAsGeoMeanIntersection,
            :time_no_ss => geo_mean => :NoSsGeoMeanIntersection,
            :time_afw => geo_mean => :AfwGeoMeanIntersection,
            renamecols=false
            )
    elseif mode == "22433" || mode == "neos5" || mode == "pg5_34" || mode == "ran14x18"
        # print(first(df_intersection, 3))
        df_intersection = combine(
            groupby(df_intersection, [:num_v]), 
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

    size_df_after = size(gdf)

    if mode != "poisson" && mode != "sparse_reg" && mode != "sparse_log_reg" && mode != "tailed_cardinality" && mode != "tailed_cardinality_sparse_log_reg" && mode != "22433" && mode != "neos5" && mode != "pg5_34" && mode != "ran14x18"
        gdf = outerjoin(gdf, df_intersection, on =[:dimension])
    elseif mode == "poisson"
        gdf = innerjoin(gdf, df_intersection, on =[:dimension, :k, :Ns])
    elseif mode == "sparse_reg"
        gdf = outerjoin(gdf, df_intersection, on =[:dimension, :p, :k])
        gdf[!,:k] = convert.(Int64,round.(gdf[!,:k]))
    elseif mode == "sparse_log_reg"
        gdf = outerjoin(gdf, df_intersection, on =[:dimension, :p, :k, :M, :varA])
        sort!(gdf, [:dimension, :p, :M, :varA])
    elseif mode == "tailed_cardinality"
        gdf[!,:M] = convert.(Int64,round.(gdf[!,:M]))
        gdf = outerjoin(gdf, df_intersection, on =[:n0, :m0, :M])
    elseif mode == "tailed_cardinality_sparse_log_reg"
        gdf = innerjoin(gdf, df_intersection, on =[:dimension, :M,:varA])
    elseif mode == "22433" || mode == "neos5" || mode == "pg5_34" || mode == "ran14x18"
        gdf = outerjoin(gdf, df_intersection, on =[:num_v])
    end

    # intersection of Boscia, Ipopt, SCIP OA
    if mode != "tailed_cardinality" && mode != "tailed_cardinality_sparse_log_reg" && mode != "ran14x18"
        df_intersection = select(df, Not([:termination_afw, :time_afw, :termination_no_ws, :time_no_ws, :termination_no_as, :time_no_as, :termination_no_ss, :time_no_ss ]))
        df_intersection = filter(row -> !(row.optimal_boscia == 0 || row.optimal_scip == 0 || row.optimal_ipopt == 0),  df_intersection)
   #= elseif mode == "ran14x18"
        df_intersection = df
        df_intersection = filter(row -> !(row.optimal_boscia == 0 || row.optimal_scip == 0 || row.optimal_ipopt == 0),  df_intersection)=#
    else
        df_intersection = select(df, Not([:termination_afw, :time_afw, :termination_no_ws, :time_no_ws, :termination_no_as, :time_no_as, :termination_no_ss, :time_no_ss ]))
        df_intersection = filter(row -> !(row.optimal_boscia == 0 || row.optimal_scip == 0),  df_intersection)
    end
    if mode == "poisson"
        df_intersection = combine(
            groupby(df_intersection, [:dimension, :k, :Ns]), 
            :time_boscia => geo_mean => :BosciaGeoMeanIntersectionSolvers,
            :time_scip => geo_mean => :ScipGeoMeanIntersectionSolvers,
            :time_ipopt => geo_mean => :IpoptGeoMeanIntersectionSolvers,
            renamecols=false
        )

        # parse to int
        df_intersection[!,:BosciaGeoMeanIntersectionSolvers] = convert.(Int64,round.(df_intersection[!,:BosciaGeoMeanIntersectionSolvers]))
        df_intersection[!,:ScipGeoMeanIntersectionSolvers] = convert.(Int64,round.(df_intersection[!,:ScipGeoMeanIntersectionSolvers]))
        df_intersection[!,:IpoptGeoMeanIntersectionSolvers] = convert.(Int64,round.(df_intersection[!,:IpoptGeoMeanIntersectionSolvers]))
    end
    
 
    if mode == "poisson"
        # gdf = outerjoin(gdf, df_intersection, on =[:dimension, :k, :Ns])
    elseif mode == "mixed_portfolio" || mode == "integer"
        df_intersection = combine(
            groupby(df_intersection, :dimension), 
            :time_boscia => geo_mean => :BosciaGeoMeanIntersectionSolvers,
            :time_scip => geo_mean => :ScipGeoMeanIntersectionSolvers,
            :time_ipopt => geo_mean => :IpoptGeoMeanIntersectionSolvers,
            renamecols=false
        )
    elseif mode == "sparse_log_reg"
        df_intersection = combine(
            groupby(df_intersection, [:dimension, :p, :k, :M, :varA]), 
            :time_boscia => geo_mean => :BosciaGeoMeanIntersectionSolvers,
            :time_scip => geo_mean => :ScipGeoMeanIntersectionSolvers,
            :time_ipopt => geo_mean => :IpoptGeoMeanIntersectionSolvers,
            renamecols=false
        )
    elseif mode == "sparse_reg"
        df_intersection = combine(
            groupby(df_intersection, [:dimension, :p, :k]), 
            :time_boscia => geo_mean => :BosciaGeoMeanIntersectionSolvers,
            :time_scip => geo_mean => :ScipGeoMeanIntersectionSolvers,
            :time_ipopt => geo_mean => :IpoptGeoMeanIntersectionSolvers,
            renamecols=false
        )
    elseif mode == "tailed_cardinality"
        df_intersection = combine(
            groupby(df_intersection, [:n0, :m0, :M]), 
            :time_boscia => geo_mean => :BosciaGeoMeanIntersectionSolvers,
            :time_scip => geo_mean => :ScipGeoMeanIntersectionSolvers,
            renamecols=false
        )
    elseif mode == "tailed_cardinality_sparse_log_reg"
        df_intersection = combine(
            groupby(df_intersection, [:dimension, :M,:varA]), 
            :time_boscia => geo_mean => :BosciaGeoMeanIntersectionSolvers,
            :time_scip => geo_mean => :ScipGeoMeanIntersectionSolvers,
            renamecols=false
        )
    elseif mode == "22433" || mode == "neos5" || mode == "pg5_34" || mode == "ran14x18"
        df_intersection = combine(
            groupby(df_intersection, [:num_v]), 
            :time_boscia => geo_mean => :BosciaGeoMeanIntersectionSolvers,
            :time_scip => geo_mean => :ScipGeoMeanIntersectionSolvers,
            :time_ipopt => geo_mean => :IpoptGeoMeanIntersectionSolvers,
            renamecols=false
        )
    end

    if size(df_intersection)[1] != 0
        # parse to int
        df_intersection[!,:BosciaGeoMeanIntersectionSolvers] = convert.(Int64,round.(df_intersection[!,:BosciaGeoMeanIntersectionSolvers]))
        df_intersection[!,:ScipGeoMeanIntersectionSolvers] = convert.(Int64,round.(df_intersection[!,:ScipGeoMeanIntersectionSolvers]))
        if mode != "tailed_cardinality" && mode != "tailed_cardinality_sparse_log_reg"
            df_intersection[!,:IpoptGeoMeanIntersectionSolvers] = convert.(Int64,round.(df_intersection[!,:IpoptGeoMeanIntersectionSolvers]))
        end
        if mode == "poisson"
            gdf = outerjoin(gdf, df_intersection, on =[:dimension, :k, :Ns])
            gdf[!,:k] = convert.(Int64,round.(gdf[!,:k]))
        elseif mode == "mixed_portfolio" || mode == "integer"
            gdf = outerjoin(gdf, df_intersection, on =[:dimension])
        elseif mode == "sparse_reg"
            gdf = outerjoin(gdf, df_intersection, on =[:dimension, :p, :k])
        elseif mode == "sparse_log_reg"
            gdf = outerjoin(gdf, df_intersection, on =[:dimension, :p, :k, :M, :varA])
            gdf[!,:k] = convert.(Int64,round.(gdf[!,:k]))
        elseif mode == "tailed_cardinality"
            gdf = outerjoin(gdf, df_intersection, on =[:n0, :m0, :M])   
            gdf[!,:M] = convert.(Int64,round.(gdf[!,:M]))  
        elseif mode == "tailed_cardinality_sparse_log_reg"
            gdf = outerjoin(gdf, df_intersection, on =[:dimension, :M,:varA])
        elseif mode == "22433" || mode == "neos5" || mode == "pg5_34" || mode == "ran14x18"
            gdf = outerjoin(gdf, df_intersection, on =[:num_v])
        end
    end

    gdf[!,:terminationBoscia] = convert.(Int64,gdf[!,:terminationBoscia])

    gdf[!,:terminationScip] = convert.(Int64,gdf[!,:terminationScip])
    gdf[!,:terminationNoAs] = convert.(Int64,gdf[!,:terminationNoAs])
    gdf[!,:terminationNoWs] = convert.(Int64,gdf[!,:terminationNoWs])
    gdf[!,:terminationNoSs] = convert.(Int64,gdf[!,:terminationNoSs])
    gdf[!,:terminationAfw] = convert.(Int64,gdf[!,:terminationAfw])

    if mode == "neos5" || mode == "ran14x18"
        rename!(gdf, :termination_sc => :terminationSc)
        gdf[!,:terminationSc] = convert.(Int64,gdf[!,:terminationSc])
        rename!(gdf, :time_sc => :timeSc)
        gdf[!,:timeSc] = convert.(Int64,round.(gdf[!,:timeSc]))
        rename!(gdf, :rel_gap_sc => :relGapSc)
        non_inf_entries = findall(isfinite, gdf[!, :relGapSc])
        gdf[non_inf_entries, :relGapSc] = convert.(Int64, round.(gdf[non_inf_entries, :relGapSc]*100))
        gdf[!,:terminationScRel] = gdf[!,:terminationSc]./gdf[!,:NumInstances]*100
        gdf[!,:terminationScRel] = convert.(Int64,gdf[!,:terminationScRel])
        rename!(gdf, :rel_gap_boscia_sc => :relGapBosciaSc)
        gdf[!, :relGapBosciaSc] = round.(gdf[!, :relGapBosciaSc]*100)
    end
    if mode != "tailed_cardinality" && mode != "tailed_cardinality_sparse_log_reg"
        gdf[!,:terminationIpopt] = convert.(Int64,gdf[!,:terminationIpopt])
    end

    # print(select(gdf,[:timeSc,:timeBoscia]))
    
    mapcols!(col -> replace(col, missing => "-"), gdf)

    # save csv
    if mode == "integer"
        file_name = joinpath(@__DIR__, "csv/integer_portfolio.csv")
    elseif mode == "mixed" 
        file_name = joinpath(@__DIR__, "csv/mixed_50.csv")
    elseif mode == "mixed_portfolio"
        file_name = joinpath(@__DIR__, "csv/mixed_portfolio.csv")
    elseif mode == "poisson" 
        file_name = joinpath(@__DIR__, "csv/poisson.csv")
    elseif mode == "sparse_reg" 
        file_name = joinpath(@__DIR__, "csv/sparse_reg.csv")   
    elseif mode == "sparse_log_reg"
        gdf[!,:k] = convert.(Int64,round.(gdf[!,:k]))
        file_name = joinpath(@__DIR__, "csv/sparse_log_reg.csv") 
    elseif mode == "tailed_cardinality"
        rename!(gdf, :n0 => :n)
        rename!(gdf, :m0 => :m)
        file_name = joinpath(@__DIR__, "csv/tailed_cardinality.csv") 
    elseif mode == "tailed_cardinality_sparse_log_reg"
        file_name = joinpath(@__DIR__, "csv/tailed_cardinality_sparse_log_reg.csv") 
    elseif mode == "22433"
        rename!(gdf, :num_v => :numV)
        file_name = joinpath(@__DIR__, "csv/mip_lib_22433.csv") 
    elseif mode == "neos5"
        rename!(gdf, :num_v => :numV)
        file_name = joinpath(@__DIR__, "csv/mip_lib_neos5.csv") 
    elseif mode == "pg5_34"
        rename!(gdf, :num_v => :numV)
        file_name = joinpath(@__DIR__, "csv/mip_lib_pg5_34.csv") 
    elseif mode == "ran14x18"
        rename!(gdf, :num_v => :numV)
        file_name = joinpath(@__DIR__, "csv/mip_lib_ran14x18.csv") 
    else
        println("Specify file name!!")
    end        
    CSV.write(file_name, gdf, append=false)
end
