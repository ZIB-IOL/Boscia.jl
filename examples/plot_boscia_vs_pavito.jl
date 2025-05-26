using PyPlot
using DataFrames
using CSV

# function plot_boscia_vs_scip(example; boscia=true, scip_oa=false, ipopt=false, afw=true, ss=true, as=true, as_ss=true, boscia_methods=true)
function plot_boscia_vs_pavito(example; use_shot=true)
    if example in ["tailed_cardinality", "tailed_cardinality_sparse_log_reg"]
        boscia=true 
        scip_oa=true
        pavito=false
        ipopt = false
        shot=false
    else
        boscia=true 
        scip_oa=true
        ipopt=true
        pavito=true
        shot=use_shot
    end

    df = DataFrame(CSV.File(joinpath(@__DIR__, "final_csvs/" * example * "_comparison_non_grouped.csv")))
    df[df.timeBoscia .> 1810,:timeBoscia] .= 1810.0
    df[df.timeScipOA .> 1810,:timeScipOA] .= 1810.0
    if !(example in ["tailed_cardinality", "tailed_cardinality_sparse_log_reg"])
        df[df.timePavito .> 1810,:timePavito] .= 1810.0
        df[df.timeShot .> 1810,:timeShot] .= 1810.0
        df[df.timeIpopt .> 1810,:timeIpopt] .= 1810.0
    end
   #= if example == "poisson"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "final_csvs/poisson_non_grouped.csv")))
    elseif example == "sparse_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "final_csvs/sparse_reg_non_grouped.csv")))
    elseif example == "mixed_portfolio"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/mixed_portfolio_non_grouped.csv")))
    elseif example == "integer"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/portfolio_integer_non_grouped.csv")))
    elseif example == "sparse_log_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/sparse_log_reg_non_grouped.csv")))
    elseif example == "tailed_sparse_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/tailed_cardinality_non_grouped.csv")))
    elseif example == "tailed_sparse_log_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/tailed_cardinality_sparse_log_reg_non_grouped.csv")))
    elseif example == "miplib"
        df_22433 = DataFrame(CSV.File(joinpath(@__DIR__, "csv/mip_lib_22433_non_grouped.csv")))
        df = select(df_22433, [:time_boscia, :termination_boscia, :optimal_boscia, :time_ipopt, :termination_ipopt, :optimal_ipopt, :time_pavito, :termination_pavito, :optimal_pavito, :time_shot, :termination_shot, :optimal_shot])
        df_neos5 = DataFrame(CSV.File(joinpath(@__DIR__, "csv/mip_lib_neos5_non_grouped.csv")))
        select!(df_neos5, [:time_boscia, :termination_boscia, :optimal_boscia, :time_ipopt, :termination_ipopt, :optimal_ipopt, :time_pavito, :termination_pavito, :optimal_pavito, :time_shot, :termination_shot, :optimal_shot])
        append!(df,df_neos5)
        df_pg534 = DataFrame(CSV.File(joinpath(@__DIR__, "csv/mip_lib_pg5_34_non_grouped.csv")))
        select!(df_pg534, [:time_boscia, :termination_boscia, :optimal_boscia, :time_ipopt, :termination_ipopt, :optimal_ipopt, :time_shot, :termination_shot, :optimal_shot]) # :time_pavito, :termination_pavito, :optimal_pavito])
        append!(df,df_pg534, cols=:subset)
        df_ran14x18 = DataFrame(CSV.File(joinpath(@__DIR__, "csv/mip_lib_ran14x18-disj-8_non_grouped.csv")))
        select!(df_ran14x18, [:time_boscia, :termination_boscia, :optimal_boscia, :time_ipopt, :termination_ipopt, :optimal_ipopt, :time_pavito, :termination_pavito, :optimal_pavito, :time_shot, :termination_shot, :optimal_shot])
        append!(df,df_ran14x18)

        df[df.time_boscia.>1800, :time_boscia] .= 1800
        df[df.time_ipopt.>1800, :time_ipopt] .= 1800
    else
        error("wrong option")
    end =#

    time_limit = 1800

    colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
    markers = ["o", "s", "^", "P", "X", "H", "D"]

    fig = plt.figure(figsize=(7.3, 5))
    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("font", size=12, family="cursive")
    PyPlot.matplotlib[:rc]("axes", labelsize=12)
    PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
    \usepackage{newtxtext}
    """)
    #\usepackage{libertinust1math}
    ax = fig.add_subplot(111)

    #ax.scatter(df_temp[!,"dimension"], df_temp[!,"time_boscia"], label="BO (ours)", color=colors[1], marker=markers[1])
    #ax.scatter(df_temp[!,"dimension"], df_temp[!,"time_scip"], label="SCIP", color=colors[3], marker=markers[2])

    if boscia 
        println("Plot Boscia")
        df_boscia = deepcopy(df)
        filter!(row -> !(row.terminationBoscia == 0),  df_boscia)
        #filter!(row -> !(row.optimal_boscia == 0),  df_boscia)

        time_boscia = sort(df_boscia[!,"timeBoscia"])
        push!(time_boscia, 1.1 * time_limit)
        if !isempty(df_boscia)
            ax.plot(time_boscia, [1:nrow(df_boscia); nrow(df_boscia)], label="Boscia", color=cb_green_lime, marker=markers[1], markevery=0.05)
        end
    end

    if scip_oa     
        println("Plot SCIP")
        df_scip = deepcopy(df)
        filter!(row -> !(row.terminationScipOA == 0),  df_scip)
        #filter!(row -> !(row.optimal_scip == 0),  df_scip)
        
        time_scip = sort(df_scip[!,"timeScipOA"])
        push!(time_scip, 1.1 * time_limit)
        if !isempty(df_scip)
            ax.plot(time_scip, [1:nrow(df_scip); nrow(df_scip)], label="SCIP+OA", color=cb_clay, marker=markers[2], markevery=0.05)
        end
    end 

    if ipopt
        println("Plot Ipopt")
        df_ipopt = deepcopy(df)
        filter!(row -> !(row.terminationIpopt == 0),  df_ipopt)
        #filter!(row -> !(row.optimal_ipopt == 0),  df_ipopt)

        time_ipopt = sort(df_ipopt[!,"timeIpopt"])
        push!(time_ipopt, 1.1 * time_limit)
        if !isempty(df_ipopt)
            ax.plot(time_ipopt, [1:nrow(df_ipopt); nrow(df_ipopt)], label="BnB Ipopt", color=cb_blue, marker=markers[3], markevery=0.05)
        end
    end 

    if pavito
        println("Plot Pavito")
        df_pavito = deepcopy(df)
        filter!(row -> !ismissing(row.terminationPavito),  df_pavito)
        filter!(row -> !(row.terminationPavito == 0),  df_pavito)
        #if boscia && scip_oa && ipopt && pavito
        #    filter!(row -> !(row.optimal_pavito == 0),  df_pavito)
        #elseif example == "miplib"
        #    filter!(row -> !ismissing(row.optimal_pavito),  df_pavito)
        #    filter!(row -> !(row.optimal_pavito == 0),  df_pavito)
        #end
        time_pavito = sort(df_pavito[!,"timePavito"])
        push!(time_pavito, 1.1 * time_limit)
        if !isempty(df_pavito)
            ax.plot(time_pavito, [1:nrow(df_pavito); nrow(df_pavito)], label="Pavito", color=cb_lilac, marker=markers[4], markevery=0.05)
        end
    end 

    if shot
        println("Plot Shot")
        df_shot = deepcopy(df)
        filter!(row -> !(row.terminationShot == 0),  df_shot)
        #filter!(row -> !(row.optimal_shot == 0),  df_shot)

        time_shot = sort(df_shot[!,"timeShot"])
        push!(time_shot, 1.1 * time_limit)
        if !isempty(df_shot)
            ax.plot(time_shot, [1:nrow(df_shot); nrow(df_shot)], label="SHOT", color=cb_brown, marker=markers[5], markevery=0.05)
        end
    end 
println("Set up labels and title")
    ylabel("Solved instances")
    #locator_params(axis="y", nbins=4)
    xlabel("Time (s)")
    ax.set_xscale("log")
    ax.grid()
    ncol = shot ? 3 : 4

    if example == "portfolio_integer"
        title("Pure-Integer Portfolio Problem", loc="center")
    elseif example == "poisson_reg"
        title("Poisson Regression", loc="center")
    elseif example == "sparse_reg"
        title("Sparse Regression", loc="center")
    elseif example == "portfolio_mixed"
        title("Mixed-Integer Portfolio Problem", loc="center")
    elseif example == "sparse_log_reg"
        title("Sparse Log Regression", loc="center")
    elseif example == "tailed_cardinality"
        title("Tailed Sparse Regression", loc="center")
    elseif example == "tailed_cardinality_sparse_log_reg"
        title("Tailed Sparse Log Regression", loc="center")
    elseif example == "miplib_22433"
        title("MIP LIB 22433", loc="center")
    elseif example == "miplib_neos5"
        title("MIP LIB neos5", loc="center")
    elseif example == "miplib_pg5_34"
        title("MIP LIB pg5_34", loc="center")
    elseif example == "miplib_ran14x18-disj-8"
        title("MIP LIB ran14x18-disj-8", loc="center")
    end

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12,
    fancybox=true, shadow=false, ncol=ncol)

fig.tight_layout()

    if pavito && !scip_oa
        file = "plots/" * example * "_boscia_pavito.pdf"
    elseif !shot
        file = joinpath(@__DIR__, "plots/" * example * "_comparison_solvers_without_shot.pdf")
    else 
        file = joinpath(@__DIR__, "plots/" * example * "_comparison_solvers.pdf")
    end
    savefig(file)
end
#=
examples = ["miplib_22433", "miplib_neos5", "miplib_pg5_34", "miplib_ran14x18-disj-8", "poisson_reg", "portfolio_integer", "portfolio_mixed", "sparse_log_reg", "sparse_reg", "tailed_cardinality", "tailed_cardinality_sparse_log_reg"]
for example in examples
    @show example
    plot_boscia_vs_pavito(example)
    #plot_boscia_vs_pavito(example, use_shot=false)
end 
=#