using PyPlot
using DataFrames
using CSV

# "sparse_reg", "30_1"
# "sparse_reg", "20_1"

# "sparse_log_reg", "18_1-1_1"
# "sparse_log_reg", "25_5-5.0_1"

function plot(example, setup)
    if example == "sparse_reg"
        df_boscia = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_sparse_reg_" * setup * ".csv")))
        df_global_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/global_tightening_sparse_reg_" * setup * ".csv")))
        df_local_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/local_tightening_sparse_reg_" * setup * ".csv")))
        df_no_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_tightening_sparse_reg_" * setup * ".csv")))
    elseif example == "sparse_log_reg"
        df_boscia = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_sparse_log_regression_" * setup * ".csv")))
        df_global_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/global_tightening_sparse_log_regression_" * setup * ".csv")))
        df_local_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/local_tightening_sparse_log_regression_" * setup * ".csv")))
        df_no_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_tightening_sparse_log_regression_" * setup * ".csv")))
    elseif example == "integer_portfolio"
        df_boscia = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_" * setup * "_integer_portfolio.csv")))
        df_global_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/global_tightening_" * setup * "_integer_portfolio.csv")))
        df_local_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/local_tightening_" * setup * "_integer_portfolio.csv")))
        df_no_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_tightening_" * setup * "_integer_portfolio.csv")))
    elseif example == "mixed_portfolio"
        df_boscia = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_" * setup * "_mixed_portfolio.csv")))
        df_global_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/global_tightening_" * setup * "_mixed_portfolio.csv")))
        df_local_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/local_tightening_" * setup * "_mixed_portfolio.csv")))
        df_no_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_tightening_" * setup * "_mixed_portfolio.csv")))
    elseif example == "neos5"
        df_boscia = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_mip_lib_" * example * "_" * setup * ".csv")))
        df_global_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/global_tightening_mip_lib_" * example * "_" * setup * ".csv")))
        df_local_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/local_tightening_mip_lib_" * example * "_" * setup * ".csv")))
        df_no_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_tightening_mip_lib_" * example * "_" * setup * ".csv")))
    # strong convexity
    elseif example == "sc_neos5"
        df_boscia = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_mip_lib_neos5_" * setup * ".csv")))
        df_sc = DataFrame(CSV.File(joinpath(@__DIR__, "csv/strong_convexity_mip_lib_neos5_" * setup * ".csv")))
    elseif example == "sc_ran14x18"
        df_boscia = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_mip_lib_ran14x18-disj-8_" * setup * ".csv")))
        df_sc = DataFrame(CSV.File(joinpath(@__DIR__, "csv/strong_convexity_mip_lib_ran14x18-disj-8_" * setup * ".csv")))
    end

    colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
    markers = ["o", "s", "^", "P", "X", "H", "D"]

    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("font", size=12, family="cursive")
    PyPlot.matplotlib[:rc]("axes", labelsize=14)
    PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
    \usepackage{libertine}
    \usepackage{libertinust1math}
    """)

    if example != "sc_neos5" && example != "sc_ran14x18"
        fig = plt.figure(figsize=(6.5,9.5))
        ax = fig.add_subplot(311)

        ax.plot(1:length(df_boscia[!,"openNodes"]), df_boscia[!,"openNodes"], label="BO (ours)", color=colors[1], marker=markers[1], markevery=0.1, alpha=.5)
        ax.plot(1:length(df_global_tightening[!,"openNodes"]), df_global_tightening[!,"openNodes"], label="global tightening", color=colors[end], marker=markers[2], markevery=0.1, alpha=.5)
        ax.plot(1:length(df_local_tightening[!,"openNodes"]), df_local_tightening[!,"openNodes"], label="local tightening", color=colors[2], marker=markers[3], markevery=0.1, alpha=.5)
        ax.plot(1:length(df_no_tightening[!,"openNodes"]), df_no_tightening[!,"openNodes"], label="no tightening", color=colors[3], marker=markers[4], markevery=0.1, alpha=.5)

        ylabel("open nodes")
        #locator_params(axis="y", nbins=4)
        xlabel("iteration")
        ax.grid()

        # lb, time 
        ax = fig.add_subplot(312)
        ax.plot(df_boscia[!,"time"], df_boscia[!,"lowerBound"], label="BO (ours)", color=colors[1], marker=markers[1], markevery=0.1, alpha=.5)
        ax.plot(df_global_tightening[!,"time"], df_global_tightening[!,"lowerBound"], label="global tightening", color=colors[end], marker=markers[2], markevery=0.1, alpha=.5)
        ax.plot(df_local_tightening[!,"time"], df_local_tightening[!,"lowerBound"], label="local tightening", color=colors[2], marker=markers[3], markevery=0.1, alpha=.5)
        ax.plot(df_no_tightening[!,"time"], df_no_tightening[!,"lowerBound"], label="no tightening", color=colors[3], marker=markers[4], markevery=0.1, alpha=.5)

        ylabel("lower bound")
        #locator_params(axis="y", nbins=4)
        xlabel("time")
        ax.grid()

        # ncalls
        ax = fig.add_subplot(313)
        ax.plot(1:length(df_boscia[!,"openNodes"]), df_boscia[!,"LMOcalls"], label="BO (ours)", color=colors[1], marker=markers[1], markevery=0.1, alpha=.5)
        ax.plot(1:length(df_global_tightening[!,"openNodes"]), df_global_tightening[!,"LMOcalls"], label="global tightening", color=colors[end], marker=markers[2], markevery=0.1, alpha=.5)
        ax.plot(1:length(df_local_tightening[!,"openNodes"]), df_local_tightening[!,"LMOcalls"], label="local tightening", color=colors[2], marker=markers[3], markevery=0.1, alpha=.5)
        ax.plot(1:length(df_no_tightening[!,"openNodes"]), df_no_tightening[!,"LMOcalls"], label="no tightening", color=colors[3], marker=markers[4], markevery=0.1, alpha=.5)

        ylabel("LMO calls")
        #locator_params(axis="y", nbins=4)
        xlabel("iteration")
        ax.grid()
    else
        df_boscia[!,:time] = df_boscia[!,:time]./1000.0
        df_sc[!,:time] = df_sc[!,:time]./1000.0

        fig = plt.figure(figsize=(6.5,3.5))
        # ax = fig.add_subplot(211)

        # ax.plot(1:length(df_boscia[!,"openNodes"]), df_boscia[!,"openNodes"], label="BO (ours)", color=colors[1], marker=markers[1], markevery=0.1, alpha=.5)
        # ax.plot(1:length(df_sc[!,"openNodes"]), df_sc[!,"openNodes"], label="strong convexity", color=colors[end], marker=markers[2], markevery=0.1, alpha=.5)
        
        # ylabel("open nodes")
        # #locator_params(axis="y", nbins=4)
        # xlabel("iteration")
        # ax.grid()

        # lb, time 
        ax = fig.add_subplot(111)
        ax.plot(df_boscia[!,"time"], df_boscia[!,"lowerBound"], label="BO (ours)", color=colors[1], marker=markers[1], markevery=0.1, alpha=.5)
        ax.plot(df_sc[!,"time"], df_sc[!,"lowerBound"], label="Strong convexity", color=colors[end], marker=markers[2], alpha=.5)
        
        ylabel("Lower bound")
        #locator_params(axis="y", nbins=4)
        xlabel("Time (s)")
        ax.grid()

    end

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12,
        fancybox=true, shadow=false, ncol=2)
    fig.tight_layout()
    file = joinpath(@__DIR__, "csv/" * example * "_" * setup * "_tightenings.pdf")
    savefig(file)
end
