using PyPlot
using DataFrames
using CSV

# "sparse_reg", "30_1"
# "sparse_reg", "20_1"

# "integer_portfolio", "30_4"
# "mixed_portfolio" "35_3"

# "sparse_log_reg", "18_1-1_1"
# "sparse_log_reg", "25_5-5.0_1"

function plot(example, setup)
    try 
    if example == "sparse_reg"
        df_boscia = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_default_" * setup * "_sparse_reg.csv")))
        df_global_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_global_tightening_" * setup * "_sparse_reg.csv")))
        df_local_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_local_tightening_" * setup * "_sparse_reg.csv")))
        df_no_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_no_tightening_" * setup * "_sparse_reg.csv")))
    elseif example == "sparse_log_reg"
        df_boscia = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_sparse_log_regression_" * setup * ".csv")))
        df_global_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/global_tightening_sparse_log_regression_" * setup * ".csv")))
        df_local_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/local_tightening_sparse_log_regression_" * setup * ".csv")))
        df_no_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_tightening_sparse_log_regression_" * setup * ".csv")))
    elseif example == "integer_portfolio"
        df_boscia = DataFrame(CSV.File(joinpath(@__DIR__, "csv/default_" * setup * "_integer_portfolio.csv")))
        df_global_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/global_tightening_" * setup * "_integer_portfolio.csv")))
        df_local_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/local_tightening_" * setup * "_integer_portfolio.csv")))
        df_no_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_tightening_" * setup * "_integer_portfolio.csv")))
    elseif example == "mixed_portfolio"
        df_boscia = DataFrame(CSV.File(joinpath(@__DIR__, "csv/default_" * setup * "_mixed_portfolio.csv")))
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
    # shadow set
    elseif example == "ss"
        df_boscia = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_" * setup * "_mixed_portfolio.csv")))
        df_global_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/global_tightening_" * setup * "_mixed_portfolio.csv")))
        df_local_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/local_tightening_" * setup * "_mixed_portfolio.csv")))
        df_no_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_tightening_" * setup * "_mixed_portfolio.csv")))
        df_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/ss_" * setup * "_mixed_portfolio.csv")))
        df_ss_gt = DataFrame(CSV.File(joinpath(@__DIR__, "csv/ss_global_tightening_" * setup * "_mixed_portfolio.csv")))
        df_ss_lt = DataFrame(CSV.File(joinpath(@__DIR__, "csv/ss_local_tightening_" * setup * "_mixed_portfolio.csv")))
        df_ss_nt = DataFrame(CSV.File(joinpath(@__DIR__, "csv/ss_no_tightening_" * setup * "_mixed_portfolio.csv")))
    end

    colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
    markers = ["o", "s", "^", "P", "X", "H", "D"]

    fig = plt.figure(figsize=(6.5,3.5))
    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("font", size=12, family="cursive")
    PyPlot.matplotlib[:rc]("axes", labelsize=12)
    PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
    \usepackage{newtxtext}
    """)

    if example != "sc_neos5" && example != "sc_ran14x18" && example != "ss"
        df_boscia[!,:time] = df_boscia[!,:time]./1000.0
        df_global_tightening[!,:time] = df_global_tightening[!,:time]./1000.0
        df_local_tightening[!,:time] = df_local_tightening[!,:time]./1000.0
        df_no_tightening[!,:time] = df_no_tightening[!,:time]./1000.0

        fig = plt.figure(figsize=(6.5,9.5))
        #ax = fig.add_subplot(311)

        #ax.plot(1:length(df_boscia[!,"openNodes"]), df_boscia[!,"openNodes"], label="BO (ours)", color=colors[1], marker=markers[1], markevery=0.1, alpha=.5)
        #ax.plot(1:length(df_global_tightening[!,"openNodes"]), df_global_tightening[!,"openNodes"], label="Global tightening", color=colors[end], marker=markers[2], markevery=0.1, alpha=.5)
        #ax.plot(1:length(df_local_tightening[!,"openNodes"]), df_local_tightening[!,"openNodes"], label="Local tightening", color=colors[2], marker=markers[3], markevery=0.1, alpha=.5)
        #ax.plot(1:length(df_no_tightening[!,"openNodes"]), df_no_tightening[!,"openNodes"], label="No tightening", color=colors[3], marker=markers[4], markevery=0.1, alpha=.5)

        #ylabel("Open nodes")
        #locator_params(axis="y", nbins=4)
        #xlabel("Iteration")
        #ax.grid()

        # lb, time 
        ax = fig.add_subplot(211)
        ax.plot(df_boscia[!,"time"], df_boscia[!,"lowerBound"], label="Default", color=cb_blue, marker=markers[1], markevery=0.05, alpha=.5)
        ax.plot(df_global_tightening[!,"time"], df_global_tightening[!,"lowerBound"], label="Global tightening", color=cb_clay, marker=markers[2], markevery=0.05, alpha=.5)
        ax.plot(df_local_tightening[!,"time"], df_local_tightening[!,"lowerBound"], label="Local tightening", color=cb_lilac, marker=markers[3], markevery=0.05, alpha=.5)
        ax.plot(df_no_tightening[!,"time"], df_no_tightening[!,"lowerBound"], label="No tightening", color=cb_rose, marker=markers[4], markevery=0.05, alpha=.5)

        ylabel("Lower bound")
        #locator_params(axis="y", nbins=4)
        xlabel("Time (s)")
        ax.grid()

        # ncalls
        ax = fig.add_subplot(212)
        ax.plot(1:length(df_boscia[!,"time"]), df_boscia[!,"LMOcalls"], label="Default", color=cb_blue, marker=markers[1], markevery=0.05, alpha=.5)
        ax.plot(1:length(df_global_tightening[!,"time"]), df_global_tightening[!,"LMOcalls"], label="Global tightening", color=cb_clay, marker=markers[2], markevery=0.05, alpha=.5)
        ax.plot(1:length(df_local_tightening[!,"time"]), df_local_tightening[!,"LMOcalls"], label="Local tightening", color=cb_lilac, marker=markers[3], markevery=0.05, alpha=.5)
        ax.plot(1:length(df_no_tightening[!,"time"]), df_no_tightening[!,"LMOcalls"], label="No tightening", color=cb_rose, marker=markers[4], markevery=0.05, alpha=.5)

        ylabel("BLMO calls")
        #locator_params(axis="y", nbins=4)
        xlabel("Iteration")
        ax.grid()
    elseif example == "sc_neos5" || example == "sc_ran14x18"
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
        ax.plot(df_boscia[!,"time"], df_boscia[!,"lowerBound"], label="BO (ours)", color=cb_blue, marker=markers[1], markevery=0.05, alpha=.5)
        ax.plot(df_sc[!,"time"], df_sc[!,"lowerBound"], label="Strong convexity", color=cb_clay, marker=markers[2], markevery=0.05, alpha=.5)
        
        ylabel("Lower bound")
        #locator_params(axis="y", nbins=4)
        xlabel("Time (s)")
        ax.grid()
    elseif example == "ss"
        fig = plt.figure(figsize=(6.5,6.5))
        # ncalls
        ax = fig.add_subplot(211)
        ax.plot(1:length(df_boscia[!,"openNodes"]), df_boscia[!,"LMOcalls"], label="BO (ours)", color=cb_blue, marker=markers[1], markevery=0.05, alpha=.5)
        ax.plot(1:length(df_global_tightening[!,"openNodes"]), df_global_tightening[!,"LMOcalls"], label="Global tightening", color=cb_clay, marker=markers[2], markevery=0.05, alpha=.5)
        ax.plot(1:length(df_local_tightening[!,"openNodes"]), df_local_tightening[!,"LMOcalls"], label="Local tightening", color=cb_lilac, marker=markers[3], markevery=0.05, alpha=.5)
        ax.plot(1:length(df_no_tightening[!,"openNodes"]), df_no_tightening[!,"LMOcalls"], label="No tightening", color=cb_rose, marker=markers[4], markevery=0.05, alpha=.5)
 
        ylabel("BLMO calls")
        #locator_params(axis="y", nbins=4)
        xlabel("Iteration")
        if setup == "50_3"
            ylim((0,6500))
        elseif setup == "40_3"
            ylim((0,1100))
        end
        ax.grid()
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12,
        fancybox=true, shadow=false, ncol=2)

        ax = fig.add_subplot(212)
        ax.plot(1:length(df_ss[!,"openNodes"]), df_boscia[!,"LMOcalls"], label="BO without shadow set", color=cb_blue, marker=markers[1], markevery=0.05, alpha=.5)
        ax.plot(1:length(df_ss_gt[!,"openNodes"]), df_ss_gt[!,"LMOcalls"], label="Global tightening without shadow set", color=cb_clay, marker=markers[2], markevery=0.05, alpha=.5)
        ax.plot(1:length(df_ss_lt[!,"openNodes"]), df_ss_lt[!,"LMOcalls"], label="Local tightening without shadow set", color=cb_lilac, marker=markers[3], markevery=0.05, alpha=.5)
        ax.plot(1:length(df_ss_nt[!,"openNodes"]), df_ss_nt[!,"LMOcalls"], label="No tightening without shadow set", color=cb_rose, marker=markers[4], markevery=0.05, alpha=.5)

        ylabel("BLMO calls")
        #locator_params(axis="y", nbins=4)
        xlabel("Iteration")
        ax.grid()
    end

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12,
        fancybox=true, shadow=false, ncol=2)
    fig.tight_layout()
    file = joinpath(@__DIR__, "plots/progress_plots/" * example *  "/tightenings_" * setup * "_" * example * ".pdf")
    savefig(file)

    file = joinpath(@__DIR__, "plots/progress_plots/tightenings_" * setup * "_" * example * ".pdf")
    savefig(file)

catch e 
    println(e)
end
end
#=
## sparse regression
# "no_tightening", "local_tigtening", "global_tightening", hybrid_branching_20", "strong_branching"
example = "sparse_reg"
modes = ["nodes", "time"]
for mode in modes
    for m in [23] #15:30
        for seed in [5] #1:10
            plot(example, string(m) * "_" * string(seed))
        end
    end
end

# portfolio mixed
example = "mixed_portfolio"
for mode in modes
    for m in [75] #20:5:120
        for seed in [10] #1:10
            plot(example, string(m) * "_" * string(seed))
        end
    end
end

# portfolio integer
example = "integer_portfolio"
for mode in modes
    for m in [120]#20:5:120
        for seed in [1]#1:10
            plot(example, string(m) * "_" * string(seed))
        end
    end
end
=#
