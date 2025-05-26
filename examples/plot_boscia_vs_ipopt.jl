using Statistics
#using Boscia
#using FrankWolfe
using Random
#using SCIP
#import Bonobo
import MathOptInterface
const MOI = MathOptInterface
using DataFrames
using CSV
using PyPlot

function plot_boscia_vs_ipopt(example; seed = 1, num_v = 4)
    file_name_boscia = joinpath(@__DIR__, "csv/default_mip_lib_" * example * "_" * string(num_v) * "_" * string(seed) * ".csv")
    file_name_ipopt = joinpath(@__DIR__, "csv/ipopt_" * example  * "_" * string(num_v) * "_" * string(seed) * ".csv")


    df_boscia = DataFrame(CSV.File(file_name_boscia))
    df_ipopt = DataFrame(CSV.File(file_name_ipopt))

    file_name = joinpath(@__DIR__, "plots/boscia_vs_ipopt_" * example * "_" * string(num_v) * "_" * string(seed) * ".csv") 
    file_name = replace(file_name, ".csv" => ".pdf")
    
    colors = ["b", "m", "c", "r", "g", "y", "k"]
    markers = ["o", "s", "^", "P", "X", "H", "D"]
    
    fig = plt.figure(figsize=(6.5,3.5))
    fig, axs = plt.subplots(2, sharex=false)
    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("font", size=12, family="cursive")
    PyPlot.matplotlib[:rc]("axes", labelsize=12)
    PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
        \usepackage{newtxtext}
    """)
    #ax = fig.add_subplot(111)

    df_boscia[!,"time"] ./= 1000
    df_ipopt[!,"time"] ./= 1000

    len_boscia = length(df_boscia[!, "lowerBound"])
    len_ipopt = length(df_ipopt[!, "lowerBound"])

    ub_ipopt_int = findfirst(isfinite, df_ipopt[!,"upperBound"])

    if example == "22433" || example == "pg5_34"
        axs[1].plot(df_boscia[!,"time"], df_boscia[!,"lowerBound"], label="Boscia LB", color=cb_green_sea, marker=markers[1], linestyle="dashed", linewidth=1.0)
        axs[1].plot(df_boscia[!,"time"], df_boscia[!,"upperBound"], label="Boscia UB", color=cb_green_sea, linestyle="dashdot", linewidth=3.0)
    else
        axs[1].plot(df_boscia[!,"time"], df_boscia[!,"lowerBound"], label="Boscia LB", color=cb_green_sea, marker=markers[1], linestyle="dashed", markevery=0.05, linewidth=1.0)
        axs[1].plot(df_boscia[!,"time"], df_boscia[!,"upperBound"], label="Boscia UB", color=cb_green_sea, linestyle="dashdot", markevery=0.05, linewidth=2.0)
    end
    axs[1].plot(df_ipopt[1:end-1,"time"], df_ipopt[1:end-1,"lowerBound"], label="BnB Ipopt LB", color=cb_rose, marker=markers[2], markevery=0.05, linewidth=1.0)
    if ub_ipopt_int !== nothing
        axs[1].plot(df_ipopt[ub_ipopt_int:end-1,"time"], df_ipopt[ub_ipopt_int:end-1,"upperBound"], label="BnB Ipopt UB", color=cb_rose, linestyle="dotted", markevery=0.05, linewidth=2.0)
    end
    axs[1].set(xlabel="Time (s)", ylabel ="Objective value")
    axs[1].grid()

    if example == "22433" || example == "pg5_34"
        axs[2].plot(1:len_boscia, df_boscia[!,"lowerBound"], label="", color=cb_green_sea, marker=markers[1], linestyle="dashed", linewidth=1.0)
        axs[2].plot(1:len_boscia, df_boscia[!,"upperBound"], label="", color=cb_green_sea, linestyle="dashdot", linewidth=3.0)
    else
        axs[2].plot(1:len_boscia, df_boscia[!,"lowerBound"], label="", color=cb_green_sea, marker=markers[1], linestyle="dashed", markevery=0.05, linewidth=1.0)
        axs[2].plot(1:len_boscia, df_boscia[!,"upperBound"], label="", color=cb_green_sea, linestyle="dashdot", markevery=0.05, linewidth=2.0)
    end
    axs[2].plot(1:len_ipopt-1, df_ipopt[1:end-1,"lowerBound"], label="", color=cb_rose, marker=markers[2], markevery=0.05, linewidth=1.0)
    if ub_ipopt_int !== nothing
        axs[2].plot(ub_ipopt_int:len_ipopt-1, df_ipopt[ub_ipopt_int:end-1,"upperBound"], label="", color=cb_rose, linestyle="dotted", markevery=0.05, linewidth=2.0)
    end
    axs[2].set(xlabel="Number of nodes", ylabel ="Objective value")
    axs[2].grid()

    lgd = fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), fontsize=12,   # bbox_to_anchor=(0.5, 0.05)
            fancybox=true, shadow=false, ncol=6)
    
    fig.tight_layout()

    savefig(file_name,  bbox_extra_artists=(lgd,), bbox_inches="tight")
end

function plot_boscia_vs_strong_convexity(example; seed = 1, num_v = 4)
    file_name_boscia = joinpath(@__DIR__, "csv/default_mip_lib_" * example * "_" * string(num_v) * "_" * string(seed) * ".csv")
    file_name_ipopt = joinpath(@__DIR__, "csv/strong_convexity_mip_lib_" * example  * "_" * string(num_v) * "_" * string(seed) * ".csv")


    df_boscia = DataFrame(CSV.File(file_name_boscia))
    df_ipopt = DataFrame(CSV.File(file_name_ipopt))

    file_name = joinpath(@__DIR__, "plots/boscia_vs_strong_convexity_" * example * "_" * string(num_v) * "_" * string(seed) * ".csv") 
    file_name = replace(file_name, ".csv" => ".pdf")
    
    colors = ["b", "m", "c", "r", "g", "y", "k"]
    markers = ["o", "s", "^", "P", "X", "H", "D"]
    
    fig = plt.figure(figsize=(6.5,3.5))
    fig, axs = plt.subplots(2, sharex=false)
    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("font", size=12, family="cursive")
    PyPlot.matplotlib[:rc]("axes", labelsize=12)
    PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
    \usepackage{newtxtext}
    """)
    #ax = fig.add_subplot(111)

    df_boscia[!,"time"] ./= 1000
    df_ipopt[!,"time"] ./= 1000

    len_boscia = length(df_boscia[!, "lowerBound"])
    len_ipopt = length(df_ipopt[!, "lowerBound"])

    ub_ipopt_int = findfirst(isfinite, df_ipopt[!,"upperBound"])

    if example == "22433" || example == "pg5_34"
        axs[1].plot(df_boscia[!,"time"], df_boscia[!,"lowerBound"], label="Default LB", color=cb_rose, marker=markers[1], linestyle="dashed", linewidth=1.0)
        axs[1].plot(df_boscia[!,"time"], df_boscia[!,"upperBound"], label="Default UB", color=cb_rose, linestyle="dashdot", linewidth=3.0)
    else
        axs[1].plot(df_boscia[!,"time"], df_boscia[!,"lowerBound"], label="Default LB", color=cb_rose, marker=markers[1], linestyle="dashed", markevery=0.05, linewidth=1.0)
        axs[1].plot(df_boscia[!,"time"], df_boscia[!,"upperBound"], label="Default UB", color=cb_rose, linestyle="dashdot", markevery=0.05, linewidth=2.0)
    end
    axs[1].plot(df_ipopt[1:end-1,"time"], df_ipopt[1:end-1,"lowerBound"], label="Strong convexity LB", color=cb_green_sea, marker=markers[2], markevery=0.05, linewidth=1.0)
    if ub_ipopt_int !== nothing
        axs[1].plot(df_ipopt[ub_ipopt_int:end-1,"time"], df_ipopt[ub_ipopt_int:end-1,"upperBound"], label="Strong convexity UB", color=cb_green_sea, linestyle="dotted", markevery=0.05, linewidth=2.0)
    end
    axs[1].set(xlabel="Time (s)", ylabel ="Objective value")
    axs[1].grid()

    if example == "22433" || example == "pg5_34"
        axs[2].plot(1:len_boscia, df_boscia[!,"lowerBound"], label="", color=cb_rose, marker=markers[1], linestyle="dashed", linewidth=1.0)
        axs[2].plot(1:len_boscia, df_boscia[!,"upperBound"], label="", color=cb_rose, linestyle="dashdot", linewidth=3.0)
    else
        axs[2].plot(1:len_boscia, df_boscia[!,"lowerBound"], label="", color=cb_rose, marker=markers[1], linestyle="dashed", markevery=0.05, linewidth=1.0)
        axs[2].plot(1:len_boscia, df_boscia[!,"upperBound"], label="", color=cb_rose, linestyle="dashdot", markevery=0.05, linewidth=2.0)
    end
    axs[2].plot(1:len_ipopt-1, df_ipopt[1:end-1,"lowerBound"], label="", color=cb_green_sea, marker=markers[2], markevery=0.05, linewidth=1.0)
    if ub_ipopt_int !== nothing
        axs[2].plot(ub_ipopt_int:len_ipopt-1, df_ipopt[ub_ipopt_int:end-1,"upperBound"], label="", color=cb_green_sea, linestyle="dotted", markevery=0.05, linewidth=2.0)
    end
    axs[2].set(xlabel="Numder of nodes", ylabel ="Objective value")
    axs[2].grid()

    lgd = fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), fontsize=12,   # bbox_to_anchor=(0.5, 0.05)
            fancybox=true, shadow=false, ncol=6)
    
    fig.tight_layout()

    savefig(file_name,  bbox_extra_artists=(lgd,), bbox_inches="tight")
end
#plot_boscia_vs_ipopt("neos5", seed=1, num_v=6)
#plot_boscia_vs_strong_convexity("neos5", seed=3, num_v=5)
#plot_boscia_vs_ipopt("neos5", seed=3, num_v=8)
#plot_boscia_vs_strong_convexity("neos5", seed=3, num_v=8)

#for seed in 1:3
    #for num_v in 5:8
        #plot_boscia_vs_ipopt("neos5", seed=seed, num_v=num_v)
        #plot_boscia_vs_strong_convexity("neos5", seed=seed, num_v=num_v)
    #end
#end