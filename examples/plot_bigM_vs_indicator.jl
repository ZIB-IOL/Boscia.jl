using PyPlot
using DataFrames
using CSV

function plot_bigM_vs_indicator(mode; dim = 70, factor = 5, seed = 1)
    lmo_calls = false
    file_name_ind = joinpath(@__DIR__, "csv/bigM_vs_indicator_" * mode * "_indicator_" * string(dim) * "_" * string(factor) * "_" * string(seed) * ".csv")
    file_name_M = joinpath(@__DIR__, "csv/bigM_vs_indicator_" * mode * "_bigM_" * string(dim) * "_" * string(factor) * "_" * string(seed) * ".csv")

    if mode == "lasso"
        example = "Lasso"
    elseif mode == "sparse_reg"
        example = "Sparse regression"
    elseif mode == "poisson_reg"
        example = "Sparse poisson regression"
    elseif mode == "int_reg"
        example = " Integer sparse regression"
    end

    df_ind = DataFrame(CSV.File(file_name_ind))
    df_M = DataFrame(CSV.File(file_name_M))

   #= if time_plot && lmo_calls
        file_name = joinpath(@__DIR__, "images/BigM_vs_Indicator/bigM_vs_indicator_" * mode * "_time_accLMO" * string(dim) * "_" * string(factor) * "_" * string(seed) * ".csv")
    elseif time_plot
        joinpath(@__DIR__, "images/BigM_vs_Indicator/bigM_vs_indicator_" * mode * "_time_" * string(dim) * "_" * string(factor) * "_" * string(seed) * ".csv")
    elseif lmo_calls
        joinpath(@__DIR__, "images/BigM_vs_Indicator/bigM_vs_indicator_" * mode * "_num_nodes_accLMO" * string(dim) * "_" * string(factor) * "_" * string(seed) * ".csv")
    else
        joinpath(@__DIR__, "images/BigM_vs_Indicator/bigM_vs_indicator_" * mode * "_num_nodes_" * string(dim) * "_" * string(factor) * "_" * string(seed) * ".csv") 
    end=#


    file_name = joinpath(@__DIR__, "images/bigM_vs_indicator_" * mode * string(dim) * "_" * string(factor) * "_" * string(seed) * ".csv") 
    file_name = replace(file_name, ".csv" => ".pdf")
    @show file_name

    colors = ["b", "m", "c", "r", "g", "y", "k"]
    markers = ["o", "s", "^", "P", "X", "H", "D"]
    
    #fig = plt.figure(figsize=(7.3, 5))
    fig, axs = plt.subplots(2, sharex=false)
    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("font", size=12, family="cursive")
    PyPlot.matplotlib[:rc]("axes", labelsize=14)
    PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
    \usepackage{libertine}
    \usepackage{libertinust1math}
    """)
    #ax = fig.add_subplot(111)

    df_ind[!,"list_time"] ./= 1000
    df_M[!,"list_time"] ./= 1000

    len_ind = length(df_ind[!, "lb"])
    len_M = length(df_M[!, "lb"])

    if len_ind == 1
        axs[1].plot(df_ind[!,"list_time"], df_ind[!,"lb"], label="Indicator LB", color=colors[1], marker=markers[1], linestyle="dashed", linewidth=1.0)
        axs[1].plot(df_ind[!,"list_time"], df_ind[!,"ub"], label="Indicator UB", color=colors[1], linestyle="dashdot", linewidth=3.0)
    else
        axs[1].plot(df_ind[!,"list_time"], df_ind[!,"lb"], label="Indicator LB", color=colors[1], marker=markers[1], linestyle="dashed", markevery=0.05, linewidth=1.0)
        axs[1].plot(df_ind[!,"list_time"], df_ind[!,"ub"], label="Indicator UB", color=colors[1], linestyle="dashdot", markevery=0.05, linewidth=2.0)
    end
    if len_M == 1
        axs[1].plot(df_M[!,"list_time"], df_M[!,"lb"], label="Big M LB", color=colors[2], marker=markers[2], linewidth=1.0)
        axs[1].plot(df_M[!,"list_time"], df_M[!,"ub"], label="Big M UB", color=colors[2], linestyle="dotted", linewidth=3.0)
    else
        axs[1].plot(df_M[!,"list_time"], df_M[!,"lb"], label="Big M LB", color=colors[2], marker=markers[2], markevery=0.05, linewidth=1.0)
        axs[1].plot(df_M[!,"list_time"], df_M[!,"ub"], label="Big M UB", color=colors[2], linestyle="dotted", markevery=0.05, linewidth=2.0)    
    end

    axs[1].set(xlabel="Time (s)", ylabel ="Lower bound")
    axs[1].grid()


    if len_ind == 1
        axs[2].plot(1:len_ind, df_ind[!,"lb"], label="", color=colors[1], marker=markers[1], linestyle="dashed", linewidth=1.0)
        axs[2].plot(1:len_ind, df_ind[!,"ub"], label="", color=colors[1], linestyle="dashdot", linewidth=3.0)
    else
        axs[2].plot(1:len_ind, df_ind[!,"lb"], label="", color=colors[1], marker=markers[1], linestyle="dashed", markevery=0.05, linewidth=1.0)
        axs[2].plot(1:len_ind, df_ind[!,"ub"], label="", color=colors[1], linestyle="dashdot", markevery=0.05, linewidth=2.0)
    end
    if len_M == 1
        axs[2].plot(1:len_M, df_M[!,"lb"], label="", color=colors[2], marker=markers[2], linewidth=1.0)
        axs[2].plot(1:len_M, df_M[!,"ub"], label="", color=colors[2], linestyle="dotted", linewidth=3.0)
    else 
        axs[2].plot(1:len_M, df_M[!,"lb"], label="", color=colors[2], marker=markers[2], markevery=0.05, linewidth=1.0)
        axs[2].plot(1:len_M, df_M[!,"ub"], label="", color=colors[2], linestyle="dotted", markevery=0.05, linewidth=2.0)
    end
  
    axs[2].set(xlabel="Numder of nodes", ylabel ="Lower bound")
    axs[2].grid()
 

    lgd = fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), fontsize=11,   # bbox_to_anchor=(0.5, 0.05)
            fancybox=true, shadow=false, ncol=6)
    
    fig.tight_layout()

    savefig(file_name,  bbox_extra_artists=(lgd,), bbox_inches="tight")
end
