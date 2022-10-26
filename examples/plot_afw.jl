using PyPlot
using JSON

function plot_baseline_vs_afw(seed, mode)
    if seed == "5177894854221866464"
        # load file
        json_file = JSON.parsefile(joinpath(@__DIR__, "results_portfolio_afw_5177894854221866464.json"))
    else
        error("wrong option")
    end

    result_baseline = json_file["result_baseline"]
    result_afw = json_file["result_afw"]

    #@show result_baseline["list_time"]

    colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
    markers = ["o", "s", "^", "P", "X", "H", "D"]

    fig = plt.figure(figsize=(6.5,3.5))
    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("font", size=12, family="cursive")
    PyPlot.matplotlib[:rc]("axes", labelsize=14)
    PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
    \usepackage{libertine}
    \usepackage{libertinust1math}
    """)
    ax = fig.add_subplot(111)

    if mode == "time"
        ax.plot(result_baseline["list_time"]/1000,result_baseline["list_lb"], label="BO", color=colors[1], marker=markers[1], markevery=0.05)
        ax.plot(result_afw["list_time"]/1000, result_afw["list_lb"], label="AFW", color=colors[end], marker=markers[2], markevery=0.05)

        ylabel("Lower bound")
        xlabel("Time (s)")
        #ax.set_xscale("log")
        ax.grid()
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12,
        fancybox=true, shadow=false, ncol=2)
        fig.tight_layout()
    
    else 
        ax.plot(result_baseline["list_lmo_calls_acc"],result_baseline["list_lb"], label="BO", color=colors[1], marker=markers[1], markevery=0.05)
        ax.plot(result_afw["list_lmo_calls_acc"], result_afw["list_lb"], label="AFW", color=colors[end], marker=markers[2], markevery=0.05)

        ylabel("Lower bound")
        xlabel("Number of lmo calls")
        ax.grid()
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12,
        fancybox=true, shadow=false, ncol=2)
        fig.tight_layout()        
    end 

    savefig("afw_" * string(seed) * "_" * mode * ".pdf")
end
