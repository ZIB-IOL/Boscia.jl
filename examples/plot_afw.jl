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

    @show result_baseline["list_time"]

    colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
    markers = ["o", "s", "^", "P", "X", "H", "D"]

    fig = plt.figure(figsize=(6.5,3))
    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("font", size=12, family="cursive")
    PyPlot.matplotlib[:rc]("axes", labelsize=14)
    PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
    \usepackage{libertine}
    \usepackage{libertinust1math}
    """)
    ax = fig.add_subplot(111)

    if mode == "time"
        ax.plot(result_baseline["list_time"]/1000,result_baseline["list_lb"], label="BL", color=colors[1])
        ax.plot(result_afw["list_time"]/1000, result_afw["list_lb"], label="AFW", color=colors[end])

        ylabel("lower bound")
        xlabel("time (s)")
        #ax.set_xscale("log")
        ax.grid()

        fig.legend(loc=(0.07, 0.05), fontsize=10, ncol=2)
        fig.tight_layout()
        fig.subplots_adjust(right=0.75)  
        PyPlot.tight_layout()
    
    else 
        ax.plot(result_baseline["list_lmo_calls_acc"],result_baseline["list_lb"], label="BL", color=colors[1])
        ax.plot(result_afw["list_lmo_calls_acc"], result_afw["list_lb"], label="AFW", color=colors[end])

        ylabel("lower bound")
        xlabel("number of lmo calls")
        #ax.set_xscale("log")
        ax.grid()

        fig.legend(loc=(0.07, 0.05), fontsize=10, ncol=2)
        fig.tight_layout()
        fig.subplots_adjust(right=0.75)  
        PyPlot.tight_layout()        
    end 

    savefig("afw_" * string(seed) * "_" * mode * ".pdf")
end
