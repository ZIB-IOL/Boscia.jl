using PyPlot
using DataFrames
using CSV

function plot(example)
    if example == "sparse_reg"
        df_boscia = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_sparse_reg_30_1.csv")))
        df_global_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/global_tightening_sparse_reg_30_1.csv")))
        df_local_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/local_tightening_sparse_reg_30_1.csv")))
        df_no_tightening = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_tightening_sparse_reg_30_1.csv")))
    end

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

    ax.plot(df_boscia[!,"openNodes"], 1:length(df_boscia[!,"openNodes"]), label="BO (ours)", color=colors[1], marker=markers[1], markevery=0.1)
    ax.plot(df_global_tightening[!,"openNodes"], 1:length(df_global_tightening[!,"openNodes"]), label="global tightening", color=colors[end], marker=markers[2], markevery=0.1)
    ax.plot(df_local_tightening[!,"openNodes"], 1:length(df_local_tightening[!,"openNodes"]), label="local tightening", color=colors[2], marker=markers[3], markevery=0.1)
    ax.plot(df_no_tightening[!,"openNodes"], 1:length(df_no_tightening[!,"openNodes"]), label="no tightening", color=colors[3], marker=markers[4], markevery=0.1)

    ylabel("open nodes")
    #locator_params(axis="y", nbins=4)
    xlabel("iteration")
    ax.grid()

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12,
        fancybox=true, shadow=false, ncol=2)
    fig.tight_layout()
    file = joinpath(@__DIR__, "csv/" * example * "_tightenings.pdf")
    savefig(file)
end
