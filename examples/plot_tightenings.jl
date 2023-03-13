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

    ax.plot(1:length(df_boscia[!,"openNodes"]), df_boscia[!,"openNodes"], label="BO (ours)", color=colors[1], marker=markers[1], markevery=0.1, alpha=.5)
    ax.plot(1:length(df_global_tightening[!,"openNodes"]), df_global_tightening[!,"openNodes"], label="global tightening", color=colors[end], marker=markers[2], markevery=0.1, alpha=.5)
    ax.plot(1:length(df_local_tightening[!,"openNodes"]), df_local_tightening[!,"openNodes"], label="local tightening", color=colors[2], marker=markers[3], markevery=0.1, alpha=.5)
    ax.plot(1:length(df_no_tightening[!,"openNodes"]), df_no_tightening[!,"openNodes"], label="no tightening", color=colors[3], marker=markers[4], markevery=0.1, alpha=.5)

    ylabel("open nodes")
    #locator_params(axis="y", nbins=4)
    xlabel("iteration")
    ax.grid()

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12,
        fancybox=true, shadow=false, ncol=2)
    fig.tight_layout()
    file = joinpath(@__DIR__, "csv/" * example * "_" * setup * "_tightenings.pdf")
    savefig(file)
end
