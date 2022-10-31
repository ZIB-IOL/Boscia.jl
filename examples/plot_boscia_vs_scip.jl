using PyPlot
using DataFrames
using CSV

function plot_boscia_vs_scip(mode; afw=false)
    if mode == "integer"
        # load file
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_vs_scip_int.csv")))
    elseif mode == "mixed"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_vs_scip_mixed.csv")))
    elseif mode == "mixed_lowdim"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_vs_scip_mixed_lowdim.csv")))
    elseif mode == "mixed_50"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_vs_scip_mixed_50.csv")))
    elseif mode == "integer_50"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_vs_scip_integer_50.csv")))
    else
        error("wrong option")
    end

    # display(df)

    indices = [index for index in 1:nrow(df) if isodd(index)]
    df_boscia = copy(df)
    df_scip = copy(df)
    delete!(df_boscia, indices)
    delete!(df_scip, indices)

    # display(df_scip)
    time_limit = 1800
    filter!(row -> !(row.termination_scip == "TIME_LIMIT"),  df_scip)
    #filter!(row -> !(row.termination_scip == "OPTIMIZE_NOT_CALLED"),  df_boscia)
    #df_boscia = filter(row -> !(row.termination_boscia == "Time limit reached"),  df_boscia)
    filter!(row -> !(row.time_boscia >= time_limit),  df_boscia)
    #df_boscia = filter(row -> !(row.termination_scip == "TIME_LIMIT" && isapprox(row.solution_boscia, row.solution_scip)),  df_boscia)

    # display(df_scip)
    # display(df_boscia)

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
    #ax.scatter(df_temp[!,"dimension"], df_temp[!,"time_boscia"], label="BO (ours)", color=colors[1], marker=markers[1])
    #ax.scatter(df_temp[!,"dimension"], df_temp[!,"time_scip"], label="SCIP", color=colors[3], marker=markers[2])
    time_boscia = sort(df_boscia[!,"time_boscia"])
    time_scip = sort(df_scip[!,"time_scip"])
    push!(time_boscia, 1.1 * time_limit)
    push!(time_scip, 1.1 * time_limit)
    ax.plot(time_boscia, [1:nrow(df_boscia); nrow(df_boscia)], label="BO (ours)", color=colors[1], marker=markers[1])
    ax.plot(time_scip, [1:nrow(df_scip); nrow(df_scip)], label="SCIP+OA", color=colors[end], marker=markers[2])
    #yticks(0:2:nrow(df_boscia)+1, 0:2:nrow(df_boscia)+1)
    
    if afw && mode == "mixed_50"
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_mixed_50.csv")))
        filter!(row -> !(row.time_afw >= time_limit),  df_afw)
        time_afw = sort(df_afw[!,"time_afw"])
        push!(time_afw, 1.1 * time_limit)
        ax.plot(time_afw, [1:nrow(df_afw); nrow(df_afw)], label="AFW", color=colors[4], marker=markers[3])
        @show nrow(df_afw), nrow(df_boscia)
    end

    if afw && mode == "integer_50"
        df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_integer_50.csv")))
        indices = [index for index in 1:nrow(df_afw) if isodd(index)]
        delete!(df_afw, indices)
        filter!(row -> !(row.time_afw >= time_limit),  df_afw)
        time_afw = sort(df_afw[!,"time_afw"])
        push!(time_afw, 1.1 * time_limit)
        ax.plot(time_afw, [1:nrow(df_afw); nrow(df_afw)], label="AFW", color=colors[4], marker=markers[3])
        @show nrow(df_afw), nrow(df_boscia)
    end

    ylabel("Solved instances")
    xlabel("Time (s)")
    ax.set_xscale("log")
    ax.grid()
    if mode == "integer" || mode == "integer_50"
        title("Pure-integer portfolio problem", loc="center")
    else
        title("Mixed-integer portfolio problem", loc="center")
    end
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12,
        fancybox=true, shadow=false, ncol=2)
    if afw
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12,
        fancybox=true, shadow=false, ncol=3)
    end
    fig.tight_layout()
    if mode == "integer"
        file = ("examples/csv/boscia_vs_scip.pdf")
    elseif mode == "mixed"
        file = ("examples/csv/boscia_vs_scip_mixed.pdf")
    elseif mode == "mixed_lowdim"
        file = ("examples/csv/boscia_vs_scip_mixed_lowdim.pdf")
    elseif mode == "mixed_50"
        file = ("examples/csv/boscia_vs_scip_mixed_50.pdf")
    elseif mode == "integer_50"
        file = ("examples/csv/boscia_vs_scip_integer_50.pdf")
    end

    savefig(file)
end
