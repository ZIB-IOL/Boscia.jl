using PyPlot
using DataFrames
using CSV

function plot_boscia_vs_scip(mode)
    if mode == "integer"
        # load file
        df = DataFrame(CSV.File("examples/csv/boscia_vs_scip_int.csv"))
    elseif mode == "mixed"
        df = DataFrame(CSV.File("examples/csv/boscia_vs_scip_mixed.csv"))
    end

    # display(df)

    indices = [index for index in 1:nrow(df) if isodd(index)]
    df_boscia = copy(df)
    df_scip = copy(df)
    delete!(df_boscia, indices) 
    delete!(df_scip, indices) 

    # display(df_scip)

    df_scip = filter(row -> !(row.termination_scip == "TIME_LIMIT"),  df_scip)
    #df_boscia = filter(row -> !(row.termination_boscia == "Time limit reached"),  df_boscia)
    df_boscia = filter(row -> !(row.time_boscia >= 1800),  df_boscia)
    #df_boscia = filter(row -> !(row.termination_scip == "TIME_LIMIT" && isapprox(row.solution_boscia, row.solution_scip)),  df_boscia)

    # display(df_scip)
    display(df_boscia)

    colors = ["b", "m", "c", "r", "g", "y", "k"]
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
    #ax.scatter(df_temp[!,"dimension"], df_temp[!,"time_boscia"], label="Boscia", color=colors[1], marker=markers[1])
    #ax.scatter(df_temp[!,"dimension"], df_temp[!,"time_scip"], label="SCIP", color=colors[3], marker=markers[2])
    ax.plot(sort(df_boscia[!,"time_boscia"]), 1:nrow(df_boscia), label="Boscia", color=colors[1], marker=markers[1])
    ax.plot(sort(df_scip[!,"time_scip"]), 1:nrow(df_scip), label="SCIP", color=colors[3], marker=markers[2])

    yticks(1:nrow(df_boscia),[])
    ylabel("Solved instances")
    xlabel("Time in s")
    ax.set_xscale("log")
    ax.grid()
    fig.legend(loc=7, fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)  

    if mode == "integer"
        file = ("examples/csv/boscia_vs_scip.png")
    elseif mode == "mixed"
        file = ("examples/csv/boscia_vs_scip_mixed.png")
    end
        
    savefig(file)
end