using PyPlot 
using CSV
using DataFrames

# depending on which folder your current location is
# you might have to add "experiment/.." everywhere

function dual_gap_plot(example, seed, dim, mode)
    # example = "low_dim_high_dim" #"lasso" #"int_sparse_reg" 
    # mode = "time"

    if example == "sparse_reg"
        switch = floor(5*dim/20)
    else
        switch = floor(dim/20)
    end

    if example == "sparse_reg"
        start= "boscia_"
    else
        start = ""
    end
    file_name1 = joinpath(@__DIR__, "csv/" * start * "hybrid_branching_" * string(dim) * "_" * string(seed) * "_" * example * ".csv")
    file_name2 = joinpath(@__DIR__, "csv/" * start * "default_" * string(dim) * "_" * string(seed) * "_" * example * ".csv")
    file_name3 = joinpath(@__DIR__, "csv/" * start * "strong_branching_" * string(dim) * "_" * string(seed) * "_" * example * ".csv")

    try
    df1 = DataFrame(CSV.File(file_name1))
    df2 = DataFrame(CSV.File(file_name2))
    df3 = DataFrame(CSV.File(file_name3))

    len1 = length(df1[!,"lowerBound"])
    len2 = length(df2[!,"lowerBound"])
    len3 = length(df3[!,"lowerBound"])
    len = min(len1, min(len2, len3))

    colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
    markers = ["o", "s", "^", "P", "X", "H", "D"]

    fig = plt.figure(figsize=(7,3.5))
    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("font", size=12, family="cursive")
    PyPlot.matplotlib[:rc]("axes", labelsize=14)
    PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
    \usepackage{libertine}
    \usepackage{libertinust1math}
    """)
    ax = fig.add_subplot(111)

    number_of_nodes = [i for i in 1:len]

    if mode == "nodes"
        ax.plot(1:len1, df1[1:len1,"lowerBound"], label="Hybrid", color=colors[1], marker=markers[1], markevery=0.05) 
        ax.plot(1:len2, df2[1:len2,"lowerBound"], label="Most \ninfeasible", color=colors[end], marker=markers[2], markevery=0.05)
        ax.plot(1:len3, df3[1:len3,"lowerBound"], label="Strong", color=colors[4], marker=markers[3], markevery=0.05)
        ax.set(xlabel="Number of nodes", ylabel="Lower bound")

        ymin = minimum(vcat(df1[!,:lowerBound], df2[!,:lowerBound], df3[!,:lowerBound])) 
        ymax = maximum(vcat(df1[!,:lowerBound], df2[!,:lowerBound], df3[!,:lowerBound])) 

        ax.vlines(x=switch, ymin=ymin, ymax=ymax,  label = "Switch", color=colors[7], linestyle = :dotted)

       #= if example == "int_sparse_reg" 
            ax.vlines(x=switch, ymin=1620, ymax=1720,  label = "Switch", color=colors[7], linestyle = :dotted)
        elseif example == "lasso"
            ax.vlines(x=[number_of_nodes[switch]], ymin=210, ymax=260,  label = "Switch", color=colors[7], linestyle = :dotted)
        elseif example == "low_dim_high_dim"
            ax.vlines(x=[number_of_nodes[switch]], ymin=-5030, ymax=-4950,  label = "Switch", color=colors[7], linestyle = :dotted)
        end =#

    elseif mode == "time"
        ax.plot(df1[1:len1,"time"], df1[1:len1,"lowerBound"], label="Hybrid", color=colors[1], marker=markers[1], markevery=0.05) 
        ax.plot(df2[1:len2,"time"], df2[1:len2,"lowerBound"], label="Most \ninfeasible", color=colors[end], marker=markers[2], markevery=0.05)
        ax.plot(df3[1:len3,"time"], df3[1:len3,"lowerBound"], label="Strong", color=colors[4], marker=markers[3], markevery=0.05)
        ax.set(xlabel="Time (s)", ylabel="Lower bound")

        ymin = minimum(vcat(df1[!,:lowerBound], df2[!,:lowerBound], df3[!,:lowerBound])) 
        ymax = maximum(vcat(df1[!,:lowerBound], df2[!,:lowerBound], df3[!,:lowerBound])) 

        switch = convert(Int64, switch)

        ax.vlines(df2[switch, :time], ymin=ymin, ymax=ymax,  label = "Switch", color=colors[7], linestyle = :dotted)

        #=if example == "int_sparse_reg" 
            ax.vlines(x=[df1[1:len, "list_time"][switch]], ymin=1620, ymax=1720,  label = "Switch", color=colors[7], linestyle = :dotted)
        elseif example == "lasso"
            ax.vlines(x=[df1[1:len, "list_time"][switch]], ymin=210, ymax=260,  label = "Switch", color=colors[7], linestyle = :dotted)
        elseif example == "low_dim_high_dim"
            ax.vlines(x=[df1[1:len, "list_time"][switch]], ymin=-5030, ymax=-4950,  label = "Switch", color=colors[7], linestyle = :dotted)
        end =#
    end

    ax.grid()
    lgd = fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.05), fontsize=12,
            fancybox=true, shadow=false, ncol=4)
    fig.tight_layout()

    file_name = joinpath(@__DIR__, "plots/progress_plots/" * example * "/branching_" * mode * "_" * example * "_" * string(dim) * "_" * string(seed) * ".pdf")
    #file_name = "images/dual_gap_" * mode * "_" * example * "_" * dim_seed * ".pdf" #"images/different_branching_strategies_time.pdf"
    savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches="tight")
catch e
    println(e)
end
end

## sparse regression
# "no_tightening", "local_tigtening", "global_tightening", hybrid_branching_20", "strong_branching"
example = "sparse_reg"
modes = ["nodes", "time"]
for mode in modes
    for m in 15:30
        for seed in 1:10
            dual_gap_plot(example, seed, m, mode)
        end
    end
end

# portfolio mixed
example = "mixed_portfolio"
for mode in modes
    for m in 20:5:120
        for seed in 1:10
            dual_gap_plot(example, seed, m, mode)
        end
    end
end

# portfolio integer
example = "integer_portfolio"
for mode in modes
    for m in 20:5:120
        for seed in 1:10
            dual_gap_plot(example, seed, m, mode)
        end
    end
end
