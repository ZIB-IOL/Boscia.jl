using PyPlot 
using CSV
using DataFrames

# depending on which folder your current location is
# you might have to add "experiment/.." everywhere

function dual_gap_plot(example, mode)
    # example = "low_dim_high_dim" #"lasso" #"int_sparse_reg" 
    # mode = "time"

    if example == "lasso"
        dim_seed = "_20_28"
    elseif example == "int_sparse_reg"
        dim_seed = "_40_30"
    elseif example == "low_dim_high_dim"
        dim_seed = "_20_2"
    end

    if example == "lasso"
        switch = 9
    elseif example == "int_sparse_reg"
        switch = 45
    elseif example == "low_dim_high_dim"
        switch = 28
    end

    file_name1 = "csv/dual_gap_"* example * dim_seed *"_hybrid_strong_branching.csv"
    file_name2 = "csv/dual_gap_"* example * dim_seed *"_most_infeasible.csv"
    file_name3 = "csv/dual_gap_"* example * dim_seed *"_strong_branching.csv"

    df1 = DataFrame(CSV.File(file_name1))
    df2 = DataFrame(CSV.File(file_name2))
    df3 = DataFrame(CSV.File(file_name3))

    len1 = length(df1[!,"lb"])
    len2 = length(df2[!,"lb"])
    len3 = length(df3[!,"lb"])
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
        ax.plot(1:len1, df1[1:len1,"lb"], label="Hybrid", color=colors[1], marker=markers[1], markevery=0.05) 
        ax.plot(1:len2, df2[1:len2,"lb"], label="Most \ninfeasible", color=colors[end], marker=markers[2], markevery=0.05)
        ax.plot(1:len3, df3[1:len3,"lb"], label="Strong", color=colors[4], marker=markers[3], markevery=0.05)
        ax.set(xlabel="Number of nodes", ylabel="Lower bound")

        if example == "int_sparse_reg" 
            ax.vlines(x=[number_of_nodes[switch]], ymin=1620, ymax=1720,  label = "Switch", color=colors[7], linestyle = :dotted)
        elseif example == "lasso"
            ax.vlines(x=[number_of_nodes[switch]], ymin=210, ymax=260,  label = "Switch", color=colors[7], linestyle = :dotted)
        elseif example == "low_dim_high_dim"
            ax.vlines(x=[number_of_nodes[switch]], ymin=-5030, ymax=-4950,  label = "Switch", color=colors[7], linestyle = :dotted)
        end

    elseif mode == "time"
        ax.plot(df1[1:len1,"list_time"], df1[1:len1,"lb"], label="Hybrid", color=colors[1], marker=markers[1], markevery=0.05) 
        ax.plot(df2[1:len2,"list_time"], df2[1:len2,"lb"], label="Most \ninfeasible", color=colors[end], marker=markers[2], markevery=0.05)
        ax.plot(df3[1:len3,"list_time"], df3[1:len3,"lb"], label="Strong", color=colors[4], marker=markers[3], markevery=0.05)
        ax.set(xlabel="Time (s)", ylabel="Lower bound")

        if example == "int_sparse_reg" 
            ax.vlines(x=[df1[1:len, "list_time"][switch]], ymin=1620, ymax=1720,  label = "Switch", color=colors[7], linestyle = :dotted)
        elseif example == "lasso"
            ax.vlines(x=[df1[1:len, "list_time"][switch]], ymin=210, ymax=260,  label = "Switch", color=colors[7], linestyle = :dotted)
        elseif example == "low_dim_high_dim"
            ax.vlines(x=[df1[1:len, "list_time"][switch]], ymin=-5030, ymax=-4950,  label = "Switch", color=colors[7], linestyle = :dotted)
        end
    end

    ax.grid()
    lgd = fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.05), fontsize=12,
            fancybox=true, shadow=false, ncol=4)
    fig.tight_layout()

    file_name = "images/dual_gap_" * mode * "_" * example * "_" * dim_seed * ".pdf" #"images/different_branching_strategies_time.pdf"
    savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches="tight")

end