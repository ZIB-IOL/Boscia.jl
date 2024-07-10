# using Plots 
# pyplot()
using PyPlot
using CSV
using DataFrames

function dual_gap_plot(file_name, mode)
    #println(file_name)
    try 
    df = DataFrame(CSV.File(file_name))

    example = if occursin("sqr_dst", file_name)
        "sqr_dst"
    elseif occursin("sparse_reg", file_name)
        "sparse_reg"
    elseif occursin("worst_case", file_name)
        "worst_case"
    elseif occursin("birkhoff", file_name)
       "birkhoff"
    elseif occursin("low_dim", file_name)
        "low_dim"
    elseif occursin("lasso", file_name)
        "lasso"
    elseif occursin("mixed_portfolio", file_name)
        "mixed_portfolio"
    elseif occursin("integer_portfolio", file_name)
        "integer_portfolio"
    elseif occursin("poisson", file_name)
       "poisson"
    elseif occursin("sparse_log_reg", file_name)
       "sparse_log_reg"
    end

    #=data = replace(file_name, "examples/csv/early_stopping_" * example * "_" => "")
    next_index = findfirst("_", data)
    dimension = data[1:next_index[1]-1]
    data = data[next_index[1]+1:end] 
    next_index = findfirst("_", data)
    seed = data[1:next_index[1]-1]
    data = data[next_index[1]+1:end] 
    next_index = findfirst("_", data)
    min_number_lower = data[1:next_index[1]-1]
    data = data[next_index[1]+1:end] 
    next_index = findfirst("_", data)
    fw_dual_gap_precision = data[1:next_index[1]-1] =#

    #=if example == "sqr_dst"
        title = "Example : squared distance, Dimension : " * string(dimension) * "\n min_number_lower=" * string(min_number_lower) * ", fw_dual_gap_precision=" * string(fw_dual_gap_precision)
    elseif example == "sparse_reg"
        title = "Example : sparse regression, Dimension : " * string(dimension) * "\n min_number_lower=" * string(min_number_lower) * ", fw_dual_gap_precision=" * string(fw_dual_gap_precision)
    elseif example == "worst_case"
        title = "Example : worst case, Dimension : " * string(dimension) * "\n min_number_lower=" * string(min_number_lower) * ", fw_dual_gap_precision=" * string(fw_dual_gap_precision)
    end =#

    if !("ub" in names(df))
        df[!,"ub"] = df[!,"upperBound"]
        df[!,"lb"] = df[!,"lowerBound"]
        df[!,"list_lmo_calls"] = df[!,"LMOcalls"]
    end
    len = length(df[!,"ub"])

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
    lns1 = ax.plot(1:len, df[!,"ub"], label="Incumbent", color=colors[end], marker=markers[2], markevery=0.05)
    lns2 = ax.plot(1:len, df[!,"lb"], label="Lower bound", color=colors[4], marker=markers[3], markevery=0.05)
    #ax.plot(1:length(df2[!,"ub"]), df2[!,"lb"], label="Lower bound, adaptive gap = 0.65", color=colors[4], linestyle="dashed")
    xlabel("Number of nodes")
    #xticks(range(1, len, step=50))
    ylabel("Objective value")

    #ax.set_ylim(bottom=2.5, top=3.5)
    # ax.set_ylim(bottom=-5500, top=-4900)

    ax2 = ax.twinx()
    lns3 = ax2.plot(1:len, df[!, "list_lmo_calls"], label="Total BLMO calls", color=colors[1], marker=markers[1], markevery=0.05)
    ylabel("BLMO calls", Dict("color"=>colors[1]))
    setp(ax2.get_yticklabels(),color=colors[1])

    #PyPlot.title(title)
    ax.grid()
    #fig.legend(loc=7, fontsize=12)
    lgd = fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), fontsize=12,
            fancybox=true, shadow=false, ncol=3)
    fig.tight_layout()
    #fig.subplots_adjust(right=0.6) 

    # plot(1:len, df[!,"ub"], label="upper bound", title=title, xlabel="number of iterations", legend=:topleft, rightmargin = 2.5Plots.cm,)
    # plot!(1:len, df[!,"lb"], label="lower bound")
    # plot!(twinx(), df[!, "list_lmo_calls"], label="#lmo calls", color=:green)
    #yaxis!("objective value")

    file_name = replace(file_name, ".csv" => ".pdf")
    file_name = replace(file_name, "csv/" => "plots/progress_plots/" * example * "/")
    if example == "sparse_reg"
        file_name = replace(file_name, "boscia_" => "dual_gap_")
    else
        file_name = replace(file_name, mode => "dual_gap_" * mode)
    end

    savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches="tight")
catch e
    println(e)
    return
end
end

## sparse regression
modes = ["default"] # "no_tightening", "local_tigtening", "global_tightening", hybrid_branching_20", "strong_branching"

for mode in modes
    for m in [19]#15:30
        for seed in [3]#1:10
            file = joinpath(@__DIR__, "csv/boscia_" * mode * "_" * string(m) * "_" * string(seed) * "_sparse_reg.csv")
            dual_gap_plot(file, mode)
        end
    end
end

# portfolio mixed
for mode in modes
    for m in [45]#20:5:120
        for seed in [1]#1:10
            file = joinpath(@__DIR__, "csv/" * mode * "_" * string(m) * "_" * string(seed) * "_mixed_portfolio.csv")
            dual_gap_plot(file, mode)
        end
    end
end

# portfolio integer
for mode in modes
    for m in [35]#20:5:120
        for seed in [10]#1:10
            file = joinpath(@__DIR__, "csv/" * mode * "_" * string(m) * "_" * string(seed) * "_integer_portfolio.csv")
            dual_gap_plot(file, mode)
        end
    end
end

# sparse log regression
mode = "default"
for dimension in [15]#[5:5:20;]
    for seed in [7,9]#1:2:10
        for M in [0.1,1]
            for var_A in [1,5]
                file = joinpath(@__DIR__, "csv/" * mode * "_" * "sparse_log_regression_" * string(dimension) * "_" * string(M) * "-" * string(var_A) * "_" * string(seed) * ".csv")
                dual_gap_plot(file, mode)
            end
        end
    end
end

# poisson
mode = "default"
for dimension in [70]#[50:20:100;]
    for seed in [1]#[1,5,10]#1:10
        for ns in [1.0]#[0.1,1,5,10]
            file = joinpath(@__DIR__, "csv/" * mode * "_" * "poisson_" * string(dimension) * "_" * string(ns) * "-" * string(dimension) * "_" * string(floor(dimension/2)) * "_" * string(seed) * ".csv")
            dual_gap_plot(file, mode)
        end
    end
end

