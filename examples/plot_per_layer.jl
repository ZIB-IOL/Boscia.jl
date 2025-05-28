# using Plots 
# pyplot()
using CSV
using DataFrames
using Statistics
using PyPlot

function per_layer_plot(file_name, mode)
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
    

    # data = replace(file_name, "experiments/csv/" * example * "_per_layer_" => "")
    # next_index = findfirst("_", data)
    # dimension = data[1:next_index[1]-1]
    # data = data[next_index[1]+1:end] 
    # next_index = findfirst("_", data)
    # if example == "birkhoff"
    #     k =  data[1:next_index[1]-1]
    #     data = data[next_index[1]+1:end] 
    #     next_index = findfirst("_", data)
    # end
    # seed = data[1:next_index[1]-1]
    # data = data[next_index[1]+1:end] 
    # next_index = findfirst("_", data)
    # min_number_lower = data[1:next_index[1]-1]
    # data = data[next_index[1]+1:end] 
    # next_index = findfirst("_", data)
    # fw_dual_gap_precision = data[1:next_index[1]-1]
    # data = data[next_index[1]+1:end] 
    # next_index = findfirst(".csv", data)
    # dual_gap_decay_factor = data[1:next_index[1]-1] 

    # if example == "sqr_dst"
    #     title = "Example : squared distance, Dimension : " * string(dimension) * "\n min_number_lower=" * string(min_number_lower) * ", dual_gap_decay_factor=" * string(dual_gap_decay_factor)
    # elseif example == "sparse_reg"
    #     title = "Example : sparse regression, Dimension : " * string(dimension) * "\n min_number_lower=" * string(min_number_lower) * ", dual_gap_decay_factor=" * string(dual_gap_decay_factor)
    # elseif example == "worst_case"
    #     title = "Example : worst case, Dimension : " * string(dimension) * "\n min_number_lower=" * string(min_number_lower) * ", dual_gap_decay_factor=" * string(dual_gap_decay_factor)
    # elseif example == "birkhoff"
    #     title = "Example : birkhoff, n : " * string(dimension) * ", k : " * string(k) * "\n min_number_lower=" * string(min_number_lower) * ", dual_gap_decay_factor=" * string(dual_gap_decay_factor)
    # end

    if !("lmo_calls" in names(df))
        #df[!,"LMOcalls"] = df[!, "list_lmo_calls"]
        lmo_calls_per_layer = df[!,:LMOcalls]
        active_set_size = df[!,:list_active_set_size_cb]
        discarded_set_size = df[!,:list_discarded_set_size_cb]
    else 
        # parse to array
        lmo_calls_per_layer = df[!,:lmo_calls]
        lmo_calls_per_layer = [parse.(Int, split(chop(i; head=1, tail=1), ',')) for i in lmo_calls_per_layer]
        active_set_size = df[!,:active_set_size]
        active_set_size = [parse.(Int, split(chop(i; head=1, tail=1), ',')) for i in active_set_size]
        discarded_set_size = df[!,:discarded_set_size]
        discarded_set_size = [parse.(Int, split(chop(i; head=1, tail=1), ',')) for i in discarded_set_size]
    end

    lmo_calls_per_layer_mean = [sum(i) for i in lmo_calls_per_layer]
    active_set_size_per_layer_mean = [mean(i) for i in active_set_size]
    discarded_set_size_per_layer_mean = [mean(i) for i in discarded_set_size]
    # @show lmo_calls_per_layer

    len = length(lmo_calls_per_layer_mean)

    colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
    markers = ["o", "s", "^", "P", "X", "H", "D"]

    fig = plt.figure(figsize=(6.5,3.5))
    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("font", size=14, family="cursive")
    PyPlot.matplotlib[:rc]("axes", labelsize=14)
    PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
    \usepackage{newtxtext}
    """)

    ax = fig.add_subplot(111)
    ax.plot(1:len, active_set_size_per_layer_mean, label="Active set", color=cb_clay, marker=markers[2])
    ylabel("Avg set size")
    ax.plot(1:len, discarded_set_size_per_layer_mean, label="Discarded set", color=cb_green_sea, marker=markers[3])

    ax2 = ax.twinx()
    ax2.plot(1:len, lmo_calls_per_layer_mean, label="BLMO calls", color=cb_lilac, marker=markers[1])
    xticks(range(1, len, step=5))
    ylabel("BLMO calls", Dict("color"=>cb_lilac))
    setp(ax2.get_yticklabels(),color=cb_lilac)
    ax.set(xlabel="Node depth")

    #ax.plot(1:len, lmo_calls_per_layer_std_below, color=:blue, linestyle="dashed")
    #ax.plot(1:len, lmo_calls_per_layer_std_above, color=:blue, linestyle="dashed")
    # add lmo calls for each node
    # for depth in 1:len
    #     ax.scatter([depth for i in 1:length(lmo_calls_per_layer[depth])], lmo_calls_per_layer[depth], color=:blue, s=1)
    # end

    #PyPlot.title(title)
    ax.grid()
    lgd = fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), fontsize=14,
            fancybox=true, shadow=false, ncol=3)
    fig.tight_layout()

    # plot(1:len, lmo_calls_per_layer, label="lmo calls", title=title, xlabel="node depth", ylabel="lmo calls", legend=:outerright, leftmargin = 1.5Plots.cm, rightmargin = 2.5Plots.cm,)
    # x = twinx()
    # plot!(x, active_set_size_per_layer, label="active set", color=:red, ylabel="set size")
    # plot!(x, discarded_set_size_per_layer, label="discarded set", color=:green)

    println("figure created")

    #=file_name = replace(file_name, ".csv" => ".pdf")
    file_name = replace(file_name, "csv" => "plots/progress_plots/" * example * "/")
    if example == "sparse_reg"
        file_name = replace(file_name, "boscia_" => "size_active_set_")
    else
        file_name = replace(file_name, mode => "size_active_set_" * mode)
    end
    #@show file_name
    savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches="tight")=#


    file_name = replace(file_name, ".csv" => ".pdf")
    file_name = replace(file_name, "csv" => "plots/progress_plots/")
    if example == "sparse_reg"
        file_name = replace(file_name, "boscia_" => "size_active_set_")
    else
        file_name = replace(file_name, mode => "size_active_set_" * mode)
    end
    #@show file_name
    savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches="tight")
catch e
    println(e)
end
end 
#=
## sparse regression
modes = ["default"] # "no_tightening", "local_tigtening", "global_tightening", hybrid_branching_20", "strong_branching"

for mode in modes
    for m in [15]#15:30
        for seed in [4] #1:10
            file = joinpath(@__DIR__, "csv/boscia_" * mode * "_" * string(m) * "_" * string(seed) * "_sparse_reg.csv")
            per_layer_plot(file, mode)
        end
    end
end

# portfolio mixed
for mode in modes
    for m in [30]#20:5:120
        for seed in [8]#1:10
            file = joinpath(@__DIR__, "csv/" * mode * "_" * string(m) * "_" * string(seed) * "_mixed_portfolio.csv")
            per_layer_plot(file, mode)
        end
    end
end

# portfolio integer
for mode in modes
    for m in [20]#20:5:120
        for seed in [6]#1:10
            file = joinpath(@__DIR__, "csv/" * mode * "_" * string(m) * "_" * string(seed) * "_integer_portfolio.csv")
            per_layer_plot(file, mode)
        end
    end
end 
=#