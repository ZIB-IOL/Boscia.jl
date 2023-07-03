# using Plots 
# pyplot()
using CSV
using DataFrames
using Statistics
using PyPlot

function per_layer_plot(file_name)
    df = DataFrame(CSV.File(file_name))

    if occursin("sqr_dst", file_name)
        example = "sqr_dst"
    elseif occursin("sparse_reg", file_name)
        example = "sparse_reg"
    elseif occursin("worst_case", file_name)
        example = "worst_case"
    elseif occursin("birkhoff", file_name)
        example = "birkhoff"
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
        df[!,"lmo_calls"] = df[!, "list_lmo_calls"]
        lmo_calls_per_layer = df[!,:lmo_calls]
        active_set_size = df[!,:active_set_size]
        discarded_set_size = df[!,:discarded_set_size]
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

    fig = plt.figure(figsize=(7,3.5))
    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("font", size=12, family="cursive")
    PyPlot.matplotlib[:rc]("axes", labelsize=14)
    PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
    \usepackage{libertine}
    \usepackage{libertinust1math}
    """)

    ax = fig.add_subplot(111)
    ax.plot(1:len, active_set_size_per_layer_mean, label="Active set", color=colors[end], marker=markers[2])
    ylabel("Avg set size")
    ax.plot(1:len, discarded_set_size_per_layer_mean, label="Discarded set", color=colors[4], marker=markers[3])

    ax2 = ax.twinx()
    ax2.plot(1:len, lmo_calls_per_layer_mean, label="LMO calls", color=colors[1], marker=markers[1])
    xticks(range(1, len, step=5))
    ylabel("LMO calls", Dict("color"=>"blue"))
    setp(ax2.get_yticklabels(),color="blue")
    ax.set(xlabel="Node depth")

    #ax.plot(1:len, lmo_calls_per_layer_std_below, color=:blue, linestyle="dashed")
    #ax.plot(1:len, lmo_calls_per_layer_std_above, color=:blue, linestyle="dashed")
    # add lmo calls for each node
    # for depth in 1:len
    #     ax.scatter([depth for i in 1:length(lmo_calls_per_layer[depth])], lmo_calls_per_layer[depth], color=:blue, s=1)
    # end

    #PyPlot.title(title)
    ax.grid()
    lgd = fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), fontsize=12,
            fancybox=true, shadow=false, ncol=3)
    fig.tight_layout()

    # plot(1:len, lmo_calls_per_layer, label="lmo calls", title=title, xlabel="node depth", ylabel="lmo calls", legend=:outerright, leftmargin = 1.5Plots.cm, rightmargin = 2.5Plots.cm,)
    # x = twinx()
    # plot!(x, active_set_size_per_layer, label="active set", color=:red, ylabel="set size")
    # plot!(x, discarded_set_size_per_layer, label="discarded set", color=:green)

    file_name = replace(file_name, ".csv" => ".pdf")
    file_name = replace(file_name, "csv" => "images")
    savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches="tight")
    @show file_name
end 

file_name = "csv/birkhoff_per_layer_3_3_1_Inf_0.7_0.001.csv"
# file_name = "csv/early_stopping_birkhoff_2_3_3_Inf_0.7_0.001_2.csv"
per_layer_plot(file_name)