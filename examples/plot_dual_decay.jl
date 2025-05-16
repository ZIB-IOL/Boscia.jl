# using Plots 
# pyplot()
using PyPlot
using CSV
using DataFrames

function plot_dual_decay(dim, seed)
    try 
    # collect the time for each factor per each epsilon
    decay_factors = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
    epsilons = [1e-2, 1e-3, 5e-3, 1e-4]

    function get_time_per_factor(epsilon, factors)
        time = []
        for factor in factors    
            file_name = joinpath(@__DIR__, "csv/Boscia/boscia_dual_gap_decay_factor_sparse_reg_" * string(seed) * "_" * string(dim) * "_" * string(factor) * "_" * string(epsilon) * ".csv")
            df = DataFrame(CSV.File(file_name))

            push!(time, df[!,:time])
        end

        return time
    end

    time1 = get_time_per_factor(1e-2, decay_factors)
    time2 = get_time_per_factor(1e-3, decay_factors)
    time3 = get_time_per_factor(5e-3, decay_factors)
    time4 = get_time_per_factor(1e-4, decay_factors)

    colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
    markers = ["o", "s", "^", "P", "X", "H", "D"]

    fig = plt.figure(figsize=(6.5,3.5))
    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("font", size=12, family="cursive")
    PyPlot.matplotlib[:rc]("axes", labelsize=14)
    PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
   \usepackage{newtxtext}
    """)

    ax = fig.add_subplot(111)
    lns1 = ax.plot(decay_factors, time1, label="0.01", color=colors[end], marker=markers[2], markevery=0.5)
    lns3 = ax.plot(decay_factors, time3, label="0.005", color=colors[4], marker=markers[3], markevery=0.5 )
    lns2 = ax.plot(decay_factors, time2, label="0.001", color=colors[6], marker=markers[1], markevery=0.5 )
    lns4 = ax.plot(decay_factors, time4, label="0.0001", color=colors[1], marker=markers[4], markevery=0.5 )
    #lns1 = ax.plot(1:len, df[!,"ub"], label="Incumbent", color=colors[end], marker=markers[2], markevery=0.05)
    #lns2 = ax.plot(1:len, df[!,"lb"], label="Lower bound", color=colors[4], marker=markers[3], markevery=0.05)
    #ax.plot(1:length(df2[!,"ub"]), df2[!,"lb"], label="Lower bound, adaptive gap = 0.65", color=colors[4], linestyle="dashed")
    xlabel("Dual gap decay factor")
    #xticks(range(1, len, step=50))
    ylabel("Time (s)")


    #PyPlot.title(title)
    ax.grid()
    #fig.legend(loc=7, fontsize=12)
    lgd = fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), fontsize=12,
            fancybox=true, shadow=false, ncol=4)
    fig.tight_layout()

    file_name = joinpath(@__DIR__, "plots/dual_gap_decay_factor_sparse_reg_" * string(dim) * "_" * string(seed) * ".pdf")

    @show file_name

    savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches="tight")
catch e
    println(e)
    return
end
end
plot_dual_decay(27, 1)

#for dim in 15:30
#    for seed in 1:3
#        plot_dual_decay(dim, seed)
#    end
#end