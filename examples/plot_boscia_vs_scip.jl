using PyPlot
using DataFrames
using CSV

function plot_boscia_vs_scip(mode; afw=false, ss=false, as=false, as_ss=false)
    if mode == "mixed_50"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_vs_scip_mixed_50.csv")))
    elseif mode == "integer_50"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/boscia_vs_scip_integer_50.csv")))
    elseif mode == "poisson"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/poisson_non_grouped.csv")))
    else
        error("wrong option")
    end

    time_limit = 1800

    # display(df)
    if mode == "poisson"
        df_boscia = copy(df)
        filter!(row -> !(row.termination_boscia == 0),  df_boscia)
        df_scip = copy(df)
        filter!(row -> !(row.termination_scip == 0),  df_scip)

    else 
        # indices = [index for index in 1:nrow(df) if isodd(index)]
        df_boscia = copy(df)
        df_scip = copy(df)
        # delete!(df_boscia, indices)
        # delete!(df_scip, indices)

        # display(df_scip)
        filter!(row -> !(row.termination_scip == "TIME_LIMIT"),  df_scip)
        filter!(row -> !(row.time_scip >= time_limit),  df_scip)
        #filter!(row -> !(row.termination_scip == "OPTIMIZE_NOT_CALLED"),  df_boscia)
        #df_boscia = filter(row -> !(row.termination_boscia == "Time limit reached"),  df_boscia)
        filter!(row -> !(row.time_boscia >= time_limit),  df_boscia)
        #df_boscia = filter(row -> !(row.termination_scip == "TIME_LIMIT" && isapprox(row.solution_boscia, row.solution_scip)),  df_boscia)

        # display(df_scip)
        # display(df_boscia)
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

    #ax.scatter(df_temp[!,"dimension"], df_temp[!,"time_boscia"], label="BO (ours)", color=colors[1], marker=markers[1])
    #ax.scatter(df_temp[!,"dimension"], df_temp[!,"time_scip"], label="SCIP", color=colors[3], marker=markers[2])
    time_boscia = sort(df_boscia[!,"time_boscia"])
    time_scip = sort(df_scip[!,"time_scip"])
    push!(time_boscia, 1.1 * time_limit)
    push!(time_scip, 1.1 * time_limit)
    if mode == "poisson"
        ax.plot(time_boscia, [1:nrow(df_boscia); nrow(df_boscia)], label="BO (ours)", color=colors[1], marker=markers[1], markevery=0.1)
        ax.plot(time_scip, [1:nrow(df_scip); nrow(df_scip)], label="SCIP+OA", color=colors[end], marker=markers[2], markevery=0.1)
        #yticks(0:2:nrow(df_boscia)+1, 0:2:nrow(df_boscia)+1)
    else 
        ax.plot(time_boscia, [1:nrow(df_boscia); nrow(df_boscia)], label="BO (ours)", color=colors[1], marker=markers[1])
        ax.plot(time_scip, [1:nrow(df_scip); nrow(df_scip)], label="SCIP+OA", color=colors[end], marker=markers[2])
    end

    if afw 
        if mode == "mixed_50"
            df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_mixed_50.csv")))
            filter!(row -> !(row.time_afw >= time_limit),  df_afw)
            time_afw = sort(df_afw[!,"time_afw"])
            push!(time_afw, 1.1 * time_limit)
            ax.plot(time_afw, [1:nrow(df_afw); nrow(df_afw)], label="AFW", color=colors[4], marker=markers[3])
        elseif mode == "integer_50"
            df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/afw_integer_50.csv")))
            filter!(row -> !(row.time_afw >= time_limit),  df_afw)
            time_afw = sort(df_afw[!,"time_afw"])
            push!(time_afw, 1.1 * time_limit)
            ax.plot(time_afw, [1:nrow(df_afw); nrow(df_afw)], label="AFW", color=colors[4], marker=markers[3])
        elseif mode == "poisson"
            df_afw = copy(df)
            filter!(row -> !(row.termination_afw == 0),  df_afw)
            time_afw = sort(df_afw[!,"time_afw"])
            push!(time_afw, 1.1 * time_limit)
            ax.plot(time_afw, [1:nrow(df_afw); nrow(df_afw)], label="AFW", color=colors[4], marker=markers[3], markevery=0.1)
        end
    end

    if ss 
        if mode == "mixed_50"
            df_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_ss_mixed_50.csv")))
            filter!(row -> !(row.time_afw >= time_limit),  df_ss)
            time_ss = sort(df_afw[!,"time_afw"])
            push!(time_ss, 1.1 * time_limit)
            ax.plot(time_ss, [1:nrow(df_ss); nrow(df_ss)], label="no shadow set", color=colors[5], marker=markers[4])
        elseif mode == "integer_50"
            df_ss = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_ss_integer_50.csv")))
            filter!(row -> !(row.time_afw >= time_limit),  df_ss)
            time_ss = sort(df_ss[!,"time_afw"])
            push!(time_ss, 1.1 * time_limit)
            ax.plot(time_ss, [1:nrow(df_ss); nrow(df_ss)], label="no shadow set", color=colors[5], marker=markers[4])
        elseif mode == "poisson"
            df_ss = copy(df)
            filter!(row -> !(row.termination_no_ss == 0), df_ss)
            time_ss = sort(df_ss[!,"time_no_ss"])
            push!(time_ss, 1.1 * time_limit)
            ax.plot(time_ss, [1:nrow(df_ss); nrow(df_ss)], label="no shadow set", color=colors[5], marker=markers[4], markevery=0.1)
        end
    end

    if as 
        if mode == "mixed_50"
            df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_mixed_50.csv")))
            filter!(row -> !(row.time_afw >= time_limit),  df_afw)
            time_afw = sort(df_afw[!,"time_afw"])
            push!(time_afw, 1.1 * time_limit)
            ax.plot(time_afw, [1:nrow(df_afw); nrow(df_afw)], label="no active set", color=colors[6], marker=markers[5])
        elseif mode == "integer_50"
            df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_integer_50.csv")))
            filter!(row -> !(row.time_afw >= time_limit),  df_afw)
            time_afw = sort(df_afw[!,"time_afw"])
            push!(time_afw, 1.1 * time_limit)
            ax.plot(time_afw, [1:nrow(df_afw); nrow(df_afw)], label="no active set", color=colors[6], marker=markers[5])
        elseif mode == "poisson"
            df_as = copy(df)
            filter!(row -> !(row.termination_no_as == 0), df_as)
            time_as = sort(df_as[!,"time_no_as"])
            push!(time_as, 1.1 * time_limit)
            ax.plot(time_as, [1:nrow(df_as); nrow(df_as)], label="no active set", color=colors[6], marker=markers[5], markevery=0.1)
        end
    end

    if as_ss 
        if mode == "mixed_50"
            df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_mixed_50.csv")))
            filter!(row -> !(row.time_afw >= time_limit),  df_afw)
            time_afw = sort(df_afw[!,"time_afw"])
            push!(time_afw, 1.1 * time_limit)
            ax.plot(time_afw, [1:nrow(df_afw); nrow(df_afw)], label="no warm start", color=colors[7], marker=markers[6])
        elseif mode == "integer_50"
            df_afw = DataFrame(CSV.File(joinpath(@__DIR__, "csv/no_warm_start_as_ss_integer_50.csv")))
            filter!(row -> !(row.time_afw >= time_limit),  df_afw)
            time_afw = sort(df_afw[!,"time_afw"])
            push!(time_afw, 1.1 * time_limit)
            ax.plot(time_afw, [1:nrow(df_afw); nrow(df_afw)], label="no warm start", color=colors[7], marker=markers[6])
        elseif mode == "poisson"
            df_as_ss = copy(df)
            filter!(row -> !(row.termination_no_ws == 0), df_as_ss)
            time_as_ss = sort(df_as_ss[!,"time_no_ws"])
            push!(time_as_ss, 1.1 * time_limit)
            ax.plot(time_as_ss, [1:nrow(df_as_ss); nrow(df_as_ss)], label="no warm start", color=colors[7], marker=markers[6], markevery=0.1)
        end
    end

    ylabel("Solved instances")
    #locator_params(axis="y", nbins=4)
    xlabel("Time (s)")
    ax.set_xscale("log")
    ax.grid()
    if mode == "integer" || mode == "integer_50"
        title("Pure-integer portfolio problem", loc="center")
    elseif mode == "poisson"
        title("Poisson regression", loc="center")
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

    if mode == "mixed_50"
        file = ("examples/csv/boscia_vs_scip_mixed_50.pdf")
    elseif mode == "integer_50"
        file = ("examples/csv/boscia_vs_scip_integer_50.pdf")
    elseif mode == "poisson"
        file = ("csv/poisson.pdf")
    end

    savefig(file)
end
