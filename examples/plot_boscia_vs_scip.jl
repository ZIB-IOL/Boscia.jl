using PyPlot
using DataFrames
using CSV

# function plot_boscia_vs_scip(example; boscia=true, scip_oa=false, ipopt=false, afw=true, ss=true, as=true, as_ss=true, boscia_methods=true)
function plot_boscia_vs_scip(example, mode)
    if mode == "boscia_methods"
        boscia=true 
        scip_oa=false
        ipopt=false
        afw=true
        ss=true
        as=true
        as_ss=true
        boscia_methods=true
    elseif mode == "solvers"
        boscia=true 
        scip_oa=true
        ipopt=true
        afw=false
        ss=false
        as=false
        as_ss=false
        boscia_methods=false
        if example == "tailed_sparse_reg" || example == "tailed_sparse_log_reg"
            ipopt = false
        end
    end
    
    if example == "poisson"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/poisson_non_grouped.csv")))
    elseif example == "sparse_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/sparse_reg_non_grouped.csv")))
    elseif example == "mixed_portfolio"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/mixed_portfolio_non_grouped.csv")))
    elseif example == "sparse_log_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/sparse_log_reg_non_grouped.csv")))
    elseif example == "tailed_sparse_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/tailed_cardinality_non_grouped.csv")))
    elseif example == "tailed_sparse_log_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/tailed_cardinality_sparse_log_reg_non_grouped.csv")))
    else
        error("wrong option")
    end

    time_limit = 1800

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

    if boscia 
        df_boscia = copy(df)
        filter!(row -> !(row.termination_boscia == 0),  df_boscia)
        if boscia & scip_oa & ipopt
            filter!(row -> !(row.optimal_boscia == 0),  df_boscia)
        end
        time_boscia = sort(df_boscia[!,"time_boscia"])
        push!(time_boscia, 1.1 * time_limit)
        ax.plot(time_boscia, [1:nrow(df_boscia); nrow(df_boscia)], label="BO (ours)", color=colors[1], marker=markers[1], markevery=0.1)
    end

    if scip_oa     
        df_scip = copy(df)
        filter!(row -> !(row.termination_scip == 0),  df_scip)
        if boscia & scip_oa & ipopt
            filter!(row -> !(row.optimal_boscia == 0),  df_boscia)
        end        
        time_scip = sort(df_scip[!,"time_scip"])
        push!(time_scip, 1.1 * time_limit)
        ax.plot(time_scip, [1:nrow(df_scip); nrow(df_scip)], label="SCIP+OA", color=colors[end], marker=markers[2], markevery=0.1)
    end

    if ipopt
        df_ipopt = copy(df)
        filter!(row -> !(row.termination_ipopt == 0),  df_ipopt)
        if boscia & scip_oa & ipopt
            filter!(row -> !(row.optimal_ipopt == 0),  df_ipopt)
        end
        time_ipopt = sort(df_ipopt[!,"time_ipopt"])
        push!(time_ipopt, 1.1 * time_limit)
        ax.plot(time_ipopt, [1:nrow(df_ipopt); nrow(df_ipopt)], label="Ipopt", color=colors[2], marker=markers[3], markevery=0.1)
    end

    if afw 
        df_afw = copy(df)
        filter!(row -> !(row.termination_afw == 0),  df_afw)
        time_afw = sort(df_afw[!,"time_afw"])
        push!(time_afw, 1.1 * time_limit)
        ax.plot(time_afw, [1:nrow(df_afw); nrow(df_afw)], label="AFW", color=colors[4], marker=markers[3], markevery=0.1)
    end

    if ss 
        df_ss = copy(df)
        filter!(row -> !(row.termination_no_ss == 0), df_ss)
        time_ss = sort(df_ss[!,"time_no_ss"])
        push!(time_ss, 1.1 * time_limit)
        ax.plot(time_ss, [1:nrow(df_ss); nrow(df_ss)], label="no shadow set", color=colors[5], marker=markers[4], markevery=0.1)
    end

    if as 
        df_as = copy(df)
        filter!(row -> !(row.termination_no_as == 0), df_as)
        time_as = sort(df_as[!,"time_no_as"])
        push!(time_as, 1.1 * time_limit)
        ax.plot(time_as, [1:nrow(df_as); nrow(df_as)], label="no active set", color=colors[6], marker=markers[5], markevery=0.1)
    end

    if as_ss 
        df_as_ss = copy(df)
        filter!(row -> !(row.termination_no_ws == 0), df_as_ss)
        time_as_ss = sort(df_as_ss[!,"time_no_ws"])
        push!(time_as_ss, 1.1 * time_limit)
        ax.plot(time_as_ss, [1:nrow(df_as_ss); nrow(df_as_ss)], label="no warm start", color=colors[7], marker=markers[6], markevery=0.1)
    end

    ylabel("Solved instances")
    #locator_params(axis="y", nbins=4)
    xlabel("Time (s)")
    ax.set_xscale("log")
    ax.grid()
    if example == "integer" || example == "integer_50"
        title("Pure-integer portfolio problem", loc="center")
    elseif example == "poisson"
        title("Poisson regression", loc="center")
    elseif example == "sparse_reg"
        title("Sparse regression", loc="center")
    elseif example == "mixed_portfolio"
        title("Mixed-integer portfolio problem", loc="center")
    elseif example == "sparse_log_reg"
        title("Sparse log regression", loc="center")
    elseif example == "tailed_sparse_reg"
        title("Tailed sparse regression", loc="center")
    elseif example == "tailed_sparse_log_reg"
        title("Tailed sparse log regression", loc="center")
    end

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12,
        fancybox=true, shadow=false, ncol=2)

    if afw
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12,
        fancybox=true, shadow=false, ncol=3)
    end

    fig.tight_layout()

    if boscia_methods
        file = "csv/" * example * "_boscia_methods.pdf"
    else 
        file = "csv/" * example * ".pdf"
    end
    savefig(file)
end