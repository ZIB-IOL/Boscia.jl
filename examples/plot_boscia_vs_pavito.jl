using PyPlot
using DataFrames
using CSV

# function plot_boscia_vs_scip(example; boscia=true, scip_oa=false, ipopt=false, afw=true, ss=true, as=true, as_ss=true, boscia_methods=true)
function plot_boscia_vs_pavito(example)
    if example == "miplib"
        boscia=true 
        scip_oa=false
        ipopt=true
        pavito=true
        shot=true
    elseif example == "tailed_sparse_reg" || example == "tailed_sparse_log_reg"
        boscia=true 
        scip_oa=true
        pavito=true
        ipopt = false
        shot=true
    else
        boscia=true 
        scip_oa=true
        ipopt=true
        pavito=true
        shot=true
    end
    
    if example == "poisson"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/poisson_non_grouped.csv")))
    elseif example == "sparse_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/sparse_reg_non_grouped.csv")))
    elseif example == "mixed_portfolio"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/mixed_portfolio_non_grouped.csv")))
    elseif example == "integer"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/portfolio_integer_non_grouped.csv")))
    elseif example == "sparse_log_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/sparse_log_reg_non_grouped.csv")))
    elseif example == "tailed_sparse_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/tailed_cardinality_non_grouped.csv")))
    elseif example == "tailed_sparse_log_reg"
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/tailed_cardinality_sparse_log_reg_non_grouped.csv")))
    elseif example == "miplib"
        df_22433 = DataFrame(CSV.File(joinpath(@__DIR__, "csv/mip_lib_22433_non_grouped.csv")))
        df = select(df_22433, [:time_boscia, :termination_boscia, :optimal_boscia, :time_ipopt, :termination_ipopt, :optimal_ipopt, :time_pavito, :termination_pavito, :optimal_pavito, :time_shot, :termination_shot, :optimal_shot])
        df_neos5 = DataFrame(CSV.File(joinpath(@__DIR__, "csv/mip_lib_neos5_non_grouped.csv")))
        select!(df_neos5, [:time_boscia, :termination_boscia, :optimal_boscia, :time_ipopt, :termination_ipopt, :optimal_ipopt, :time_pavito, :termination_pavito, :optimal_pavito, :time_shot, :termination_shot, :optimal_shot])
        append!(df,df_neos5)
        df_pg534 = DataFrame(CSV.File(joinpath(@__DIR__, "csv/mip_lib_pg5_34_non_grouped.csv")))
        select!(df_pg534, [:time_boscia, :termination_boscia, :optimal_boscia, :time_ipopt, :termination_ipopt, :optimal_ipopt, :time_shot, :termination_shot, :optimal_shot]) # :time_pavito, :termination_pavito, :optimal_pavito])
        append!(df,df_pg534, cols=:subset)
        df_ran14x18 = DataFrame(CSV.File(joinpath(@__DIR__, "csv/mip_lib_ran14x18-disj-8_non_grouped.csv")))
        select!(df_ran14x18, [:time_boscia, :termination_boscia, :optimal_boscia, :time_ipopt, :termination_ipopt, :optimal_ipopt, :time_pavito, :termination_pavito, :optimal_pavito, :time_shot, :termination_shot, :optimal_shot])
        append!(df,df_ran14x18)

        df[df.time_boscia.>1800, :time_boscia] .= 1800
        df[df.time_ipopt.>1800, :time_ipopt] .= 1800
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
        df_boscia = deepcopy(df)
        filter!(row -> !(row.termination_boscia == 0),  df_boscia)
        if boscia && scip_oa
            filter!(row -> !(row.optimal_boscia == 0),  df_boscia)
        elseif example == "miplib"
            filter!(row -> !(row.optimal_boscia == 0),  df_boscia)
        end
        time_boscia = sort(df_boscia[!,"time_boscia"])
        push!(time_boscia, 1.1 * time_limit)
        ax.plot(time_boscia, [1:nrow(df_boscia); nrow(df_boscia)], label="BO (ours)", color=colors[1], marker=markers[1], markevery=0.1)
    end

    if scip_oa     
        df_scip = deepcopy(df)
        filter!(row -> !(row.termination_scip == 0),  df_scip)
        if boscia && scip_oa
            filter!(row -> !(row.optimal_scip == 0),  df_scip)
        end        
        time_scip = sort(df_scip[!,"time_scip"])
        push!(time_scip, 1.1 * time_limit)
        ax.plot(time_scip, [1:nrow(df_scip); nrow(df_scip)], label="SCIP+OA", color=colors[end], marker=markers[2], markevery=0.1)
    end

    if ipopt
        df_ipopt = deepcopy(df)
        filter!(row -> !(row.termination_ipopt == 0),  df_ipopt)
        if boscia && scip_oa && ipopt
            filter!(row -> !(row.optimal_ipopt == 0),  df_ipopt)
        elseif example == "miplib"
            filter!(row -> !(row.optimal_ipopt == 0),  df_ipopt)
        end
        time_ipopt = sort(df_ipopt[!,"time_ipopt"])
        push!(time_ipopt, 1.1 * time_limit)
        ax.plot(time_ipopt, [1:nrow(df_ipopt); nrow(df_ipopt)], label="BnB Ipopt", color=colors[2], marker=markers[3], markevery=0.1)
    end

    if pavito
        df_pavito = deepcopy(df)
        # filter!(row -> !(row.termination_pavito == 0),  df_pavito)
        if boscia && scip_oa && ipopt && pavito
            filter!(row -> !(row.optimal_pavito == 0),  df_pavito)
        elseif example == "miplib"
            filter!(row -> !ismissing(row.optimal_pavito),  df_pavito)
            filter!(row -> !(row.optimal_pavito == 0),  df_pavito)
        end
        time_pavito = sort(df_pavito[!,"time_pavito"])
        push!(time_pavito, 1.1 * time_limit)
        ax.plot(time_pavito, [1:nrow(df_pavito); nrow(df_pavito)], label="Pavito", color=colors[3], marker=markers[4], markevery=0.1)
    end

    if shot
        df_shot = deepcopy(df)
        # filter!(row -> !(row.termination_shot == 0),  df_shot)
        if boscia && scip_oa && ipopt && shot
            filter!(row -> !(row.optimal_shot == 0),  df_shot)
        elseif example == "miplib"
            filter!(row -> !ismissing(row.optimal_shot),  df_shot)
            filter!(row -> !(row.optimal_shot == 0),  df_shot)
        end
        time_shot = sort(df_shot[!,"time_shot"])
        push!(time_shot, 1.1 * time_limit)
        ax.plot(time_shot, [1:nrow(df_shot); nrow(df_shot)], label="SHOT", color=colors[4], marker=markers[5], markevery=0.1)
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

    fig.tight_layout()

    if pavito
        file = "images/" * example * "_boscia_pavito.pdf"
    else 
        file = "images/" * example * ".pdf"
    end
    savefig(file)
end