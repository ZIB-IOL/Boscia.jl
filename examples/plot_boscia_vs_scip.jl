using PyPlot
using DataFrames
using CSV

# function plot_boscia_vs_scip(example; boscia=true, scip_oa=false, ipopt=false, afw=true, ss=true, as=true, as_ss=true, boscia_methods=true)
function plot_boscia_vs_scip(example)
    boscia=true 
    afw=true
    ss=true
    as=true
    as_ss=true
    boscia_methods=true


    df = DataFrame(CSV.File(joinpath(@__DIR__, "final_csvs/" * example * "_settings_non_grouped.csv")))
     
    
   #= if example == "poisson"
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
        df = select(df_22433, [:time_boscia, :termination_boscia, :optimal_boscia, :time_ipopt, :termination_ipopt, :optimal_ipopt])
        df_neos5 = DataFrame(CSV.File(joinpath(@__DIR__, "csv/mip_lib_neos5_non_grouped.csv")))
        select!(df_neos5, [:time_boscia, :termination_boscia, :optimal_boscia, :time_ipopt, :termination_ipopt, :optimal_ipopt])
        append!(df,df_neos5)
        df_pg534 = DataFrame(CSV.File(joinpath(@__DIR__, "csv/mip_lib_pg5_34_non_grouped.csv")))
        select!(df_pg534, [:time_boscia, :termination_boscia, :optimal_boscia, :time_ipopt, :termination_ipopt, :optimal_ipopt])
        append!(df,df_pg534)
        df_ran14x18 = DataFrame(CSV.File(joinpath(@__DIR__, "csv/mip_lib_ran14x18-disj-8_non_grouped.csv")))
        select!(df_ran14x18, [:time_boscia, :termination_boscia, :optimal_boscia, :time_ipopt, :termination_ipopt, :optimal_ipopt])
        append!(df,df_ran14x18)

        df[df.time_boscia.>1800, :time_boscia] .= 1800
        df[df.time_ipopt.>1800, :time_ipopt] .= 1800
    else
        error("wrong option")
    end =#

    time_limit = 1800

    colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
    markers = ["o", "s", "^", "P", "X", "H", "D"]

    fig = plt.figure(figsize=(7.3, 5))
    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("font", size=12, family="cursive")
    PyPlot.matplotlib[:rc]("axes", labelsize=14)
    PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
    \usepackage{newtxtext}
    """)
    ax = fig.add_subplot(111)

    #ax.scatter(df_temp[!,"dimension"], df_temp[!,"time_boscia"], label="BO (ours)", color=colors[1], marker=markers[1])
    #ax.scatter(df_temp[!,"dimension"], df_temp[!,"time_scip"], label="SCIP", color=colors[3], marker=markers[2])

    if boscia 
        df_boscia = deepcopy(df)
        filter!(row -> !(row.terminationBoscia == 0),  df_boscia)
        #if boscia && scip_oa
        #    filter!(row -> !(row.optimal_boscia == 0),  df_boscia)
        #elseif example == "miplib"
        #    filter!(row -> !(row.optimal_boscia == 0),  df_boscia)
        #end
        time_boscia = sort(df_boscia[!,"timeBoscia"])
        push!(time_boscia, 1.1 * time_limit)
        ax.plot(time_boscia, [1:nrow(df_boscia); nrow(df_boscia)], label="Default", color=colors[1], marker=markers[1], markevery=0.1)
    end

    if afw 
        df_afw = deepcopy(df)
        filter!(row -> !(row.terminationBoscia_Afw == 0),  df_afw)
        time_afw = sort(df_afw[!,"timeBoscia_Afw"])
        push!(time_afw, 1.1 * time_limit)
        ax.plot(time_afw, [1:nrow(df_afw); nrow(df_afw)], label="AFW", color=colors[4], marker=markers[3], markevery=0.1)
    end

    if ss 
        df_ss = deepcopy(df)
        filter!(row -> !(row.terminationBoscia_No_Ss == 0), df_ss)
        time_ss = sort(df_ss[!,"timeBoscia_No_Ss"])
        push!(time_ss, 1.1 * time_limit)
        ax.plot(time_ss, [1:nrow(df_ss); nrow(df_ss)], label="no shadow set", color=colors[5], marker=markers[4], markevery=0.1)
    end

    if as 
        df_as = deepcopy(df)
        filter!(row -> !(row.terminationBoscia_No_As == 0), df_as)
        time_as = sort(df_as[!,"timeBoscia_No_As"])
        push!(time_as, 1.1 * time_limit)
        ax.plot(time_as, [1:nrow(df_as); nrow(df_as)], label="no warm start", color=colors[6], marker=markers[5], markevery=0.1)
    end

    if as_ss 
        df_as_ss = deepcopy(df)
        filter!(row -> !(row.terminationBoscia_No_As_No_Ss == 0), df_as_ss)
        time_as_ss = sort(df_as_ss[!,"timeBoscia_No_As_No_Ss"])
        push!(time_as_ss, 1.1 * time_limit)
        ax.plot(time_as_ss, [1:nrow(df_as_ss); nrow(df_as_ss)], label="no warm start \nand no shadow set", color=colors[7], marker=markers[6], markevery=0.1)
    end

    ylabel("Solved instances")
    #locator_params(axis="y", nbins=4)
    xlabel("Time (s)")
    ax.set_xscale("log")
    ax.grid()
    if example == "portfolio_integer"
        title("Pure-Integer Portfolio Problem", loc="center")
    elseif example == "poisson_reg"
        title("Poisson Regression", loc="center")
    elseif example == "sparse_reg"
        title("Sparse Regression", loc="center")
    elseif example == "portfolio_mixed"
        title("Mixed-Integer Portfolio Problem", loc="center")
    elseif example == "sparse_log_reg"
        title("Sparse Log Regression", loc="center")
    elseif example == "tailed_cardinality"
        title("Tailed Sparse Regression", loc="center")
    elseif example == "tailed_cardinality_sparse_log_reg"
        title("Tailed Sparse Log Regression", loc="center")
    elseif example == "miplib_22433"
        title("MIP LIB 22433", loc="center")
    elseif example == "miplib_neos5"
        title("MIP LIB neos5", loc="center")
    elseif example == "miplib_pg5_34"
        title("MIP LIB pg5_34", loc="center")
    elseif example == "miplib_ran14x18-disj-8"
        title("MIP LIB ran14x18-disj-8", loc="center")
    end

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12,
        fancybox=true, shadow=false, ncol=2)

    if afw
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12,
        fancybox=true, shadow=false, ncol=3)
    end

    fig.tight_layout()

    if boscia_methods
        file = joinpath(@__DIR__, "plots/" * example * "_boscia_settings.pdf")
    else 
        file = joinpath(@__DIR__, "csv/" * example * ".pdf")
    end
    savefig(file)
end

examples = ["miplib_22433", "miplib_neos5", "miplib_pg5_34", "miplib_ran14x18-disj-8", "poisson_reg", "portfolio_integer", "portfolio_mixed", "sparse_log_reg", "sparse_reg", "tailed_cardinality", "tailed_cardinality_sparse_log_reg"]
for example in examples
    @show example
    plot_boscia_vs_scip(example)
end