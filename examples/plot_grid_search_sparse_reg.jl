# using Plots 
# pyplot()
using PyPlot
using DataFrames
using CSV
using Statistics

# either min_number_lower or adaptive_gap
compare_mode = "adaptive_gap" #"min_number_lower" #     
iter = 1

if compare_mode == "adaptive_gap" 
    # file_names = [
    # "early_stopping_worst_case_16_1_Inf_0.6",
    # "early_stopping_worst_case_16_1_Inf_0.65",
    # "early_stopping_worst_case_16_1_Inf_0.7",
    # "early_stopping_worst_case_16_1_Inf_0.75",
    # "early_stopping_worst_case_16_1_Inf_0.8",
    # "early_stopping_worst_case_16_1_Inf_0.85",
    # "early_stopping_worst_case_16_1_Inf_0.9"
    # ]
    # file_names = [
    #     "early_stopping_sparse_reg_25_1_Inf_0.65",
    #     "early_stopping_sparse_reg_25_1_Inf_0.7",
    #     "early_stopping_sparse_reg_25_1_Inf_0.75",
    #     "early_stopping_sparse_reg_25_1_Inf_0.8",
    #     "early_stopping_sparse_reg_25_1_Inf_0.85",
    #     "early_stopping_sparse_reg_25_1_Inf_0.9",
    #     "early_stopping_sparse_reg_25_1_Inf_1.0",
    # ]

    # file_names = [
    #     "early_stopping_int_sparsereg_40_2_Inf_0.65",
    #     "early_stopping_int_sparsereg_40_2_Inf_0.7",
    #     "early_stopping_int_sparsereg_40_2_Inf_0.75",
    #     "early_stopping_int_sparsereg_40_2_Inf_0.8",
    #     "early_stopping_int_sparsereg_40_2_Inf_0.85",
    #     "early_stopping_int_sparsereg_40_2_Inf_0.9",
    #     "early_stopping_int_sparsereg_40_2_Inf_1.0",
    # ]

    file_names = [
        "early_stopping_int_sparsereg_40_30_Inf_0.65",
        "early_stopping_int_sparsereg_40_30_Inf_0.7",
        "early_stopping_int_sparsereg_40_30_Inf_0.75",
        "early_stopping_int_sparsereg_40_30_Inf_0.8",
        "early_stopping_int_sparsereg_40_30_Inf_0.85",
        "early_stopping_int_sparsereg_40_30_Inf_0.9",
        "early_stopping_int_sparsereg_40_30_Inf_1.0",
    ]
    df = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
    for file in file_names
        # dataframe
        df_iter = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
        for i in 1:iter
            df_temp = DataFrame(CSV.File("examples/csv/" * file * "_0.0001_" * string(i) * ".csv"))
            df_temp_size = nrow(df_temp)
            df_temp = df_temp[[df_temp_size], :]
            df_temp = df_temp[!,Not(:lb)]
            df_temp = df_temp[!,Not(:ub)]
            df_temp = df_temp[!,Not(:list_time)]
            append!(df_iter, df_temp)
        end
        row_median = Dict(
            "seed"=>df_iter[[iter], :][!, :seed][1], 
            "dimension"=>df_iter[[iter], :][!, :dimension][1], 
            "min_number_lower"=>df_iter[[iter], :][!, :min_number_lower][1], 
            "adaptive_gap"=>df_iter[[iter], :][!, :adaptive_gap][1], 
            "iteration"=>median(df_iter[!, :iteration]), 
            "time"=>median(df_iter[!, :time]), 
            "memory"=>median(df_iter[!, :memory]), 
            "list_num_nodes"=>median(df_iter[!, :list_num_nodes]), 
            "list_lmo_calls"=>median(df_iter[!, :list_lmo_calls]),
            "active_set_size"=>median(df_iter[!, :active_set_size]), 
            "discarded_set_size"=>median(df_iter[!, :discarded_set_size]) )
        push!(df, row_median)
    end
    rename!(df,:list_num_nodes => :num_nodes)
    rename!(df,:list_lmo_calls => :lmo_calls)
    #display(df)

    # df of iteration 0.001
    df_2 = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
    for file in file_names
        # dataframe
        df_iter = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
        for i in 1:iter
            df_temp = DataFrame(CSV.File("examples/csv/" * file * "_0.001_" * string(i) * ".csv"))
            df_temp_size = nrow(df_temp)
            df_temp = df_temp[[df_temp_size], :]
            df_temp = df_temp[!,Not(:lb)]
            df_temp = df_temp[!,Not(:ub)]
            df_temp = df_temp[!,Not(:list_time)]
            append!(df_iter, df_temp)
        end
        row_median = Dict(
            "seed"=>df_iter[[iter], :][!, :seed][1], 
            "dimension"=>df_iter[[iter], :][!, :dimension][1], 
            "min_number_lower"=>df_iter[[iter], :][!, :min_number_lower][1], 
            "adaptive_gap"=>df_iter[[iter], :][!, :adaptive_gap][1], 
            "iteration"=>median(df_iter[!, :iteration]), 
            "time"=>median(df_iter[!, :time]), 
            "memory"=>median(df_iter[!, :memory]), 
            "list_num_nodes"=>median(df_iter[!, :list_num_nodes]), 
            "list_lmo_calls"=>median(df_iter[!, :list_lmo_calls]),
            "active_set_size"=>median(df_iter[!, :active_set_size]), 
            "discarded_set_size"=>median(df_iter[!, :discarded_set_size]) )
        push!(df_2, row_median)
    end
    rename!(df_2,:list_num_nodes => :num_nodes)
    rename!(df_2,:list_lmo_calls => :lmo_calls)

    # df 1.0e-7
    # df_3 = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
    # for file in file_names
    #     # dataframe
    #     df_iter = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
    #     for i in 1:iter
    #         df_temp = DataFrame(CSV.File("experiments/csv/" * file * "_1.0e-7_" * string(i) * ".csv"))
    #         df_temp_size = nrow(df_temp)
    #         df_temp = df_temp[[df_temp_size], :]
    #         df_temp = df_temp[!,Not(:lb)]
    #         df_temp = df_temp[!,Not(:ub)]
    #         df_temp = df_temp[!,Not(:list_time)]
    #         append!(df_iter, df_temp)
    #     end
    #     row_median = Dict(
    #         "seed"=>df_iter[[iter], :][!, :seed][1], 
    #         "dimension"=>df_iter[[iter], :][!, :dimension][1], 
    #         "min_number_lower"=>df_iter[[iter], :][!, :min_number_lower][1], 
    #         "adaptive_gap"=>df_iter[[iter], :][!, :adaptive_gap][1], 
    #         "iteration"=>median(df_iter[!, :iteration]), 
    #         "time"=>median(df_iter[!, :time]), 
    #         "memory"=>median(df_iter[!, :memory]), 
    #         "list_num_nodes"=>median(df_iter[!, :list_num_nodes]), 
    #         "list_lmo_calls"=>median(df_iter[!, :list_lmo_calls]),
    #         "active_set_size"=>median(df_iter[!, :active_set_size]), 
    #         "discarded_set_size"=>median(df_iter[!, :discarded_set_size]) )
    #     push!(df_3, row_median)
    # end
    # rename!(df_3,:list_num_nodes => :num_nodes)
    # rename!(df_3,:list_lmo_calls => :lmo_calls)

    # df 0.005
    df_4 = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
    for file in file_names
        # dataframe
        df_iter = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
        for i in 1:iter
            df_temp = DataFrame(CSV.File("examples/csv/" * file * "_0.005_" * string(i) * ".csv"))
            df_temp_size = nrow(df_temp)
            df_temp = df_temp[[df_temp_size], :]
            df_temp = df_temp[!,Not(:lb)]
            df_temp = df_temp[!,Not(:ub)]
            df_temp = df_temp[!,Not(:list_time)]
            append!(df_iter, df_temp)
        end
        row_median = Dict(
            "seed"=>df_iter[[iter], :][!, :seed][1], 
            "dimension"=>df_iter[[iter], :][!, :dimension][1], 
            "min_number_lower"=>df_iter[[iter], :][!, :min_number_lower][1], 
            "adaptive_gap"=>df_iter[[iter], :][!, :adaptive_gap][1], 
            "iteration"=>median(df_iter[!, :iteration]), 
            "time"=>median(df_iter[!, :time]), 
            "memory"=>median(df_iter[!, :memory]), 
            "list_num_nodes"=>median(df_iter[!, :list_num_nodes]), 
            "list_lmo_calls"=>median(df_iter[!, :list_lmo_calls]),
            "active_set_size"=>median(df_iter[!, :active_set_size]), 
            "discarded_set_size"=>median(df_iter[!, :discarded_set_size]) )
        push!(df_4, row_median)
    end
    rename!(df_4,:list_num_nodes => :num_nodes)
    rename!(df_4,:list_lmo_calls => :lmo_calls)

    # df 0.002
    # df_5 = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
    # for file in file_names
    #     # dataframe
    #     df_iter = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
    #     for i in 1:iter
    #         df_temp = DataFrame(CSV.File("experiments/csv/" * file * "_0.002_" * string(i) * ".csv"))
    #         df_temp_size = nrow(df_temp)
    #         df_temp = df_temp[[df_temp_size], :]
    #         df_temp = df_temp[!,Not(:lb)]
    #         df_temp = df_temp[!,Not(:ub)]
    #         df_temp = df_temp[!,Not(:list_time)]
    #         append!(df_iter, df_temp)
    #     end
    #     row_median = Dict(
    #         "seed"=>df_iter[[iter], :][!, :seed][1], 
    #         "dimension"=>df_iter[[iter], :][!, :dimension][1], 
    #         "min_number_lower"=>df_iter[[iter], :][!, :min_number_lower][1], 
    #         "adaptive_gap"=>df_iter[[iter], :][!, :adaptive_gap][1], 
    #         "iteration"=>median(df_iter[!, :iteration]), 
    #         "time"=>median(df_iter[!, :time]), 
    #         "memory"=>median(df_iter[!, :memory]), 
    #         "list_num_nodes"=>median(df_iter[!, :list_num_nodes]), 
    #         "list_lmo_calls"=>median(df_iter[!, :list_lmo_calls]),
    #         "active_set_size"=>median(df_iter[!, :active_set_size]), 
    #         "discarded_set_size"=>median(df_iter[!, :discarded_set_size]) )
    #     push!(df_5, row_median)
    # end
    # rename!(df_5,:list_num_nodes => :num_nodes)
    # rename!(df_5,:list_lmo_calls => :lmo_calls)

    # set up plot
    if occursin("sqr_dst", file_names[1])
        example = "sqr_dst"
    elseif occursin("sparse_reg", file_names[1])
        example = "sparse_reg"
    elseif occursin("worst_case", file_names[1])
        example = "worst_case"
    elseif occursin("sparse_group_poisson", file_names[1])
        example = "sparse_group_poisson"
    elseif occursin("int_sparsereg", file_names[1])
        example = "int_sparsereg"
    end

    title = "Example : " * example * ", Dimension : " * string(string(df[[iter], :][!, :dimension][1]))
    
    colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
    markers = ["o", "s", "^", "P", "X", "H", "D"]

    fig = plt.figure(figsize=(6.5,2.5))
    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("font", size=10, family="cursive")
    PyPlot.matplotlib[:rc]("axes", labelsize=10)
    PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
    \usepackage{libertine}
    \usepackage{libertinust1math}
    """)
    ax = fig.add_subplot(111)
    ax.plot(1:length(df[!,"num_nodes"]), df_2[!,"time"]./1000, label="0.001", color=colors[1], marker=markers[1])
    #ax.plot(1:length(df[!,"num_nodes"]), df_5[!,"time"]./1000, label="0.002", color=:red)
    ax.plot(1:length(df[!,"num_nodes"]), df_4[!,"time"]./1000, label="0.005", color=colors[end], marker=markers[2])
    ax.plot(1:length(df[!,"num_nodes"]), df[!,"time"]./1000, label="0.0001", color=colors[4], marker=markers[3])
    #ax.plot(1:length(df[!,"num_nodes"]), df_3[!,"time"]./1000, label="1e-7", color=colors[4], marker=markers[4])

    xticks(1:length(df[!,"adaptive_gap"]),[string(factor) for factor in df[!,"adaptive_gap"]])
    xlabel("Dual gap decay factor")
    ylabel("Time (s)")
    ax.grid()
    lgd = fig.legend(loc="upper center", title="FW gap tolerance", bbox_to_anchor=(0.43, 0.05), fontsize=8,
          fancybox=true, shadow=false, ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)  

elseif compare_mode == "min_number_lower"
    # file_names = [
    # "early_stopping_worst_case_16_1_20.0_0.7",
    # "early_stopping_worst_case_16_1_40.0_0.7",
    # "early_stopping_worst_case_16_1_60.0_0.7",
    # "early_stopping_worst_case_16_1_80.0_0.7",
    # "early_stopping_worst_case_16_1_100.0_0.7",
    # "early_stopping_worst_case_16_1_200.0_0.7",
    # "early_stopping_worst_case_16_1_Inf_0.7"
    # ]

    # file_names = [
    #     "early_stopping_sparse_reg_25_1_20.0_0.7",
    #     "early_stopping_sparse_reg_25_1_40.0_0.7",
    #     "early_stopping_sparse_reg_25_1_60.0_0.7",
    #     "early_stopping_sparse_reg_25_1_80.0_0.7",
    #     "early_stopping_sparse_reg_25_1_100.0_0.7",
    #     "early_stopping_sparse_reg_25_1_200.0_0.7",
    #     "early_stopping_sparse_reg_25_1_Inf_0.7"
    # ]

    file_names = [
        "early_stopping_int_sparsereg_40_2_20.0_0.7",
        "early_stopping_int_sparsereg_40_2_40.0_0.7",
        "early_stopping_int_sparsereg_40_2_60.0_0.7",
        "early_stopping_int_sparsereg_40_2_80.0_0.7",
        "early_stopping_int_sparsereg_40_2_100.0_0.7",
        "early_stopping_int_sparsereg_40_2_200.0_0.7",
        "early_stopping_int_sparsereg_40_2_Inf_0.7",
    ]
    
    df = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
    for file in file_names
        # dataframe
        df_iter = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
        for i in 1:iter
            df_temp = DataFrame(CSV.File("experiments/csv/" * file * "_0.001_" * string(i) * ".csv"))
            df_temp_size = nrow(df_temp)
            df_temp = df_temp[[df_temp_size], :]
            df_temp = df_temp[!,Not(:lb)]
            df_temp = df_temp[!,Not(:ub)]
            df_temp = df_temp[!,Not(:list_time)]
            append!(df_iter, df_temp)
        end
        row_median = Dict(
            "seed"=>df_iter[[iter], :][!, :seed][1], 
            "dimension"=>df_iter[[iter], :][!, :dimension][1], 
            "min_number_lower"=>df_iter[[iter], :][!, :min_number_lower][1], 
            "adaptive_gap"=>df_iter[[iter], :][!, :adaptive_gap][1], 
            "iteration"=>median(df_iter[!, :iteration]), 
            "time"=>median(df_iter[!, :time]), 
            "memory"=>median(df_iter[!, :memory]), 
            "list_num_nodes"=>median(df_iter[!, :list_num_nodes]), 
            "list_lmo_calls"=>median(df_iter[!, :list_lmo_calls]),
            "active_set_size"=>median(df_iter[!, :active_set_size]), 
            "discarded_set_size"=>median(df_iter[!, :discarded_set_size]) )
        push!(df, row_median)
    end
    rename!(df,:list_num_nodes => :num_nodes)
    rename!(df,:list_lmo_calls => :lmo_calls)

    df1 = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
    for file in file_names
        # dataframe
        df_iter = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
        for i in 1:iter
            df_temp = DataFrame(CSV.File("experiments/csv/" * file * "_0.0001_" * string(i) * ".csv"))
            df_temp_size = nrow(df_temp)
            df_temp = df_temp[[df_temp_size], :]
            df_temp = df_temp[!,Not(:lb)]
            df_temp = df_temp[!,Not(:ub)]
            df_temp = df_temp[!,Not(:list_time)]
            append!(df_iter, df_temp)
        end
        row_median = Dict(
            "seed"=>df_iter[[iter], :][!, :seed][1], 
            "dimension"=>df_iter[[iter], :][!, :dimension][1], 
            "min_number_lower"=>df_iter[[iter], :][!, :min_number_lower][1], 
            "adaptive_gap"=>df_iter[[iter], :][!, :adaptive_gap][1], 
            "iteration"=>median(df_iter[!, :iteration]), 
            "time"=>median(df_iter[!, :time]), 
            "memory"=>median(df_iter[!, :memory]), 
            "list_num_nodes"=>median(df_iter[!, :list_num_nodes]), 
            "list_lmo_calls"=>median(df_iter[!, :list_lmo_calls]),
            "active_set_size"=>median(df_iter[!, :active_set_size]), 
            "discarded_set_size"=>median(df_iter[!, :discarded_set_size]) )
        push!(df1, row_median)
    end
    rename!(df1,:list_num_nodes => :num_nodes)
    rename!(df1,:list_lmo_calls => :lmo_calls)

    # df 0.005
    df_4 = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
    for file in file_names
        # dataframe
        df_iter = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
        for i in 1:iter
            df_temp = DataFrame(CSV.File("experiments/csv/" * file * "_0.005_" * string(i) * ".csv"))
            df_temp_size = nrow(df_temp)
            df_temp = df_temp[[df_temp_size], :]
            df_temp = df_temp[!,Not(:lb)]
            df_temp = df_temp[!,Not(:ub)]
            df_temp = df_temp[!,Not(:list_time)]
            append!(df_iter, df_temp)
        end
        row_median = Dict(
            "seed"=>df_iter[[iter], :][!, :seed][1], 
            "dimension"=>df_iter[[iter], :][!, :dimension][1], 
            "min_number_lower"=>df_iter[[iter], :][!, :min_number_lower][1], 
            "adaptive_gap"=>df_iter[[iter], :][!, :adaptive_gap][1], 
            "iteration"=>median(df_iter[!, :iteration]), 
            "time"=>median(df_iter[!, :time]), 
            "memory"=>median(df_iter[!, :memory]), 
            "list_num_nodes"=>median(df_iter[!, :list_num_nodes]), 
            "list_lmo_calls"=>median(df_iter[!, :list_lmo_calls]),
            "active_set_size"=>median(df_iter[!, :active_set_size]), 
            "discarded_set_size"=>median(df_iter[!, :discarded_set_size]) )
        push!(df_4, row_median)
    end
    rename!(df_4,:list_num_nodes => :num_nodes)
    rename!(df_4,:list_lmo_calls => :lmo_calls)

    # df 1e-7
    df_5 = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
    for file in file_names
        # dataframe
        df_iter = DataFrame("seed"=>[], "dimension"=>[], "min_number_lower"=>[], "adaptive_gap"=>[], "iteration"=>[], "time"=>[], "memory"=>[], "list_num_nodes"=>[], "list_lmo_calls"=>[], "active_set_size"=>[], "discarded_set_size"=>[]) 
        for i in 1:iter
            df_temp = DataFrame(CSV.File("experiments/csv/" * file * "_1.0e-7_" * string(i) * ".csv"))
            df_temp_size = nrow(df_temp)
            df_temp = df_temp[[df_temp_size], :]
            df_temp = df_temp[!,Not(:lb)]
            df_temp = df_temp[!,Not(:ub)]
            df_temp = df_temp[!,Not(:list_time)]
            append!(df_iter, df_temp)
        end
        row_median = Dict(
            "seed"=>df_iter[[iter], :][!, :seed][1], 
            "dimension"=>df_iter[[iter], :][!, :dimension][1], 
            "min_number_lower"=>df_iter[[iter], :][!, :min_number_lower][1], 
            "adaptive_gap"=>df_iter[[iter], :][!, :adaptive_gap][1], 
            "iteration"=>median(df_iter[!, :iteration]), 
            "time"=>median(df_iter[!, :time]), 
            "memory"=>median(df_iter[!, :memory]), 
            "list_num_nodes"=>median(df_iter[!, :list_num_nodes]), 
            "list_lmo_calls"=>median(df_iter[!, :list_lmo_calls]),
            "active_set_size"=>median(df_iter[!, :active_set_size]), 
            "discarded_set_size"=>median(df_iter[!, :discarded_set_size]) )
        push!(df_5, row_median)
    end
    rename!(df_5,:list_num_nodes => :num_nodes)
    rename!(df_5,:list_lmo_calls => :lmo_calls)


    # set up plot
    if occursin("sqr_dst", file_names[1])
        example = "sqr_dst"
    elseif occursin("sparse_reg", file_names[1])
        example = "sparse_reg"
    elseif occursin("worst_case", file_names[1])
        example = "worst_case"
    elseif occursin("sparse_group_poisson", file_names[1])
        example = "sparse_group_poisson"
    elseif occursin("int_sparsereg", file_names[1])
        example = "int_sparsereg"
    end

    colors = ["b", "m", "c", "r", "g", "y", "k"]
    markers = ["o", "s", "^", "P", "X", "H", "D"]

    fig = plt.figure(figsize=(6.5,3))
    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("font", size=12, family="cursive")
    PyPlot.matplotlib[:rc]("axes", labelsize=14)
    PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
    \usepackage{libertine}
    \usepackage{libertinust1math}
    """)
    ax = fig.add_subplot(111)
    ax.scatter(1:length(df[!,"num_nodes"]), df[!,"time"]./1000, label="0.001", color=colors[1], marker=markers[1])
    ax.scatter(1:length(df_4[!,"num_nodes"]), df_4[!,"time"]./1000, label="0.005", color=colors[2], marker=markers[2])
    ax.scatter(1:length(df1[!,"num_nodes"]), df1[!,"time"]./1000, label="0.0001", color=colors[3], marker=markers[3])
    ax.scatter(1:length(df_5[!,"num_nodes"]), df_5[!,"time"]./1000, label="1e-7", color=colors[4], marker=markers[4])
    labels = [string(Int(factor)) for factor in df[!,"min_number_lower"][1:end-1]]
    push!(labels, "Inf")
    xticks(1:length(df[!,"min_number_lower"]),labels)
    xlabel("Min number lower")
    ylabel("Time (s)")
    ax.grid()
    fig.legend(loc=7, title="FW gap tolerance", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)  
end


file = "examples/images/early_stopping_grid_search_" * example * "_" *
    string(df[[iter], :][!, :dimension][1]) * "_" *
    string(df[[iter], :][!, :seed][1]) * "_" *
    #string(2) * "_" *
    string(df[[iter], :][!, :min_number_lower][1]) * "_" *
    string(df[[iter], :][!, :adaptive_gap][1]) * "_" *
    compare_mode * ".pdf"

savefig(file, bbox_extra_artists=(lgd,), bbox_inches="tight")
