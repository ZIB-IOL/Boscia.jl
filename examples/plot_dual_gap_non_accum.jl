using PyPlot
using CSV
using DataFrames

#file_name = "experiments/csv/dual_gap_low_dim_high_dim_20_2_most_infeasible.csv"
#file_name = "experiments/csv/early_stopping_birkhoff_3_2_1_Inf_0.7_0.001_1.csv"
#file_name = "experiments/csv/early_stopping_worst_case_16_1_Inf_0.7_0.001_3.csv"
#file_name = "experiments/csv/early_stopping_sparse_reg_25_1_Inf_0.65_0.001_2.csv"
#file_name = "experiments/csv/early_stopping_sparse_reg_16_1_Inf_0.65_0.001_2.csv"
#file_name = "experiments/csv/early_stopping_low_dim_400_20_1_Inf_0.8_0.001_1.csv"
file_name = "csv/early_stopping_birkhoff_3_2_2_Inf_0.7_0.001_1.csv"
#file_name = "experiments/csv/early_stopping_sparse_reg_16_1_Inf_0.7_0.001_1.csv"
#file_name = "experiments/csv/early_stopping_low_dim_100_5_3_Inf_1.0_1.0e-7_1.csv"
#file_name = "experiments/csv/early_stopping_portfolio_40_2_Inf_0.7_0.001_1.csv"
df = DataFrame(CSV.File(file_name))

# file_name_2 = "experiments/csv/early_stopping_sparse_reg_25_1_Inf_0.65_1.0e-7_3.csv"
# df2 = DataFrame(CSV.File(file_name))

if occursin("sqr_dst", file_name)
    example = "sqr_dst"
elseif occursin("sparse_reg", file_name)
    example = "sparse_reg"
elseif occursin("worst_case", file_name)
    example = "worst_case"
elseif occursin("birkhoff", file_name)
    example = "birkhoff"
elseif occursin("low_dim", file_name)
    example = "low_dim"
elseif occursin("lasso", file_name)
    example = "lasso"
elseif occursin("portfolio", file_name)
    example = "portfolio"
end

data = replace(file_name, "experiments/csv/early_stopping_" * example * "_" => "")
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
fw_dual_gap_precision = data[1:next_index[1]-1] 

if example == "sqr_dst"
    title = "Example : squared distance, Dimension : " * string(dimension) * "\n min_number_lower=" * string(min_number_lower) * ", fw_dual_gap_precision=" * string(fw_dual_gap_precision)
elseif example == "sparse_reg"
    title = "Example : sparse regression, Dimension : " * string(dimension) * "\n min_number_lower=" * string(min_number_lower) * ", fw_dual_gap_precision=" * string(fw_dual_gap_precision)
elseif example == "worst_case"
    title = "Example : worst case, Dimension : " * string(dimension) * "\n min_number_lower=" * string(min_number_lower) * ", fw_dual_gap_precision=" * string(fw_dual_gap_precision)
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
#ax.set_ylim(bottom=-5500, top=-4900)

ax2 = ax.twinx()
total_lmo_calls = df[!, "list_lmo_calls"]
previous_lmo_calls = [0]
append!(previous_lmo_calls, total_lmo_calls[1:end-1])
lmo_calls_non_accum = total_lmo_calls - previous_lmo_calls
ax2.plot(1:len, lmo_calls_non_accum, label="LMO calls", color=colors[1], marker=markers[1], markevery=0.05, alpha=0.5)
ylabel("LMO calls", Dict("color"=>colors[1]))
setp(ax2.get_yticklabels(),color=colors[1])

#PyPlot.title(title)
ax.grid()
lgd = fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), fontsize=12,
          fancybox=true, shadow=false, ncol=3)
fig.tight_layout()

# plot(1:len, df[!,"ub"], label="upper bound", title=title, xlabel="number of iterations", legend=:topleft, rightmargin = 2.5Plots.cm,)
# plot!(1:len, df[!,"lb"], label="lower bound")
# plot!(twinx(), df[!, "list_lmo_calls"], label="#lmo calls", color=:green)
#yaxis!("objective value")

file_name = replace(file_name, ".csv" => ".pdf")
file_name = replace(file_name, "csv" => "images")
file_name = replace(file_name, "early_stopping" => "dual_gap_non_accum_")
#file_name = replace(file_name, "dual_gap" => "dual_gap_non_accum_")

savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches="tight")