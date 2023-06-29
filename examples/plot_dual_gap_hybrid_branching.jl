using PyPlot
using CSV
using DataFrames

example = "int_sparse_reg"# "low_dim_high_dim"#"lasso" #   
if example == "lasso"
    dim_seed = "_20_28"
elseif example == "int_sparse_reg"
    dim_seed = "_40_30"
elseif example == "low_dim_high_dim"
    dim_seed = "_20_2"
end
# for int sparse reg, lasso: divided by [1,5,10,20]
# for low dim high dim [1, 2, 4, 10]
time = true

file_name1 = "csv/hybrid_branching_" * example * dim_seed * "_num_integer_dividedby" * string(1) * ".csv"
file_name2 = "csv/hybrid_branching_" * example * dim_seed * "_num_integer_dividedby" * string(5) * ".csv"
file_name3 = "csv/hybrid_branching_" * example * dim_seed * "_num_integer_dividedby" * string(10) * ".csv"
file_name4 = "csv/hybrid_branching_" * example * dim_seed * "_num_integer_dividedby" * string(20) * ".csv"


df1 = DataFrame(CSV.File(file_name1))
df2 = DataFrame(CSV.File(file_name2))
df3 = DataFrame(CSV.File(file_name3))
df4 = DataFrame(CSV.File(file_name4))

len1 = length(df1[!,"lb"])
len2 = length(df2[!,"lb"])
len3 = length(df3[!,"lb"])
len4 = length(df4[!,"lb"])
len = min(len1, len2, len3, len4)

colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
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


if time
    ax.plot(df1[1:len1, "list_time"]/1000, df1[1:len1,"lb"], label="Num_int", color=colors[1], marker=markers[1], markevery=0.05, alpha=0.5)
    ax.plot(df2[1:len2, "list_time"]/1000, df2[1:len2,"lb"], label="Num_int/2", color=colors[end], marker=markers[2], markevery=0.05, alpha=0.5)
    ax.plot(df3[1:len3, "list_time"]/1000, df3[1:len3,"lb"], label="Num_int/4", color=colors[4], marker=markers[3], markevery=0.05, alpha=0.5)
    ax.plot(df4[1:len4, "list_time"]/1000, df4[1:len4,"lb"], label="Num_int/10", color=colors[3], marker=markers[6], markevery=0.05, alpha=0.5)
    ax.set(xlabel="Time (s)", ylabel="Lower bound")
else 
    ax.plot(1:len1, df1[1:len1,"lb"], label="Num_int", color=colors[1], marker=markers[1], markevery=0.05, alpha=0.5) 
    ax.plot(1:len2, df2[1:len2,"lb"], label="Num_int/5", color=colors[end], marker=markers[2], markevery=0.05, alpha=0.5)
    ax.plot(1:len3, df3[1:len3,"lb"], label="Num_int/10", color=colors[4], marker=markers[3], markevery=0.05, alpha=0.5)
    ax.plot(1:len4, df4[1:len4,"lb"], label="Num_int/20", color=colors[3], marker=markers[6], markevery=0.05, alpha=0.5)
    ax.set(xlabel="Number of nodes", ylabel="Lower bound")
end

ax.grid()

# ax2 = ax.twinx()
# ax2.plot(1:len, df1[1:len, "node_level"], label = "Node depth", color = colors[5]) 
# setp(ax2.get_yticklabels(),color=colors[5])
# ax2.set(ylabel="Node level")
# ax2.yaxis.label.set_color(colors[5])
#ylabel(Dict("color"=>colors[5]))

lgd = fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.05), fontsize=12,
          fancybox=true, shadow=false, ncol=4)
fig.tight_layout()

#vline!([50], label = "switch", color = :darkorchid, linestyle = :dot)

#=plot(df1[1:len,"list_time"], df1[1:len,"lb"], label="num_int", xlabel="time in ms", legend=:outerright,linewidth = 1.5, rightmargin = 0.5Plots.cm, leftmargin = 0.5Plots.cm, topmargin = 0.5Plots.cm, color = :darkorange,)
plot!(df1[1:len,"list_time"], label="num_int/2", color = :chartreuse2, linewidth = 1.5, linestyle = :dashdot)
plot!(df1[1:len,"list_time"], label="num_int/4", color = :teal, linewidth = 1.5, linestyle = :dash)
plot!(df1[1:len,"list_time"], label="num_int/8", color = :turquoise, linewidth = 1.5, linestyle = :dashdotdot)
yaxis!("lower bound")=#

#=fig = Plots.figure(figsize=(6.5,3))
ax = fig.add_subplot(111)
ax.plot(1:len, df1[1:len,"lb"], label="num_int", xlabel="number of nodes", legend=:outerright,linewidth = 1.5, color = :darkorange,)
ylabel("lower bound")
ax.plot(1:len, df2[1:len,"lb"], label="num_int/2", color = :chartreuse2, linewidth = 1.5, linestyle = :dashdot)
ax.plot(1:len, df3[1:len,"lb"], label="num_int/4", color = :teal, linewidth = 1.5, linestyle = :dash)
ax.plot(1:len, df4[1:len,"lb"], label="num_int/8", color = :turquoise, linewidth = 1.5, linestyle = :dashdotdot)

ax2 = ax.twinx()
ax2.plot(1:len, df1[1:len,"node_level"], label="node level", color=:darkorange)
xlabel("Node depth")
xticks(range(1, len, step=5))
ylabel("Lmo calls", Dict("color"=>"blue"))
#setp(ax2.get_yticklabels(),color="blue")=#

#file_name = replace(file_name1, "csv/" => "images/")
#file_name = replace(file_name1, ".csv" => ".png")
if time 
    file_name = "images/hybrid_branching_time_" * example * dim_seed * ".pdf"
else 
    file_name = "images/hybrid_branching_nodes_" * example * dim_seed * ".pdf"
end 
savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches="tight")
