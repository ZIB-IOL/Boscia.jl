using PyPlot
using DataFrames
using CSV

iter = 1

df = DataFrame(CSV.File("examples/csv/boscia_vs_scip.csv"))
df_temp = df[19:nrow(df), :]

# display(df_temp)

indices = [index for index in 1:nrow(df_temp) if isodd(index)]
delete!(df_temp, indices) 

display(df_temp)

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
ax.scatter(df_temp[!,"dimension"], df_temp[!,"time_boscia"], label="Boscia", color=colors[1], marker=markers[1])
ax.scatter(df_temp[!,"dimension"], df_temp[!,"time_scip"], label="SCIP", color=colors[3], marker=markers[2])

#xticks(1:length(df_temp[!,"dimension"]),[string(dim) for dim in df_temp[!,"dimension"]])
xlabel("Dimension")
ylabel("Time in s")
ax.grid()
fig.legend(loc=7, fontsize=12)
fig.tight_layout()
fig.subplots_adjust(right=0.75)  

file = ("examples/csv/boscia_vs_scip.png")
savefig(file)
