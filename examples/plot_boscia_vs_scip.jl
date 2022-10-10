using PyPlot
using DataFrames
using CSV

iter = 1

df = DataFrame(CSV.File("examples/csv/boscia_vs_scip_1.csv"))
df_temp = df#[19:nrow(df), :]

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
#ax.scatter(df_temp[!,"dimension"], df_temp[!,"time_boscia"], label="Boscia", color=colors[1], marker=markers[1])
#ax.scatter(df_temp[!,"dimension"], df_temp[!,"time_scip"], label="SCIP", color=colors[3], marker=markers[2])
ax.plot(log.(sort(df_temp[!,"time_boscia"])), 1:nrow(df_temp), label="Boscia", color=colors[1], marker=markers[1])
ax.plot(log.(sort(df_temp[!,"time_scip"])), 1:nrow(df_temp), label="SCIP", color=colors[3], marker=markers[2])

yticks(1:nrow(df_temp),[])
#xlabel("Dimension")
xlabel("log time in s")
ax.grid()
fig.legend(loc=7, fontsize=12)
fig.tight_layout()
fig.subplots_adjust(right=0.75)  

file = ("examples/csv/boscia_vs_scip.png")
savefig(file)
