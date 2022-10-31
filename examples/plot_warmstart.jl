using JSON
using PyPlot

json_file = JSON.parsefile(joinpath(@__DIR__, "results_portfolio_warmstart.json"))

result_baseline = json_file["result_baseline"]
result_no_warmstart = json_file["result_no_warmstart"]
result_no_shadow = json_file["result_no_shadow"]
result_no_activeset = json_file["result_no_active_set"]

#@show result_baseline["list_time"]

colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
markers = ["o", "s", "^", "P", "X", "H", "D"]

fig = plt.figure(figsize=(6.5,4.5))
PyPlot.matplotlib[:rc]("text", usetex=true)
PyPlot.matplotlib[:rc]("font", size=12, family="cursive")
PyPlot.matplotlib[:rc]("axes", labelsize=14)
PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
\usepackage{libertine}
\usepackage{libertinust1math}
""")
ax = fig.add_subplot(111)
ax.plot(result_baseline["list_time"]/1000,result_baseline["list_lb"], label="BO (ours)", color=colors[1], marker=markers[1], markevery=0.05)
ax.plot(result_no_warmstart["list_time"]/1000, result_no_warmstart["list_lb"], label="NW", color=colors[end], marker=markers[2], markevery=0.05)
ax.plot(result_no_activeset["list_time"]/1000, result_no_activeset["list_lb"], label="NA", color=colors[2], marker=markers[3], markevery=0.05)
ax.plot(result_no_shadow["list_time"]/1000, result_no_shadow["list_lb"], label="NS", color=colors[3], marker=markers[4], markevery=0.05)
ylabel("Lower bound")
xlabel("Time (s)")
ax.grid()
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12, fancybox=true, shadow=false, ncol=4)
fig.tight_layout()
savefig("warmstart_time.pdf")

fig = plt.figure(figsize=(6.5,4.5))
PyPlot.matplotlib[:rc]("text", usetex=true)
PyPlot.matplotlib[:rc]("font", size=12, family="cursive")
PyPlot.matplotlib[:rc]("axes", labelsize=14)
PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
\usepackage{libertine}
\usepackage{libertinust1math}
""")
ax = fig.add_subplot(111)
ax.plot(result_baseline["list_lmo_calls_acc"],result_baseline["list_lb"], label="BO (ours)", color=colors[1], marker=markers[1], markevery=0.05)
ax.plot(result_no_warmstart["list_lmo_calls_acc"], result_no_warmstart["list_lb"], label="NW", color=colors[end], marker=markers[2], markevery=0.05)
ax.plot(result_no_activeset["list_lmo_calls_acc"], result_no_activeset["list_lb"], label="NA", color=colors[2], marker=markers[3], markevery=0.05)
ax.plot(result_no_shadow["list_lmo_calls_acc"], result_no_shadow["list_lb"], label="NS", color=colors[3], marker=markers[4], markevery=0.05)
ylabel("Lower bound")
xlabel("LMO calls")
ax.grid()
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=12, fancybox=true, shadow=false, ncol=4)
fig.tight_layout()
savefig("warmstart_lmo.pdf")
