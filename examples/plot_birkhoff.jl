using PyPlot
using DataFrames
using CSV

function plot_term(modes; by_time = false)
	file_name = ""

	if by_time
		df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/birkhoff_comparison_non_grouped.csv")))

		#fig = plt.figure(figsize=(6.5,4.5)) 
		fig, ax = plt.subplots(1, sharex = true, sharey = false, figsize = (6.5, 4.5))
		colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
		markers = ["o", "s", "^", "P", "X", "H", "D"]
		linestyle = ["-", ":", "-.", "--"]

		PyPlot.matplotlib[:rc]("text", usetex = true)
		PyPlot.matplotlib[:rc]("font", size = 11, family = "cursive")
		PyPlot.matplotlib[:rc]("axes", labelsize = 14)
		PyPlot.matplotlib[:rc]("text.latex", preamble = raw"""
		  \usepackage{libertine}
		  \usepackage{libertinust1math}
		  """)
		#ax = fig.add_subplot(111)


		if "custom" in modes
			df_hun = deepcopy(df)
			filter!(row -> !(row.terminationBoscia_Custom == 0), df_hun)
			x_hun = sort(df_hun[!, "timeBoscia_Custom"])
			ax.plot(x_hun, 1:nrow(df_hun), label = "Boscia + Hungarian", color = colors[2], linestyle = linestyle[1], marker = markers[1])
		end

		if "custom_lazy" in modes
			df_hun_lazy = deepcopy(df)
			filter!(row -> !(row.terminationBoscia_Custom_Lazy == 0), df_hun_lazy)
			x_hun_lazy = sort(df_hun_lazy[!, "timeBoscia_Custom_Lazy"])
			ax.plot(x_hun_lazy, 1:nrow(df_hun_lazy), label = "Boscia + Hungarian + Lazy", color = colors[3], linestyle = linestyle[1], marker = markers[4])
		end

		if "custom_dicg" in modes
			df_dicg = deepcopy(df)
			filter!(row -> !(row.terminationBoscia_Custom_DICG == 0), df_dicg)
			x_dicg = sort(df_dicg[!, "timeBoscia_Custom_DICG"])
			ax.plot(x_dicg, 1:nrow(df_dicg), label = "Boscia + Hungarian + DICG", color = colors[4], linestyle = linestyle[2], marker = markers[2])
		end

		if "custom_dicg_lazy" in modes
			df_custom_dicg_lazy = deepcopy(df)
			filter!(row -> !(row.terminationBoscia_Custom_DICG_Lazy == 0), df_custom_dicg_lazy)
			x_custom_dicg_lazy = sort(df_custom_dicg_lazy[!, "timeBoscia_Custom_DICG_Lazy"])
			ax.plot(x_custom_dicg_lazy, 1:nrow(df_custom_dicg_lazy), label = "Boscia + Hungarian + DICG + Lazy", color = colors[7], linestyle = linestyle[1], marker = markers[1])
		end

		if "custom_dicg_ws" in modes
			df_custom_dicg_ws = deepcopy(df)
			filter!(row -> !(row.terminationBoscia_Custom_DICG_WS == 0), df_custom_dicg_ws)
			x_custom_dicg_ws = sort(df_custom_dicg_ws[!, "timeBoscia_Custom_DICG_WS"])
			ax.plot(x_custom_dicg_ws, 1:nrow(df_custom_dicg_ws), label = "Boscia + Hungarian + DICG + WarmStart", color = colors[5], linestyle = linestyle[3], marker = markers[1])
		end

		if "custom_dicg_lazy_ws" in modes
			df_custom_dicg_lazy_ws = deepcopy(df)
			filter!(row -> !(row.terminationBoscia_Custom_DICG_Lazy_WS == 0), df_custom_dicg_lazy_ws)
			x_custom_dicg_lazy_ws = sort(df_custom_dicg_lazy_ws[!, "timeBoscia_Custom_DICG_Lazy_WS"])
			ax.plot(x_custom_dicg_lazy_ws, 1:nrow(df_custom_dicg_lazy_ws), label = "Boscia + Hungarian + DICG + Lazy + WarmStart", color = colors[6], linestyle = linestyle[1], marker = markers[1])
		end

		if "mip" in modes
			df_mip = deepcopy(df)
			filter!(row -> !(row.terminationBoscia_MIP == 0), df_mip)
			x_mip = sort(df_mip[!, "timeBoscia_MIP"])
			ax.plot(x_mip, 1:nrow(df_mip), label = "Boscia + MIP SCIP", color = colors[6], linestyle = linestyle[3], marker = markers[3])
		end

		if "mip_lazy" in modes
			df_mip_lazy = deepcopy(df)
			filter!(row -> !(row.terminationBoscia_MIP_Lazy == 0), df_mip_lazy)
			x_mip_lazy = sort(df_mip_lazy[!, "timeBoscia_MIP_Lazy"])
			ax.plot(x_mip_lazy, 1:nrow(df_mip_lazy), label = "Boscia + MIP SCIP + Lazy", color = colors[2], linestyle = linestyle[1], marker = markers[2])
		end

		if "mip_dicg" in modes
			df_mip_dicg = deepcopy(df)
			filter!(row -> !(row.terminationBoscia_MIP_DICG == 0), df_mip_dicg)
			x_mip_dicg = sort(df_mip_dicg[!, "timeBoscia_MIP_DICG"])
			ax.plot(x_mip_dicg, 1:nrow(df_mip_dicg), label = "Boscia + MIP SCIP + DICG", color = colors[5], linestyle = linestyle[4], marker = markers[4])
		end

		if "mip_dicg_lazy" in modes
			df_mip_dicg_lazy = deepcopy(df)
			filter!(row -> !(row.terminationBoscia_MIP_DICG_Lazy == 0), df_mip_dicg_lazy)
			x_mip_dicg_lazy = sort(df_mip_dicg_lazy[!, "timeBoscia_MIP_DICG_Lazy"])
			ax.plot(x_mip_dicg_lazy, 1:nrow(df_mip_dicg_lazy), label = "Boscia + MIP SCIP + DICG + Lazy", color = colors[4], linestyle = linestyle[2], marker = markers[4])
		end

		if "mip_dicg_lazy_ws" in modes
			df_mip_dicg_lazy_ws = deepcopy(df)
			filter!(row -> !(row.terminationBoscia_MIP_DICG_Lazy_WS == 0), df_mip_dicg_lazy_ws)
			x_mip_dicg_lazy_ws = sort(df_mip_dicg_lazy_ws[!, "timeBoscia_MIP_DICG_Lazy_WS"])
			ax.plot(x_mip_dicg_lazy_ws, 1:nrow(df_mip_dicg_lazy_ws), label = "Boscia + MIP SCIP + DICG + Lazy + WarmStart", color = colors[2], linestyle = linestyle[2], marker = markers[3])
		end

		if "mip_dicg_ws" in modes
			df_mip_dicg_ws = deepcopy(df)
			filter!(row -> !(row.terminationBoscia_MIP_DICG_WS == 0), df_mip_dicg_ws)
			x_mip_dicg_ws = sort(df_mip_dicg_ws[!, "timeBoscia_MIP_DICG_WS"])
			ax.plot(x_mip_dicg_ws, 1:nrow(df_mip_dicg_ws), label = "Boscia + MIP SCIP + DICG + WarmStart", color = colors[6], linestyle = linestyle[3], marker = markers[1])
		end

		ax.grid()
		#ax.set_xscale("log")

		ax.legend(loc = "lower right")#, bbox_to_anchor=(0.5, -0.3), fontsize=12,fancybox=true, shadow=false, ncol=2) 

		ylabel("Solved to optimality")
		xlabel("Time (s)")

		fig.tight_layout()

		title = join(modes, " vs ")

		file_name = joinpath(@__DIR__, "plots/" * title * "_plot_birkhoff_termination.pdf")
	else
		df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/birkhoff_comparison_summary_by_dimension.csv")))

		fig, axs = plt.subplots(2, sharex = true, sharey = false, figsize = (6.5, 5.5))

		colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
		markers = ["o", "s", "^", "P", "X", "H", "D"]
		linestyle = ["-", ":", "-.", "--"]

		PyPlot.matplotlib[:rc]("text", usetex = true)
		PyPlot.matplotlib[:rc]("font", size = 11, family = "cursive")
		PyPlot.matplotlib[:rc]("axes", labelsize = 14)
		PyPlot.matplotlib[:rc]("text.latex", preamble = raw"""
		  \usepackage{libertine}
		  \usepackage{libertinust1math}
		  """)

		linewidth = 2

		if "custom" in modes
			axs[1].plot(df[!, "Dimension"], df[!, "Boscia_CustomTerm"], label = "Hungarian", color = colors[2], linestyle = linestyle[1], marker = markers[1])
		end

		if "custom_lazy" in modes
			axs[1].plot(df[!, "Dimension"], df[!, "Boscia_Custom_LazyTerm"], label = "Hungarian Lazy", color = colors[3], linestyle = linestyle[1], marker = markers[4])
		end

		if "custom_dicg" in modes
			axs[1].plot(df[!, "Dimension"], df[!, "Boscia_Custom_DICGTerm"], label = "Hungarian DICG", color = colors[4], linestyle = linestyle[2], marker = markers[2])
		end

		if "mip" in modes
			axs[1].plot(df[!, "Dimension"], df[!, "Boscia_MIPTerm"], label = "MIP SCIP", color = colors[6], linestyle = linestyle[3], marker = markers[3])
		end

		if "mip_lazy" in modes
			axs[1].plot(df[!, "Dimension"], df[!, "Boscia_MIP_LazyTerm"], label = "MIP SCIP Lazy", color = colors[2], linestyle = linestyle[1], marker = markers[2])
		end

		if "mip_dicg" in modes
			axs[1].plot(df[!, "Dimension"], df[!, "Boscia_MIP_DICGTerm"], label = "MIP SCIP DICG", color = colors[5], linestyle = linestyle[4], marker = markers[4])
		end

		if "custom_dicg_lazy" in modes
			axs[1].plot(df[!, "Dimension"], df[!, "Boscia_Custom_DICG_LazyTerm"], label = "Hungarian DICG Lazy", color = colors[7], linestyle = linestyle[1], marker = markers[1])
		end

		if "custom_dicg_ws" in modes
			axs[1].plot(df[!, "Dimension"], df[!, "Boscia_Custom_DICG_WSTerm"], label = "Hungarian DICG WS", color = colors[5], linestyle = linestyle[3], marker = markers[1])
		end

		if "custom_dicg_lazy_ws" in modes
			axs[1].plot(df[!, "Dimension"], df[!, "Boscia_Custom_DICG_Lazy_WSTerm"], label = "Hungarian DICG Lazy WS", color = colors[6], linestyle = linestyle[1], marker = markers[1])
		end

		if "mip_dicg_lazy" in modes
			axs[1].plot(df[!, "Dimension"], df[!, "Boscia_MIP_DICG_LazyTerm"], label = "MIP SCIP DICG Lazy", color = colors[4], linestyle = linestyle[2], marker = markers[4])
		end

		if "mip_dicg_lazy_ws" in modes
			axs[1].plot(df[!, "Dimension"], df[!, "Boscia_MIP_DICG_Lazy_WSTerm"], label = "MIP SCIP DICG Lazy WS", color = colors[2], linestyle = linestyle[2], marker = markers[3])
		end

		if "mip_dicg_ws" in modes
			axs[1].plot(df[!, "Dimension"], df[!, "Boscia_MIP_DICG_WSTerm"], label = "MIP SCIP DICG WS", color = colors[6], linestyle = linestyle[3], marker = markers[1])
		end


		axs[1].grid()
		if "custom" in modes
			axs[2].plot(df[!, "Dimension"], df[!, "Boscia_CustomTime"], label = "Hungarian", color = colors[2], linestyle = linestyle[1], marker = markers[1])
		end

		if "custom_lazy" in modes
			axs[2].plot(df[!, "Dimension"], df[!, "Boscia_Custom_LazyTime"], label = "Hungarian Lazy", color = colors[3], linestyle = linestyle[1], marker = markers[4])
		end

		if "custom_dicg" in modes
			axs[2].plot(df[!, "Dimension"], df[!, "Boscia_Custom_DICGTime"], label = "Hungarian DICG", color = colors[4], linestyle = linestyle[2], marker = markers[2])
		end

		if "mip" in modes
			axs[2].plot(df[!, "Dimension"], df[!, "Boscia_MIPTime"], label = "MIP SCIP", color = colors[6], linestyle = linestyle[3], marker = markers[3])
		end

		if "mip_lazy" in modes
			axs[2].plot(df[!, "Dimension"], df[!, "Boscia_MIP_LazyTime"], label = "MIP SCIP Lazy", color = colors[2], linestyle = linestyle[1], marker = markers[2])
		end

		if "mip_dicg" in modes
			axs[2].plot(df[!, "Dimension"], df[!, "Boscia_MIP_DICGTime"], label = "MIP SCIP DICG", color = colors[5], linestyle = linestyle[4], marker = markers[4])
		end

		if "custom_dicg_lazy" in modes
			axs[2].plot(df[!, "Dimension"], df[!, "Boscia_Custom_DICG_LazyTime"], label = "Hungarian DICG Lazy", color = colors[7], linestyle = linestyle[1], marker = markers[1])
		end

		if "custom_dicg_ws" in modes
			axs[2].plot(df[!, "Dimension"], df[!, "Boscia_Custom_DICG_WSTime"], label = "Hungarian DICG WS", color = colors[5], linestyle = linestyle[3], marker = markers[1])
		end

		if "custom_dicg_lazy_ws" in modes
			axs[2].plot(df[!, "Dimension"], df[!, "Boscia_Custom_DICG_Lazy_WSTime"], label = "Hungarian DICG Lazy WS", color = colors[6], linestyle = linestyle[1], marker = markers[1])
		end

		if "mip_dicg_lazy" in modes
			axs[2].plot(df[!, "Dimension"], df[!, "Boscia_MIP_DICG_LazyTime"], label = "MIP SCIP DICG Lazy", color = colors[4], linestyle = linestyle[2], marker = markers[4])
		end

		if "mip_dicg_ws" in modes
			axs[2].plot(df[!, "Dimension"], df[!, "Boscia_MIP_DICG_WSTime"], label = "MIP SCIP DICG WS", color = colors[6], linestyle = linestyle[3], marker = markers[1])
		end

		if "mip_dicg_lazy_ws" in modes
			axs[2].plot(df[!, "Dimension"], df[!, "Boscia_MIP_DICG_Lazy_WSTime"], label = "MIP SCIP DICG Lazy WS", color = colors[2], linestyle = linestyle[2], marker = markers[3])
		end


		axs[2].grid()
		axs[2].legend(loc = "lower right")#, bbox_to_anchor=(0.5, -0.3), fontsize=12,fancybox=true, shadow=false, ncol=2) 

		axs[1].set_ylabel("Solved to optimality", loc = "center")
		axs[2].set_ylabel("Average time", loc = "center")
		xlabel("Dimension n")

		fig.tight_layout()

		title = join(modes, " vs ")

		file_name = joinpath(@__DIR__, "plots/" * title * "_plot_birkhoff_by_dimension.pdf")
	end
	PyPlot.savefig(file_name)
end


modes =
	[
		["custom_lazy", "custom_dicg", "mip_lazy", "mip_dicg", "custom_dicg_lazy"],
		["custom_dicg", "custom_dicg_lazy", "custom_dicg_ws", "custom_dicg_lazy_ws"],
		["mip_dicg", "mip_dicg_lazy", "mip_dicg_ws", "mip_dicg_lazy_ws"],
		["custom_lazy", "custom", "mip", "mip_lazy"],
		["custom_lazy", "custom_dicg_lazy"],]



plot_term(modes[1];)
plot_term(modes[1]; by_time = true)

plot_term(modes[2];)
plot_term(modes[2]; by_time = true)

plot_term(modes[3];)
plot_term(modes[3]; by_time = true)

plot_term(modes[4];)
plot_term(modes[4]; by_time = true)

plot_term(modes[5];)
plot_term(modes[5]; by_time = true)