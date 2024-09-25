using CSV
using DataFrames

function merge_csvs(; example = "birkhoff", mode = "custom", seeds = 1:10, dimensions = 55:70, time_limit = 1800.0)
	# setup df
	df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/integer/boscia_" * mode * "_" * example * "_" * string(dimensions[1]) * "_" * string(seeds[1]) * ".csv")))
	select!(df, Not(:termination))
	df[!, "termination"] = ["ALMOST_LOCALLY_SOLVED"]
	select!(df, Not(:time))
	df[!, "time"] = [time_limit]
	deleteat!(df, 1)

	# add results
	for dimension in dimensions
		for seed in seeds
			try
				df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/integer/boscia_" * mode * "_" * example * "_" * string(dimension) * "_" * string(seed) * ".csv")))
				if df[!, :termination] == "Time limit reached"
					df[!, :termination] = "TIME_LIMIT"
				end
				append!(df, df_temp)
			catch e
				println(e)
			end
		end
	end

	file_name = joinpath(@__DIR__, "csv/integer/boscia_" * mode * "_" * example * ".csv")

	CSV.write(file_name, df, append = false)

end

"""
Create the set up data like seeds, dimensions etc.
"""
function set_up_data(df, dims, seeds, example = "birkhoff")
	#seeds = collect(1:5)
	#dims = collect(3:10)

	dimensions = Int[]
	for i in dims
		append!(dimensions, fill(i, length(seeds)))
	end

	df[!, :seed] = repeat(seeds, length(dims))
	df[!, :dimension] = dimensions

	return df
end

function build_non_grouped_csv(solvers; option = "comparison", example = "birkhoff")
	"""
	Read out the time, solution etc from the individuals job files.
	"""
	function read_data(example::String, solver::String, folder; time_limit = 1800.0)
		time = []
		solution = []
		termination = []
		num_n_o_c = []
		lower_bound = []
		dual_gap = []

		@show example, solver

		df = DataFrame(CSV.File(joinpath(@__DIR__, folder * "/" * solver * "_" * example * ".csv")))

		time = df[!, :time]
		solution = df[!, :solution]
		optimality = ["OPTIMAL", "optimal", "Optimal", "Optimal (tolerance reached)", "tree.lb>primal-dual_gap", "primal>=tree.incumbent", "Optimal (tree empty)", "ALMOST_LOCALLY_SOLVED", "LOCALLY_SOLVED"]
		termination = [row in optimality ? 1 : 0 for row in df[!, :termination]]

		# All the problems are feasible.
		# If a solver returns that it isn't, it counts as non-solved.
		infeas_idx = findall(x -> x == "INFEASIBLE", df[!, :termination])
		if !isempty(infeas_idx)
			time[infeas_idx] = time_limit
		end

		if contains(solver, "boscia")
			dual_gap = df[!, :dual_gap]
			lower_bound = df[!, :solution] - df[!, :dual_gap]
			num_n_o_c = df[!, :ncalls]
		elseif solver == "ipopt"
			num_n_o_c = df[!, :num_nodes]
			time ./= 1000
		elseif solver == "scip_oa"
			num_n_o_c = df[!, :calls]
		end

		# The MIP LIB instances sometimes overshoot the time limit.
		over_time_idx = findall(x -> x > time_limit + 60, df[!, :time])
		if !isempty(over_time_idx)
			time[over_time_idx] .= time_limit
			termination[over_time_idx] .= 0
		end

		@show length(time), length(solution), length(termination), length(num_n_o_c), length(dual_gap), length(lower_bound)

		return time, solution, termination, num_n_o_c, dual_gap, lower_bound
	end

	"""
	Compute relative gap with respect to Boscia's lower bound
	"""
	function relative_gap(solution, lower_bound)
		rel_gap = []
		for (i, _) in enumerate(solution)
			if min(abs(solution[i]), abs(lower_bound[i])) == 0
				push!(rel_gap, solution[i] - lower_bound[i])
			elseif sign(lower_bound[i]) != sign(solution[i])
				push!(rel_gap, Inf)
			else
				push!(rel_gap, (solution[i] - lower_bound[i]) / min(abs(solution[i]), abs(lower_bound[i])))
			end
		end
		return rel_gap
	end

	function combine_data(df, example, solver, solver_id, minimumTime)
		time, solution, termination, num_n_o_c, dual_gap, lower_bound = read_data(example, solver_id, "csv/integer", time_limit = 3600.0)

		df[!, Symbol("time" * solver)] = time
		df[!, Symbol("solution" * solver)] = solution
		df[!, Symbol("termination" * solver)] = termination
		if occursin("Boscia", solver) #solver == "Boscia"
			df[!, Symbol("numberNodes" * solver)] = num_n_o_c
			df[!, Symbol("dualGap" * solver)] = dual_gap
			df[!, Symbol("lb" * solver)] = lower_bound

			rel_gap = relative_gap(solution, lower_bound)
		elseif solver in ["Ipopt", "ScipOA"]
			df[!, Symbol("numberNodes" * solver)] = num_n_o_c
		end

		#rel_gap = solver == "Boscia" ? relative_gap(solution, lower_bound) : relative_gap(solution, df[!,:lbBoscia])
		rel_gap = relative_gap(solution, lower_bound)
		df[!, Symbol("relGap" * solver)] = rel_gap

		minimumTime = min.(minimumTime, time)

		return df, minimumTime
	end

	if option == "comparison"
		solvers = solvers
	else
		error("Unknown option!")
	end

	@show example

	# set up data
	df = DataFrame()
	df = set_up_data(df, collect(55:70), collect(1:10), example)

	@show size(df)

	minimumTime = fill(Inf, length(df[!, :seed]))

	# read out solver data
	for solver in solvers
		if example in ["tailed_cardinality", "tailed_cardinality_sparse_log_reg"] && solver in ["Ipopt", "Pavito", "Shot"]
			continue
		end
		if solver == "Boscia_Strong_Convexity" && !contains(example, "miplib")
			continue
		end
		@show solver
		if solver == "Boscia"
			solver1 = "boscia_default"
		elseif solver == "ScipOA"
			solver1 = "scip_oa"
		else
			solver1 = lowercase(solver)
		end

		df, minimumTime = combine_data(df, example, solver, solver1, minimumTime)
	end

	#=if option == "comparison"
		scip_not_optimal = 0
		for (idx, _) in enumerate(df[!,:solutionScipOA])
			if df[idx, :terminationScipOA] == 1 && df[idx, :solutionScipOA] > df[idx,:solutionBoscia] + 1e-2
			   # @show df[idx, :solutionBoscia], df[idx,:solutionScipOA], df[idx,:terminationScipOA]
				df[idx,:terminationScipOA] = 0
				df[idx,:timeScipOA] = 1800.0
				scip_not_optimal += 1
			end
		end
		if scip_not_optimal != 0
			println("\n SCIP FALSE POSITIVES")
			@show scip_not_optimal
			println("\n")
		end

	end =#

	df[!, :minimumTime] = minimumTime

	file_name = joinpath(@__DIR__, "csv/integer/" * example * "_" * option * "_non_grouped.csv")
	CSV.write(file_name, df, append = false)
	println("\n")
end

function build_grouped_csv(solvers; example = "birkhoff", option = "comparison", by_time = false)

	function geo_mean(group)
		prod = 1.0
		n = 0
		if isempty(group)
			return -1
		end
		for element in group
			# @show element
			if element != Inf
				prod = prod * abs(element)
				n += 1
			end
		end
		if n == 0
			return Inf
		end
		return prod^(1 / n)
	end

	function geom_shifted_mean(xs; shift = big"1.0")
		a = length(xs)
		n = 0
		prod = 1.0
		if a != 0
			for xi in xs
				if xi != Inf
					prod = prod * (xi + shift)
					n += 1
				end
			end
			return Float64(prod^(1 / n) - shift)
		end
		return Inf
	end

	function custom_mean(group)
		sum = 0.0
		n = 0
		dash = false

		if isempty(group)
			return -1
		end
		for element in group
			if element == "-"
				dash = true
				continue
			end
			if element != Inf
				if typeof(element) == String7 || typeof(element) == String3
					element = parse(Float64, element)
				end
				sum += element
				n += 1
			end
		end
		if n == 0
			return dash ? "-" : Inf
		end
		return sum / n
	end

	function summarize_by_time(example, timeslots, solver, option)
		num_instances = []
		term = []
		term_rel = []
		time = []
		num_nodes = []
		rel_gap_nt = []

		@show solver

		df_ng = DataFrame(CSV.File(joinpath(@__DIR__, "csv/integer/" * example * "_" * option * "_non_grouped.csv")))

		termination = findall(x -> x == 1, df_ng[!, Symbol("termination" * solver)])

		for timeslot in timeslots
			instances = findall(x -> x > timeslot, df_ng[!, :minimumTime])
			push!(num_instances, length(instances))

			term_in_time = intersect(instances, termination)
			push!(term, length(term_in_time))

			push!(term_rel, length(term_in_time) / length(instances) * 100)
			push!(time, geom_shifted_mean(df_ng[instances, Symbol("time" * solver)]))
			if contains(solver, "Boscia") || solver in ["Ipopt", "ScipOA"]
				push!(num_nodes, custom_mean(df_ng[instances, Symbol("numberNodes" * solver)]))
			end

			# notSolved = intersect(instances, notAllSolved)
			if isempty(instances) #isempty(notSolved)
				push!(rel_gap_nt, NaN)
			else
				push!(rel_gap_nt, custom_mean(df_ng[instances, Symbol("relGap" * solver)]))
			end
		end

		# rounding
		non_inf = findall(isfinite, rel_gap_nt)
		rel_gap_nt[non_inf] = round.(rel_gap_nt[non_inf], digits = 2)
		non_inf = findall(isfinite, time)
		time[non_inf] = round.(time[non_inf], digits = 2)
		non_inf = findall(isfinite, num_nodes)
		num_nodes[non_inf] = convert.(Int64, round.(num_nodes[non_inf]))
		non_inf = findall(isfinite, term_rel)
		term_rel[non_inf] = convert.(Int64, round.(term_rel[non_inf]))
		term_rel[non_inf] = string.(term_rel[non_inf]) .* " %"

		println("\n")

		return num_instances, term, term_rel, time, num_nodes, rel_gap_nt
	end

	function summarize_by_dim(example, dims, solver, option)
		num_instances = []
		term = []
		term_rel = []
		time = []
		num_nodes = []
		rel_gap_nt = []

		@show solver

		df_ng = DataFrame(CSV.File(joinpath(@__DIR__, "csv/integer/" * example * "_" * option * "_non_grouped.csv")))

		termination = findall(x -> x == 1, df_ng[!, Symbol("termination" * solver)])

		for dim in dims
			instances = findall(x -> x == dim, df_ng[!, :dimension])
			push!(num_instances, length(instances))

			term_in_time = intersect(instances, termination)
			push!(term, length(term_in_time))

			push!(term_rel, length(term_in_time) / length(instances) * 100)
			push!(time, geom_shifted_mean(df_ng[instances, Symbol("time" * solver)]))
			if contains(solver, "Boscia") || solver in ["Ipopt", "ScipOA"]
				push!(num_nodes, custom_mean(df_ng[instances, Symbol("numberNodes" * solver)]))
			end

			# notSolved = intersect(instances, notAllSolved)
			if isempty(instances) #isempty(notSolved)
				push!(rel_gap_nt, NaN)
			else
				push!(rel_gap_nt, custom_mean(df_ng[instances, Symbol("relGap" * solver)]))
			end
		end

		# rounding
		non_inf = findall(isfinite, rel_gap_nt)
		rel_gap_nt[non_inf] = round.(rel_gap_nt[non_inf], digits = 2)
		non_inf = findall(isfinite, time)
		time[non_inf] = round.(time[non_inf], digits = 2)
		non_inf = findall(isfinite, num_nodes)
		num_nodes[non_inf] = convert.(Int64, round.(num_nodes[non_inf]))
		non_inf = findall(isfinite, term_rel)
		term_rel[non_inf] = convert.(Int64, round.(term_rel[non_inf]))
		term_rel[non_inf] = string.(term_rel[non_inf]) .* " %"

		println("\n")

		return num_instances, term, term_rel, time, num_nodes, rel_gap_nt
	end

	time_slots = [0, 10, 60, 300, 600, 1200, 1800, 2700]
	dimensions = collect(55:70)

	df = DataFrame()
	@show size(df)

	for (i, solver) in enumerate(solvers)
		if example in ["tailed_cardinality", "tailed_cardinality_sparse_log_reg"] && solver in ["Ipopt", "Pavito", "Shot"]
			continue
		end
		if solver == "Boscia_Strong_Convexity" && !contains(example, "miplib")
			continue
		end
		if by_time
			num_instances, num_terminated, rel_terminated, m_time, m_nodes_cuts, rel_gap_nt = summarize_by_time(example, time_slots, solver, option)
		else
			num_instances, num_terminated, rel_terminated, m_time, m_nodes_cuts, rel_gap_nt = summarize_by_dim(example, dimensions, solver, option)
		end

		if i == 1
			if by_time
				df[!, :minTime] = time_slots
			else
				df[!, :Dimension] = dimensions
			end
			df[!, :numInstances] = num_instances
			@show length(df[!, :numInstances])
		end
		@show length(num_terminated)
		df[!, Symbol(solver * "Term")] = num_terminated
		df[!, Symbol(solver * "TermRel")] = rel_terminated
		df[!, Symbol(solver * "Time")] = m_time
		df[!, Symbol(solver * "RelGapNT")] = rel_gap_nt

		if contains(solver, "Boscia") || solver in ["Ipopt", "ScipOA"]
			df[!, Symbol(solver * "NodesOCuts")] = m_nodes_cuts
		end
	end

	summary_by_what = by_time ? "difficulty" : "dimension"
	file_name = joinpath(@__DIR__, "csv/integer/" * example * "_" * option * "_summary_by_" * summary_by_what * ".csv")
	CSV.write(file_name, df, append = false)
	println("\n")
end

solvers1 = [
	"Boscia_Custom",
	"Boscia_Custom_Lazy",
	"Boscia_Custom_DICG",
	"Boscia_MIP",
	"Boscia_MIP_Lazy",
	"Boscia_MIP_DICG",
	"Boscia_Custom_DICG_Lazy",
	"Boscia_MIP_DICG_Lazy",
	"Boscia_Custom_DICG_WS",
	"Boscia_Custom_DICG_Lazy_WS",
	"Boscia_MIP_DICG_Lazy_WS",
	"Boscia_MIP_DICG_WS",
]

modes1 = [
	"custom",
	"custom_lazy",
	"mip",
	"mip_lazy",
	"custom_dicg",
	"custom_dicg_lazy",
	"custom_dicg_ws",
	"custom_dicg_lazy_ws",
	"mip_dicg",
	"mip_dicg_lazy",
	"mip_dicg_ws",
	"mip_dicg_lazy_ws",
]

solvers2 = [
	"Boscia_BLMO_Lazy",
	"Boscia_BLMO_DICG",
	"Boscia_MIP_Lazy",
	"Boscia_MIP_DICG",
	"Boscia_BLMO_DICG_Lazy",
	"Boscia_MIP_DICG_Lazy",
	"Boscia_BLMO_DICG_WS",
	"Boscia_BLMO_DICG_Lazy_WS",
	"Boscia_MIP_DICG_Lazy_WS",
	"Boscia_MIP_DICG_WS",
]

modes2 = [
	"blmo_lazy",
	"mip_lazy",
	"blmo_dicg",
	"blmo_dicg_lazy",
	"blmo_dicg_ws",
	"blmo_dicg_lazy_ws",
	"mip_dicg",
	"mip_dicg_lazy",
	"mip_dicg_ws",
	"mip_dicg_lazy_ws",
]

for mode in modes2
	# Compile the results and evaluate
	merge_csvs(mode = mode, time_limit = 3600.0; example = "approx_planted_point")
end


build_non_grouped_csv(solvers2; example = "approx_planted_point")
build_grouped_csv(solvers2; example = "approx_planted_point")
build_grouped_csv(solvers2; by_time = true, example = "approx_planted_point")
