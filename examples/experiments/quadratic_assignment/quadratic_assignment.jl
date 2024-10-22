using Boscia
using FrankWolfe
using Test
using Random
using FiniteDifferences
using SCIP
using LinearAlgebra
using DataFrames
using CSV
using MathOptInterface: MathOptInterface
const MOI = MathOptInterface
using HiGHS: HiGHS

include("quadratic_assignment_utils.jl")
include("birkhoff_Blmo.jl")

function qa_boscia(
	seed,
	dim;
	mode = "bpcg",
	line_search = FrankWolfe.Adaptive(),
	fw_epsilon = 1e-2,
	use_DICG_warm_start = false,
	verbose = true,
	time_limit = 3600,
	write = true,
	variant = Boscia.BPCG(),
	rel_dual_gap = 1e-2, lazy = true,
)
	@show seed
	Random.seed!(seed)

	n = dim

	flow_matrix, distance_matrix, cost_matrix = generate_facility_layout_data(n)
	P, Q = generate_p_and_q_matrices(flow_matrix, distance_matrix)
	f, grad! = build_facility_objective(P, Q)
	f, grad! = build_Koopmans_Beckmann_objective(flow_matrix, distance_matrix, cost_matrix)

	lower_bounds = fill(0.0, dim^2)
	upper_bounds = fill(1.0, dim^2)

	sblmo = BirkhoffBLMO(true, dim, collect(1:dim^2))
	blmo = Boscia.ManagedBoundedLMO(sblmo, lower_bounds, upper_bounds, collect(1:dim^2), dim^2)
	x0 = build_start_point(flow_matrix, distance_matrix)
	active_set = FrankWolfe.ActiveSet([(1.0, x0)])

	# domain_oracle = function (x)
	# 	return true
	# end
	# line_search = FrankWolfe.Secant(40, 1e-8, domain_oracle)

	x, _, result =
		Boscia.solve(
			f,
			grad!,
			blmo,
			active_set = active_set,
			variant = variant,
			verbose = verbose,
			lazy = lazy,
			line_search = line_search,
			fw_epsilon = fw_epsilon,
			time_limit = time_limit,
			use_DICG_warm_start = use_DICG_warm_start,
			rel_dual_gap = rel_dual_gap,
		)


	total_time_in_sec = result[:total_time_in_sec]
	status = result[:status]
	if occursin("Optimal", result[:status])
		status = "OPTIMAL"
	elseif occursin("Time", result[:status])
		status = "TIME_LIMIT"
	end

	if write
		lb_list = result[:list_lb]
		ub_list = result[:list_ub]
		time_list = result[:list_time]
		list_lmo_calls = result[:list_lmo_calls_acc]
		list_active_set_size_cb = result[:list_active_set_size]
		list_discarded_set_size_cb = result[:list_discarded_set_size]
		list_local_tightening = result[:local_tightenings]
		list_global_tightening = result[:global_tightenings]

		dir_path = joinpath(@__DIR__, "./csv")

		mkpath(dir_path)

		# df_full = DataFrame(
		# 	time = time_list / 1000,
		# 	lowerBound = lb_list,
		# 	upperBound = ub_list,
		# 	termination = status,
		# 	LMOcalls = list_lmo_calls,
		# 	localTighteings = list_local_tightening,
		# 	globalTightenings = list_global_tightening,
		# 	list_active_set_size_cb = list_active_set_size_cb,
		# 	list_discarded_set_size_cb = list_discarded_set_size_cb,
		# )
		# file_name_full = joinpath(
		# 	dir_path,
		# 	"full_run_boscia_" * mode * "_.csv",
		# )
		# CSV.write(file_name_full, df_full, append = false)



		@show result[:primal_objective]
		df = DataFrame(
			time = total_time_in_sec,
			solution = result[:primal_objective],
			dual_gap = result[:dual_gap],
			rel_dual_gap = result[:rel_dual_gap],
			termination = status,
			ncalls = result[:lmo_calls],
		)



		file_name = joinpath(
			dir_path,
			"boscia_" * mode * "_quadratic_assignment_" * string(dim) * "_" * string(seed) * ".csv",
		)
		CSV.write(file_name, df, append = false, writeheader = true)
	end

end

# qa_boscia(4, 7; mode = "bpcg", variant = Boscia.BPCG(), lazy = true, write = false, verbose = true, fw_epsilon = 1e-2, rel_dual_gap = 0.1 * 1.1)
qa_boscia(4, 5; mode = "bpcg", variant = Boscia.DICG(), lazy = false, write = false, verbose = true, fw_epsilon = 1e-2, rel_dual_gap = 1.0 * 1.1)



