using Boscia
using Random
using Distributions
using LinearAlgebra
using FrankWolfe
using Statistics
using DataFrames
using CSV

import Boscia.build_dicg_start_point



# The function building the problem data and other structures is in a separate file.
include("oed_utils.jl")

"""
	The Optimal Experiment Design Problem consists of choosing a subset of experiments
	maximising the information gain. 
	We generate the Experiment matrix A ∈ R^{mxn} randomly. The information matrix is a linear
	map X(x) = A' * diag(x) * A. There exists different information measures Φ. 
	We concentrate on the A-criterion and D-criterion, i.e. 

		Trace(X^{-1})       (A-Criterion)
	and
		-logdet(X)          (D-Criterion).

	Consequently, the optimization problems we want to solve are

	min_x  Trace( (A' * diag(x) * A)^{-1} )
	s.t.   ∑ x_i = N                            (A-Optimal Design Problem)
		   0 ≤ x ≤ u

		   min_x  -logdet(A' * diag(x) * A)
	s.t.   ∑ x_i = N                            (D-Optimal Design Problem)
		   0 ≤ x ≤ u

	where N is our bugdet for the experiments, i.e. this is the amount of experiments
	we can perform. We set N = 3/2 * n. The upperbounds u are randomly generated. 
	
	Also, check this paper: https://arxiv.org/abs/2312.11200 and the corresponding 
	respository https://github.com/ZIB-IOL/OptimalDesignWithBoscia.

	A continuous version of the problem can be found in the examples in FrankWolfe.jl:
	https://github.com/ZIB-IOL/FrankWolfe.jl/blob/master/examples/optimal_experiment_design.jl
"""

function oed_boscia(
	seed,
	dim;
	mode = "bpcg",
	type = "D-Optimal",
	line_search = nothing,
	fw_epsilon = 1e-2,
	use_DICG_warm_start = false,
	verbose = true,
	time_limit = 3600,
	write = true,
	variant = Boscia.BPCG(),
	lazy = true,
)
	@show seed
	Random.seed!(seed)

	m = dim


	Ex_mat, n, N, ub = build_data(m)

	# sharpness constants
	σ = minimum(Ex_mat' * Ex_mat)
	λ_max = maximum(ub) * maximum([norm(Ex_mat[i, :])^2 for i ∈ 1:size(Ex_mat, 1)])
	θ = 1 / 2

	## A-Optimal Design Problem

	if type == "A-Optimal"


		# sharpness constants
		M = sqrt(λ_max^3 / n * σ^4)


		g, grad! = build_a_criterion(Ex_mat, build_safe = true)
		blmo = build_blmo(m, N, ub)
		x0, active_set = build_start_point(Ex_mat, N, ub)
		Boscia.build_dicg_start_point = build_start_point_func(Ex_mat, N)
		z = greedy_incumbent(Ex_mat, N, ub)
		domain_oracle = build_domain_oracle(Ex_mat, n)
		if mode == "dicg"
			line_search = FrankWolfe.Adaptive()
		else
			line_search = FrankWolfe.MonotonicGenericStepsize(FrankWolfe.Adaptive(), domain_oracle)
		end
		heu = Boscia.Heuristic(Boscia.rounding_hyperplane_heuristic, 0.7, :hyperplane_aware_rounding)

		x, _, result =
			Boscia.solve(
				g,
				grad!,
				blmo,
				active_set = active_set,
				start_solution = z,
				variant = variant,
				verbose = false,
				lazy = lazy,
				domain_oracle = domain_oracle,
				custom_heuristics = [heu],
				sharpness_exponent = θ,
				sharpness_constant = M,
				line_search = line_search,
				fw_epsilon = fw_epsilon,
				time_limit = 10,
				use_DICG_warm_start = use_DICG_warm_start,
			) #sharpness_exponent=θ, sharpness_constant=M,

		_, active_set = build_start_point(Ex_mat, N, ub)
		z = greedy_incumbent(Ex_mat, N, ub)

		x, _, result =
			Boscia.solve(
				g,
				grad!,
				blmo,
				active_set = active_set,
				start_solution = z,
				variant = variant,
				verbose = verbose,
				lazy = lazy,
				domain_oracle = domain_oracle,
				custom_heuristics = [heu],
				sharpness_exponent = θ,
				sharpness_constant = M,
				line_search = line_search,
				fw_epsilon = fw_epsilon,
				time_limit = time_limit,
				use_DICG_warm_start = use_DICG_warm_start,
			) #sharpness_exponent=θ, sharpness_constant=M,


	elseif type == "D-Optimal"
		## D-Optimal Design Problem

		# sharpness constants
		M = sqrt(2 * λ_max^2 / n * σ^4)

		g, grad! = build_d_criterion(Ex_mat, build_safe = true)
		blmo = build_blmo(m, N, ub)
		x0, active_set = build_start_point(Ex_mat, N, ub)
		Boscia.build_dicg_start_point = build_start_point_func(Ex_mat, N)
		z = greedy_incumbent(Ex_mat, N, ub)
		domain_oracle = build_domain_oracle(Ex_mat, n)
		if mode == "dicg"
			# line_search = FrankWolfe.Adaptive()
			line_search = FrankWolfe.Secant(domain_oracle = domain_oracle)
		else
			line_search = FrankWolfe.Secant(domain_oracle = domain_oracle)
		end
		heu = Boscia.Heuristic(Boscia.rounding_hyperplane_heuristic, 0.7, :hyperplane_aware_rounding)



		x, _, result =
			Boscia.solve(
				g,
				grad!,
				blmo,
				active_set = active_set,
				start_solution = z,
				variant = variant,
				verbose = false,
				lazy = lazy,
				fw_epsilon = fw_epsilon,
				domain_oracle = domain_oracle,
				custom_heuristics = [heu],
				sharpness_exponent = θ,
				time_limit = 10,
				sharpness_constant = M,
				line_search = line_search,
				use_DICG_warm_start = use_DICG_warm_start,
			) #sharpness_exponent=θ, sharpness_constant=M,

		blmo = build_blmo(m, N, ub)
		x0, active_set = build_start_point(Ex_mat, N, ub)
		z = greedy_incumbent(Ex_mat, N, ub)
		domain_oracle = build_domain_oracle(Ex_mat, n)
		heu = Boscia.Heuristic(Boscia.rounding_hyperplane_heuristic, 0.7, :hyperplane_aware_rounding)

		x, _, result =
			Boscia.solve(
				g,
				grad!,
				blmo,
				active_set = active_set,
				start_solution = z,
				variant = variant,
				verbose = verbose,
				fw_verbose = true,
				lazy = lazy,
				fw_epsilon = fw_epsilon,
				domain_oracle = domain_oracle,
				custom_heuristics = [heu], time_limit = time_limit, line_search = line_search,
				use_DICG_warm_start = use_DICG_warm_start,
			) #sharpness_exponent=θ, sharpness_constant=M,

	end

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
			"boscia_" * mode * "_oed_" * string(dim) * "_" * string(seed) * ".csv",
		)
		CSV.write(file_name, df, append = false, writeheader = true)
	end

end

oed_boscia(10, 180; mode = "bpcg", variant = Boscia.BPCG(), lazy = false, write = false)

# oed_boscia(2, 30;)
