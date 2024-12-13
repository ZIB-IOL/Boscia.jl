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



# For bug hunting:
seed = rand(UInt64)
@show seed
Random.seed!(seed)

include("co_localization_BLMO.jl")





function build_objective()

	param_mu = 0.4
	param_lambda = 0.1

	linVector = CSV.read("./data/qp_linVector.csv", DataFrame; header = false) |> Matrix
	lapMatrix = CSV.read("./data/qp_lapMatrix.csv", DataFrame; header = false) |> Matrix
	rrMatrix = CSV.read("./data/qp_rrMatrix.csv", DataFrame; header = false) |> Matrix
	A = lapMatrix + param_mu * rrMatrix
	@show size(A)
	b = param_lambda * linVector

	function f(x)

		return (1 / 2) * (transpose(x) * A * x) + sum(transpose(b) * x)
	end

	function grad!(storage, x)
		@. storage = $(A * x) + b
	end

	return f, grad!
end






function run_args_boscia(mode, f, grad!, lmo, verbose, time_limit; variant = Boscia.BPCG(), fw_epsilon = 1e-2, dual_gap = 1e-6, rel_dual_gap = 1e-2, min_node_fw_epsilon = 1e-6)
	if contains(mode, "lazy")
		if contains(mode, "ws")
			if contains(mode, "full")
				x, _, result =
					Boscia.solve(
						f,
						grad!,
						lmo,
						verbose = verbose,
						time_limit = time_limit,
						lazy = true,
						variant = variant,
						use_DICG_warm_start = true,
						use_full_ws = true,
						fw_epsilon = fw_epsilon,
						dual_gap = dual_gap,
						rel_dual_gap = rel_dual_gap,
						min_node_fw_epsilon = min_node_fw_epsilon,
					)
			else
				x, _, result =
					Boscia.solve(
						f,
						grad!,
						lmo,
						verbose = verbose,
						time_limit = time_limit,
						lazy = true,
						variant = variant,
						use_DICG_warm_start = true,
						use_full_ws = false,
						fw_epsilon = fw_epsilon,
						dual_gap = dual_gap,
						rel_dual_gap = rel_dual_gap,
						min_node_fw_epsilon = min_node_fw_epsilon,
					)
			end
		else
			x, _, result = Boscia.solve(
				f,
				grad!,
				lmo,
				verbose = verbose,
				time_limit = time_limit,
				lazy = true,
				variant = variant,
				use_DICG_warm_start = false,
				fw_epsilon = fw_epsilon,
				dual_gap = dual_gap,
				rel_dual_gap = rel_dual_gap,
				min_node_fw_epsilon = min_node_fw_epsilon,
			)
		end
	else
		if contains(mode, "ws")
			if contains(mode, "full")
				x, _, result =
					Boscia.solve(
						f,
						grad!,
						lmo,
						verbose = verbose,
						time_limit = time_limit,
						lazy = false,
						variant = variant,
						use_DICG_warm_start = true,
						use_full_ws = true,
						fw_epsilon = fw_epsilon,
						dual_gap = dual_gap,
						rel_dual_gap = rel_dual_gap,
						min_node_fw_epsilon = min_node_fw_epsilon,
					)
			else
				x, _, result = Boscia.solve(
					f,
					grad!,
					lmo,
					verbose = verbose,
					time_limit = time_limit,
					lazy = false,
					variant = variant,
					use_DICG_warm_start = true,
					fw_epsilon = fw_epsilon,
					dual_gap = dual_gap,
					rel_dual_gap = rel_dual_gap,
					min_node_fw_epsilon = min_node_fw_epsilon,
				)
			end
		else
			x, _, result = Boscia.solve(
				f,
				grad!,
				lmo,
				verbose = verbose,
				time_limit = time_limit,
				lazy = false,
				variant = variant,
				use_DICG_warm_start = false,
				fw_epsilon = fw_epsilon,
				dual_gap = dual_gap,
				rel_dual_gap = rel_dual_gap,
				min_node_fw_epsilon = min_node_fw_epsilon,
			)
		end
	end

	return x, result
end

function cl_boscia(
	seed,
	boxes_per_img,
	n_imgs;
	mode = "custom",
	int_vars = nothing,
	lower_bounds = nothing,
	upper_bounds = nothing,
	dual_gap = 1e-6,
	rel_dual_gap = 1e-2,
	verbose = true,
	time_limit = 3600,
	write = true,
	fw_epsilon = 1e-2,
	min_node_fw_epsilon = 1e-6,
)
	@show seed
	Random.seed!(seed)
	dim = 630

	f, grad! = build_objective()

	sblmo = FlowPolytopeBLMO(10, 63)
	lb = zeros(dim)
	ub = ones(dim)
	int_vars = collect(1:dim)
	lmo = Boscia.ManagedBoundedLMO(sblmo, lb, ub, int_vars, dim)

	custom_modes = ["custom", "custom_lazy", "custom_dicg", "custom_dicg_lazy", "custom_dicg_ws", "custom_dicg_lazy_ws", "custom_dicg_lazy_full_ws", "custom_dicg_full_ws"]
	mip_modes = ["mip", "mip_lazy", "mip_dicg", "mip_dicg_lazy", "mip_dicg_ws", "mip_dicg_lazy_ws", "mip_dicg_lazy_full_ws", "mip_dicg_full_ws"]

	if mode in custom_modes
	elseif mode in mip_modes

	else
		error("Mode not known")
	end

	if contains(mode, "dicg")
		x, result = run_args_boscia(mode, f, grad!, lmo, verbose, time_limit; variant = Boscia.DICG(), fw_epsilon = fw_epsilon, dual_gap = dual_gap, rel_dual_gap = rel_dual_gap, min_node_fw_epsilon = min_node_fw_epsilon)
	else
		x, result = run_args_boscia(mode, f, grad!, lmo, verbose, time_limit; variant = Boscia.BPCG(), fw_epsilon = fw_epsilon, dual_gap = dual_gap, rel_dual_gap = rel_dual_gap, min_node_fw_epsilon = min_node_fw_epsilon)
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

		df_full = DataFrame(
			time = time_list / 1000,
			lowerBound = lb_list,
			upperBound = ub_list,
			termination = status,
			LMOcalls = list_lmo_calls,
			localTighteings = list_local_tightening,
			globalTightenings = list_global_tightening,
			list_active_set_size_cb = list_active_set_size_cb,
			list_discarded_set_size_cb = list_discarded_set_size_cb,
		)
		file_name_full = joinpath(
			dir_path,
			"full_run_boscia_" * mode * "_.csv",
		)
		CSV.write(file_name_full, df_full, append = false)



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
			"boscia_" * mode * "_" * ".csv",
		)
		CSV.write(file_name, df, append = false, writeheader = true)
	end
end

cl_boscia(4, 20, 33; mode = "custom_lazy", time_limit = 3600, dual_gap = 1e-6, rel_dual_gap = 1e-1, fw_epsilon = 1e-3, min_node_fw_epsilon = 1e-6)
