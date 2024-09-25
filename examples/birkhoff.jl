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

include("birkhoff_Blmo.jl")

"""
Check if the gradient using finite differences matches the grad! provided.
Copied from FrankWolfe package: https://github.com/ZIB-IOL/FrankWolfe.jl/blob/master/examples/plot_utils.jl
"""
function check_gradients(grad!, f, gradient, num_tests = 10, tolerance = 1.0e-5)
	for i in 1:num_tests
		random_point = rand(length(gradient))
		grad!(gradient, random_point)
		if norm(grad(central_fdm(5, 1), f, random_point)[1] - gradient) > tolerance
			@warn "There is a noticeable difference between the gradient provided and
			the gradient computed using finite differences.:\n$(norm(grad(central_fdm(5, 1), f, random_point)[1] - gradient))"
			return false
		end
	end
	return true
end

# min_{X} 1/2 * || X - Xhat ||_F^2
# X ∈ P_n (permutation matrix)

#n = 8

function build_objective(n, append_by_column = true)
	# generate random doubly stochastic matrix
	Xstar = rand(n, n)
	while norm(sum(Xstar, dims = 1) .- 1) > 1e-6 || norm(sum(Xstar, dims = 2) .- 1) > 1e-6
		Xstar ./= sum(Xstar, dims = 1)
		Xstar ./= sum(Xstar, dims = 2)
	end

	function f(x)
		X = append_by_column ? reshape(x, (n, n)) : transpose(reshape(x, (n, n)))
		return 1 / 2 * LinearAlgebra.tr(LinearAlgebra.transpose(X .- Xstar) * (X .- Xstar))
	end

	function grad!(storage, x)
		X = append_by_column ? reshape(x, (n, n)) : transpose(reshape(x, (n, n)))
		storage .= if append_by_column
			reduce(vcat, X .- Xstar)
		else
			reduce(vcat, LinearAlgebra.transpose(X .- Xstar))
		end
		#storage .= X .- Xstar
		return storage
	end

	return f, grad!
end


function build_birkhoff_mip(
	n;
	lower_bounds = nothing,
	upper_bounds = nothing,
	int_vars = nothing,
	append_by_column = true,
)
	o = SCIP.Optimizer()
	MOI.set(o, MOI.Silent(), true)
	MOI.empty!(o)
	if append_by_column
		X = reshape(MOI.add_variables(o, n^2), n, n)
	else
		X = transpose(reshape(MOI.add_variables(o, n^2), n, n))
	end

	MOI.add_constraint.(o, X, MOI.ZeroOne())
	# doubly stochastic constraints
	MOI.add_constraint.(
		o,
		vec(sum(X, dims = 1, init = MOI.ScalarAffineFunction{Float64}([], 0.0))),
		MOI.EqualTo(1.0),
	)
	MOI.add_constraint.(
		o,
		vec(sum(X, dims = 2, init = MOI.ScalarAffineFunction{Float64}([], 0.0))),
		MOI.EqualTo(1.0),
	)
	for idx in X
		idx = idx.value
		if idx in int_vars
			loc = findfirst(x -> x == idx, int_vars)
			lb = lower_bounds[loc]
			ub = upper_bounds[loc]
			if append_by_column
				j = ceil(Int, idx / n)
				i = Int(idx - n * (j - 1))
				MOI.add_constraint(o, X[i, j], MOI.ZeroOne())
				MOI.add_constraint(o, X[i, j], MOI.GreaterThan(lb))
				MOI.add_constraint(o, X[i, j], MOI.LessThan(ub))
			else
				i = ceil(Int, idx / n)
				j = Int(idx - n * (j - 1))
				MOI.add_constraint(o, X[i, j], MOI.ZeroOne())
				MOI.add_constraint(o, X[i, j], MOI.GreaterThan(lower_bounds[loc]))
				MOI.add_constraint(o, X[i, j], MOI.LessThan(upper_bounds[loc]))

			end
		else
			if append_by_column
				j = ceil(Int, idx / n)
				i = Int(idx - n * (j - 1))
				MOI.add_constraint(o, X[i, j], MOI.GreaterThan(0.0))
				MOI.add_constraint(o, X[i, j], MOI.LessThan(1.0))
			else
				i = ceil(Int, idx / n)
				j = Int(idx - n * (j - 1))
				MOI.add_constraint(o, X[i, j], MOI.GreaterThan(0.0))
				MOI.add_constraint(o, X[i, j], MOI.LessThan(1.0))
			end
		end
	end
	return Boscia.MathOptBLMO(o)
end

function run_args_boscia(mode, f, grad!, lmo, verbose, time_limit; variant = Boscia.BPCG())
	if contains(mode, "lazy")
		if contains(mode, "ws")
			if contains(mode, "full")
				x, _, result = Boscia.solve(f, grad!, lmo, verbose = verbose, time_limit = time_limit, lazy = true, variant = variant, use_DICG_warm_start = true, use_full_ws = true)
			else
				x, _, result = Boscia.solve(f, grad!, lmo, verbose = verbose, time_limit = time_limit, lazy = true, variant = variant, use_DICG_warm_start = true, use_full_ws = false)
			end
		else
			x, _, result = Boscia.solve(f, grad!, lmo, verbose = verbose, time_limit = time_limit, lazy = true, variant = variant, use_DICG_warm_start = false)
		end
	else
		if contains(mode, "ws")
			if contains(mode, "full")
				x, _, result = Boscia.solve(f, grad!, lmo, verbose = verbose, time_limit = time_limit, lazy = false, variant = variant, use_DICG_warm_start = true, use_full_ws = true)
			else
				x, _, result = Boscia.solve(f, grad!, lmo, verbose = verbose, time_limit = time_limit, lazy = false, variant = variant, use_DICG_warm_start = true)
			end
		else
			x, _, result = Boscia.solve(f, grad!, lmo, verbose = verbose, time_limit = time_limit, lazy = false, variant = variant, use_DICG_warm_start = false)
		end
	end

	return x, result
end

function birkhoff_boscia(seed, dim; mode = "custom", int_vars = nothing, lower_bounds = nothing, upper_bounds = nothing, verbose = true, time_limit = 3600, write = true)
	@show seed
	Random.seed!(seed)

	f, grad! = build_objective(dim)
	lower_bounds = fill(0.0, dim^2)
	upper_bounds = fill(1.0, dim^2)

	custom_modes = ["custom", "custom_lazy", "custom_dicg", "custom_dicg_lazy", "custom_dicg_ws", "custom_dicg_lazy_ws", "custom_dicg_lazy_full_ws", "custom_dicg_full_ws"]
	mip_modes = ["mip", "mip_lazy", "mip_dicg", "mip_dicg_lazy", "mip_dicg_ws", "mip_dicg_lazy_ws", "mip_dicg_lazy_full_ws", "mip_dicg_full_ws"]

	if mode in custom_modes
		sblmo = BirkhoffBLMO(true, dim, collect(1:dim^2))
		lmo = Boscia.ManagedBoundedLMO(sblmo, lower_bounds, upper_bounds, collect(1:dim^2), dim^2)
	elseif mode in mip_modes
		lmo = build_birkhoff_mip(dim; lower_bounds = lower_bounds, upper_bounds = upper_bounds, int_vars = collect(1:dim^2))
	else
		error("Mode not known")
	end

	if contains(mode, "dicg")
		x, result = run_args_boscia(mode, f, grad!, lmo, verbose, time_limit; variant = Boscia.DICG())
	else
		x, result = run_args_boscia(mode, f, grad!, lmo, verbose, time_limit)
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
		"""
		df_full = DataFrame(
			seed = seed,
			dimension = dim,
			time = time_list,
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
			@__DIR__,
			"csv/full_run_boscia_" *
			mode *
			"_" *
			string(dim) *
			"_" *
			string(seed) *
			"_birkhoff.csv",
		)
		CSV.write(file_name_full, df_full, append = false)
		"""


		@show result[:primal_objective]
		df = DataFrame(
			seed = seed,
			dimension = dim,
			time = total_time_in_sec,
			solution = result[:primal_objective],
			dual_gap = result[:dual_gap],
			rel_dual_gap = result[:rel_dual_gap],
			termination = status,
			ncalls = result[:lmo_calls],
		)
		file_name = joinpath(
			@__DIR__,
			"csv/boscia_" * mode * "_birkhoff_" * string(dim) * "_" * string(seed) * ".csv",
		)
		CSV.write(file_name, df, append = false, writeheader = true)
	end
end

# birkhoff_boscia(8, 7; mode = "custom_dicg_lazy")

"""
@testset "Birkhoff" begin
	n = 7
	f, grad! = build_objective(n)

	#=@testset "Test Derivative" begin
		gradient = rand(n^2)
		@test check_gradients(grad!, f, gradient)
	end=#

	x = zeros(n, n)
	lower_bounds = fill(0.0, n^2)
	upper_bounds = fill(1.0, n^2)
	@testset "Custom BLMO" begin
		lower_bounds = fill(0.0, n^2)
		upper_bounds = fill(1.0, n^2)
		sblmo = BirkhoffBLMO(true, n, collect(1:n^2))

		#=x_, _, result = Boscia.solve(
			f,
			grad!,
			sblmo,
			lower_bounds,
			upper_bounds,
			collect(1:n^2),
			n^2,
			verbose=true,
			variant=Boscia.DICG(),
			lazy=false,
		)=#

		x, _, result = Boscia.solve(
			f,
			grad!,
			sblmo,
			lower_bounds,
			upper_bounds,
			collect(1:n^2),
			n^2,
			verbose=true,
		)
		
		#@test f(x) <= f(result[:raw_solution]) + 1e-6
		#@test Boscia.is_simple_linear_feasible(sblmo, x)
		#@test f(x_) <= f(result[:raw_solution]) + 1e-6
		#@test Boscia.is_simple_linear_feasible(sblmo, x_)
	end

	#=x_mip = zeros(n, n)
	@testset "MIP BLMO" begin
		lmo = build_birkhoff_mip(
			n;
			lower_bounds=lower_bounds,
			upper_bounds=upper_bounds,
			int_vars=collect(1:n^2),
		)

		x_mip, _, result_mip = Boscia.solve(f, grad!, lmo, verbose=true)
		@test f(x_mip) <= f(result_mip[:raw_solution]) + 1e-6
		@test Boscia.is_linear_feasible(lmo, x_mip)
	end

	@show x
	@show x_mip
	@show f(x), f(x_mip)
	#@test isapprox(f(x_mip), f(x), atol=1e-6, rtol=1e-2)=#
end

#=@testset "Birkhoff polytope" begin
	n = 4
	d = randn(n, n)
	int_vars = collect(1:n^2)
	lower_bounds = fill(0.0, n^2)
	upper_bounds = fill(1.0, n^2)
	sblmo = BirkhoffBLMO(true, n, int_vars)
	lmo = build_birkhoff_mip(n; lower_bounds=lower_bounds, upper_bounds=upper_bounds, int_vars=int_vars)
	x = ones(n, n) ./ n
	# test without fixings
	v_if = Boscia.bounded_compute_inface_extreme_point(sblmo, d, x, lower_bounds, upper_bounds, int_vars)
	v_fw = Boscia.bounded_compute_extreme_point(sblmo, d, lower_bounds, upper_bounds, int_vars)
	v_fw_MOI = Boscia.compute_extreme_point(lmo, d)
	v_fw_MOI = vec(v_fw_MOI)
	@test norm(v_fw - v_if) ≤ n * eps()
	@test norm(v_if - v_fw_MOI) ≤ n * eps()
	fixed_col = 2
	fixed_row = 3
	# fix one transition and renormalize
	x2 = copy(x)
	x2[:, fixed_col] .= 0
	x2[fixed_row, :] .= 0
	x2[fixed_row, fixed_col] = 1
	x2 = x2 ./ sum(x2, dims=1)
	v_fixed = Boscia.bounded_compute_inface_extreme_point(sblmo, d, x2, lower_bounds, upper_bounds, int_vars)
	idx = (fixed_col-1)*n+fixed_row
	@test v_fixed[idx] == 1
	# If matrix is already a vertex, away-step can give only itself
	@test norm(Boscia.bounded_compute_inface_extreme_point(sblmo, d, v_fixed, lower_bounds, upper_bounds, int_vars) - v_fixed) ≤ eps()
	# fixed a zero only
	x3 = copy(x)
	x3[4, 3] = 0
	# fixing zeros by creating a cycle 4->3->1->4->4
	x3[4, 4] += 1 / n
	x3[1, 4] -= 1 / n
	x3[1, 3] += 1 / n
	v_zero = Boscia.bounded_compute_inface_extreme_point(sblmo, d, x3, lower_bounds, upper_bounds, int_vars)
	idx = (3-1)*n+4
	@test v_zero[idx] == 0
	idx = (4-1)*n+1
	@test v_zero[idx] == 0
end=#

# test with mixed integer constraint
#=@testset "Birkhoff polytope mixed interger" begin
	n = 4
	d = randn(n, n)
	int_vars = collect(n^2/2:n^2)
	int_num = Int(n^2 / 2 + 1)
	lower_bounds = fill(0.0, int_num)
	upper_bounds = fill(1.0, int_num)
	sblmo = BirkhoffBLMO(true, n, int_vars)
	lmo = build_birkhoff_mip(n; lower_bounds=lower_bounds, upper_bounds=upper_bounds, int_vars=int_vars)
	x = ones(n, n) ./ n
	# test without fixings
	v_if = Boscia.bounded_compute_inface_extreme_point(sblmo, d, x, lower_bounds, upper_bounds, int_vars)
	v_fw = Boscia.bounded_compute_extreme_point(sblmo, d, lower_bounds, upper_bounds, int_vars)
	v_fw_MOI = Boscia.compute_extreme_point(lmo, d)
	v_fw_MOI = vec(v_fw_MOI)
	@test norm(v_fw - v_if) ≤ n * eps()
	@test norm(v_if - v_fw_MOI) ≤ n * eps()
	fixed_col = 2
	fixed_row = 3
	# fix one transition and renormalize
	x2 = copy(x)
	x2[:, fixed_col] .= 0
	x2[fixed_row, :] .= 0
	x2[fixed_row, fixed_col] = 1
	x2 = x2 ./ sum(x2, dims=1)
	v_fixed = Boscia.bounded_compute_inface_extreme_point(sblmo, d, x2, lower_bounds, upper_bounds, int_vars)
	idx = (fixed_col-1)*n+fixed_row
	@test v_fixed[idx] == 1
	# If matrix is already a vertex, away-step can give only itself
	@test norm(Boscia.bounded_compute_inface_extreme_point(sblmo, d, v_fixed, lower_bounds, upper_bounds, int_vars) - v_fixed) ≤ eps()
	# fixed a zero only
	x3 = copy(x)
	x3[4, 3] = 0
	# fixing zeros by creating a cycle 4->3->1->4->4
	x3[4, 4] += 1 / n
	x3[1, 4] -= 1 / n
	x3[1, 3] += 1 / n
	v_zero = Boscia.bounded_compute_inface_extreme_point(sblmo, d, x3, lower_bounds, upper_bounds, int_vars)
	idx = (3-1)*n+4
	@test v_zero[idx] == 0
	idx = (4-1)*n+1
	@test v_zero[idx] == 0
	# test with fixed bounds
	lower_bounds[Int(n^2/2)] = 1.0
	sblmo = BirkhoffBLMO(true, n, int_vars)
	lmo = build_birkhoff_mip(n; lower_bounds=lower_bounds, upper_bounds=upper_bounds, int_vars=int_vars)
end=#
"""
