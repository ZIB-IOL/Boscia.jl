using Boscia
using FrankWolfe
using Test
using Random
using SCIP
# using Statistics
using LinearAlgebra
using DataFrames
using CSV
using Distributions
using MathOptInterface: MathOptInterface
const MOI = MathOptInterface

include("cube_blmo.jl")

# For bug hunting:
seed = rand(UInt64)
@show seed


"""

n = 20
diffi = Random.rand(Bool, n) * 0.6 .+ 0.3

@testset "Approximate planted point - Integer" begin

	function f(x)
		return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
	end
	function grad!(storage, x)
		@. storage = x - diffi
	end

	@testset "Using SCIP" begin
		o = SCIP.Optimizer()
		MOI.set(o, MOI.Silent(), true)
		MOI.empty!(o)
		x = MOI.add_variables(o, n)
		for xi in x
			MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
			MOI.add_constraint(o, xi, MOI.LessThan(1.0))
			MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
		end
		lmo = FrankWolfe.MathOptLMO(o)

		x, _, result = Boscia.solve(f, grad!, lmo, verbose = true)

		@test x == round.(diffi)
		@test isapprox(f(x), f(result[:raw_solution]), atol = 1e-6, rtol = 1e-3)
	end

	@testset "Using Cube LMO" begin
		int_vars = collect(1:n)

		bounds = Boscia.IntegerBounds()
		for i in 1:n
			push!(bounds, (i, 0.0), :greaterthan)
			push!(bounds, (i, 1.0), :lessthan)
		end
		blmo = CubeBLMO(n, int_vars, bounds)

		x, _, result = Boscia.solve(f, grad!, blmo, verbose = true)

		@test x == round.(diffi)
		@test isapprox(f(x), f(result[:raw_solution]), atol = 1e-6, rtol = 1e-3)
	end

	@testset "Using Cube Simple LMO" begin
		int_vars = collect(1:n)
		lbs = zeros(n)
		ubs = ones(n)

		sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)

		x, _, result =
			Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, verbose = true)

		@test x == round.(diffi)
		@test isapprox(f(x), f(result[:raw_solution]), atol = 1e-6, rtol = 1e-3)
	end
end


@testset "Approximate planted point - Mixed" begin

	function f(x)
		return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
	end
	function grad!(storage, x)
		@. storage = x - diffi
	end

	int_vars = unique!(rand(collect(1:n), Int(floor(n / 2))))

	@testset "Using SCIP" begin
		o = SCIP.Optimizer()
		MOI.set(o, MOI.Silent(), true)
		MOI.empty!(o)
		x = MOI.add_variables(o, n)
		for xi in x
			MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
			MOI.add_constraint(o, xi, MOI.LessThan(1.0))
			if xi.value in int_vars
				MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
			end
		end
		lmo = FrankWolfe.MathOptLMO(o)

		x, _, result = Boscia.solve(f, grad!, lmo, verbose = true)

		sol = diffi
		sol[int_vars] = round.(sol[int_vars])
		@test sum(isapprox.(x, sol, atol = 1e-6, rtol = 1e-2)) == n
		@test isapprox(f(x), f(result[:raw_solution]), atol = 1e-6, rtol = 1e-3)
	end

	@testset "Using Cube LMO" begin
		bounds = Boscia.IntegerBounds()
		for i in 1:n
			push!(bounds, (i, 0.0), :greaterthan)
			push!(bounds, (i, 1.0), :lessthan)
		end
		blmo = CubeBLMO(n, int_vars, bounds)

		x, _, result = Boscia.solve(f, grad!, blmo, verbose = true)

		sol = diffi
		sol[int_vars] = round.(sol[int_vars])
		@test sum(isapprox.(x, sol, atol = 1e-6, rtol = 1e-2)) == n
		@test isapprox(f(x), f(result[:raw_solution]), atol = 1e-6, rtol = 1e-3)
	end

	@testset "Using Cube Simple LMO" begin
		lbs = zeros(n)
		ubs = ones(n)

		sblmo = Boscia.CubeSimpleBLMO(lbs, ubs, int_vars)

		x, _, result =
			Boscia.solve(f, grad!, sblmo, lbs[int_vars], ubs[int_vars], int_vars, n, verbose = true)

		sol = diffi
		sol[int_vars] = round.(sol[int_vars])
		@test sum(isapprox.(x, sol, atol = 1e-6, rtol = 1e-2)) == n
		@test isapprox(f(x), f(result[:raw_solution]), atol = 1e-6, rtol = 1e-3)
	end
end

"""


function build_cube_blmo(seed, n; problem_type = "integer")
	Random.seed!(seed)
	if problem_type == "integer"
		int_vars = collect(1:n)
	else
		int_vars = unique!(rand(collect(1:n), Int(floor(n / 2))))
	end

	lower_bounds = zeros(n)
	upper_bounds = ones(n)

	sblmo = Boscia.CubeSimpleBLMO(lower_bounds, upper_bounds, int_vars)

	blmo = Boscia.ManagedBoundedLMO(sblmo, lower_bounds[int_vars], upper_bounds[int_vars], int_vars, n)
	return blmo
end

function build_mip_lmo(seed, n; problem_type = "integer")
	Random.seed!(seed)
	if problem_type == "integer"
		int_vars = collect(1:n)
	else
		int_vars = unique!(rand(collect(1:n), Int(floor(n / 2))))
	end

	o = SCIP.Optimizer()
	MOI.set(o, MOI.Silent(), true)
	MOI.empty!(o)
	x = MOI.add_variables(o, n)
	for xi in x
		MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
		MOI.add_constraint(o, xi, MOI.LessThan(1.0))
		if xi.value in int_vars
			MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
		end
	end
	lmo = Boscia.MathOptBLMO(o)
	return lmo
end

function run_args_boscia(mode, f, grad!, lmo, verbose, time_limit; variant = Boscia.BPCG())
	if contains(mode, "lazy")
		if contains(mode, "ws")
			x, _, result = Boscia.solve(f, grad!, lmo, verbose = verbose, time_limit = time_limit, lazy = true, variant = variant, use_DICG_warm_start = true)
		else
			x, _, result = Boscia.solve(f, grad!, lmo, verbose = verbose, time_limit = time_limit, lazy = true, variant = variant, use_DICG_warm_start = false)
		end
	else
		if contains(mode, "ws")
			x, _, result = Boscia.solve(f, grad!, lmo, verbose = verbose, time_limit = time_limit, lazy = false, variant = variant, use_DICG_warm_start = true)
		else
			x, _, result = Boscia.solve(f, grad!, lmo, verbose = verbose, time_limit = time_limit, lazy = false, variant = variant, use_DICG_warm_start = false)
		end
	end

	return x, result
end




function approx_planted_point_boscia(seed, dim; mode = "blmo_lazy", problem_type = "integer", verbose = true, time_limit = 3600, write = true)
	@show seed
	Random.seed!(seed)

	diffi = Random.rand(Bool, dim) * 0.6 .+ 0.3

	function f(x)
		return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
	end

	function grad!(storage, x)
		@. storage = x - diffi
	end

	blmo_modes = ["blmo_lazy", "blmo_dicg", "blmo_dicg_lazy", "blmo_dicg_ws", "blmo_dicg_lazy_ws"]
	mip_modes = ["mip_lazy", "mip_dicg", "mip_dicg_lazy", "mip_dicg_ws", "mip_dicg_lazy_ws"]


	if mode in blmo_modes
		lmo = build_cube_blmo(seed, dim; problem_type = problem_type)
	else
		lmo = build_mip_lmo(seed, dim; problem_type = problem_type)
	end

	if contains(mode, "dicg")
		x, result = run_args_boscia(mode, f, grad!, lmo, verbose, time_limit; variant = Boscia.DICG())
	else
		x, result = run_args_boscia(mode, f, grad!, lmo, verbose, time_limit; variant = Boscia.BPCG())
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

		# df_full = DataFrame(
		# 	seed = seed,
		# 	dimension = dim,
		# 	time = time_list,
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
		# 	@__DIR__,
		# 	"csv/full_run_boscia_" *
		# 	mode *
		# 	"_" *
		# 	string(dim) *
		# 	"_" *
		# 	string(seed) *
		# 	"_birkhoff.csv",
		# )
		# CSV.write(file_name_full, df_full, append = false)

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

		dir_path = joinpath(@__DIR__, "csv", problem_type)

		mkpath(dir_path)

		file_name = joinpath(
			dir_path,
			"boscia_" * mode * "_approx_planted_point_" * string(dim) * "_" * string(seed) * ".csv",
		)
		CSV.write(file_name, df, append = false, writeheader = true)
	end
end

# approx_planted_point_boscia(1, 55; mode = "blmo_dicg_lazy_ws", problem_type = "integer")

