using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
using MathOptInterface: MathOptInterface
const MOI = MathOptInterface
using HiGHS: HiGHS

# Example on the Birkhoff polytope but using permutation matrices directly
# https://arxiv.org/pdf/2011.02752.pdf
# https://www.sciencedirect.com/science/article/pii/S0024379516001257

# For bug hunting:
seed = rand(UInt64)
@show seed
Random.seed!(seed)


# min_{X, θ} 1/2 * || ∑_{i in [k]} θ_i X_i - Xhat ||^2
# θ ∈ Δ_k (simplex)
# X_i ∈ P_n (permutation matrix)

# we linearize the bilinear terms in the objective
# min_{X, Y, θ} 1/2 ||∑_{i in [k]} Y_i - Xhat ||^2
# θ ∈ Δ_k (simplex)
# X_i ∈ P_n (permutation matrix)
# 0 ≤ Y_i ≤ X_i
# 0 ≤ θ_i - Y_i ≤ 1 - X_i

# The variables are ordered (Y, X, theta) in the MOI model
# the objective only uses the last n^2 variables
# Small dimensions since the size of the problem grows quickly (2 k n^2 + k variables)

function Boscia.build_dicg_start_point(lmo)
	# We pick a random point.
	n, _ = Boscia.get_list_of_variables(lmo.blmo)
	d = ones(n)
	x0 = FrankWolfe.compute_extreme_point(lmo, d)
	return x0
end


function build_objective(n, k)
	# generate random doubly stochastic matrix
	Xstar = rand(n, n)
	while norm(sum(Xstar, dims = 1) .- 1) > 1e-6 || norm(sum(Xstar, dims = 2) .- 1) > 1e-6
		Xstar ./= sum(Xstar, dims = 1)
		Xstar ./= sum(Xstar, dims = 2)
	end

	function f(x)
		s = zero(eltype(x))
		for i in eachindex(Xstar)
			s += 0.5 * (sum(x[(j-1)*n^2+i] for j in 1:k) - Xstar[i])^2
		end
		return s
	end

	# note: reshape gives a reference to the same data, so this is updating storage in-place
	function grad!(storage, x)
		storage .= 0
		for j in 1:k
			Sk = reshape(@view(storage[(j-1)*n^2+1:j*n^2]), n, n)
			@. Sk = -Xstar
			for m in 1:k
				Yk = reshape(@view(x[(m-1)*n^2+1:m*n^2]), n, n)
				@. Sk += Yk
			end
		end
		return storage
	end

	return f, grad!
end


function build_birkhoff_lmo(n, k)
	o = SCIP.Optimizer()
	MOI.set(o, MOI.Silent(), true)
	MOI.empty!(o)
	Y = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
	X = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
	theta = MOI.add_variables(o, k)

	for i in 1:k
		MOI.add_constraint.(o, Y[i], MOI.GreaterThan(0.0))
		MOI.add_constraint.(o, Y[i], MOI.LessThan(1.0))
		MOI.add_constraint.(o, X[i], MOI.ZeroOne())
		MOI.add_constraint(o, theta[i], MOI.GreaterThan(0.0))
		MOI.add_constraint(o, theta[i], MOI.LessThan(1.0))
		# doubly stochastic constraints
		MOI.add_constraint.(
			o,
			vec(sum(X[i], dims = 1, init = MOI.ScalarAffineFunction{Float64}([], 0.0))),
			MOI.EqualTo(1.0),
		)
		MOI.add_constraint.(
			o,
			vec(sum(X[i], dims = 2, init = MOI.ScalarAffineFunction{Float64}([], 0.0))),
			MOI.EqualTo(1.0),
		)
		# 0 ≤ Y_i ≤ X_i
		MOI.add_constraint.(o, 1.0 * Y[i] - X[i], MOI.LessThan(0.0))
		# 0 ≤ θ_i - Y_i ≤ 1 - X_i
		MOI.add_constraint.(o, 1.0 * theta[i] .- Y[i] .+ X[i], MOI.LessThan(1.0))
	end
	MOI.add_constraint(o, sum(theta, init = 0.0), MOI.EqualTo(1.0))
	return FrankWolfe.MathOptLMO(o)
end



function birkhoff_decomposition_boscia(
	seed,
	dim,
	k;
	mode = "bpcg",
	line_search = FrankWolfe.Adaptive(),
	fw_epsilon = 1e-2,
	use_DICG_warm_start = false,
	verbose = true,
	time_limit = 3600,
	write = true,
	variant = Boscia.BPCG(),
	rel_dual_gap = 1e-2,
	lazy = true,
)

	@show seed
	Random.seed!(seed)

	lmo = build_birkhoff_lmo(dim, k)

	f, grad! = build_objective(dim, k)

	x, _, result =
		Boscia.solve(
			f,
			grad!,
			lmo,
			variant = variant,
			verbose = verbose,
			lazy = lazy,
			line_search = line_search,
			fw_epsilon = fw_epsilon,
			time_limit = time_limit,
			use_DICG_warm_start = use_DICG_warm_start,
			rel_dual_gap = rel_dual_gap,
			fw_verbose = true,
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
			"boscia_" * mode * "_birkhoff_decomposition_" * string(dim) * "_" * string(seed) * ".csv",
		)
		CSV.write(file_name, df, append = false, writeheader = true)
	end


end

birkhoff_decomposition_boscia(4, 3, 2; mode = "bpcg", variant = Boscia.DICG(), lazy = false, write = true, verbose = true)

