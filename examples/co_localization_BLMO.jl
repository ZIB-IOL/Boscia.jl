## FlowPolytope BLMO
using Boscia
using FrankWolfe
using LinearAlgebra
using SparseArrays
using Random

Random.seed!(4)

struct FlowPolytopeBLMO <: Boscia.SimpleBoundableLMO
	m::Int
	n::Int
	atol::Float64
	rtol::Float64
end

FlowPolytopeBLMO(m, n) =
	FlowPolytopeBLMO(m, n, 1e-6, 1e-3)

"""
Computes the extreme point given an direction d, the current lower and upper bounds on the integer variables, and the set of integer variables.
"""
function Boscia.bounded_compute_extreme_point(sblmo::FlowPolytopeBLMO, d, lb, ub, int_vars; kwargs...)
	m = sblmo.m
	n = sblmo.n
	d = reshape(d, m, n)
	v = spzeros(m * n)

	for j in collect(1:n)
		min = Inf
		min_i = -1
		for i in collect(1:m)
			idx = Int((j - 1) * m + i)
			int_vars_idx = findfirst(x -> x == idx, int_vars)
			if int_vars_idx !== nothing
				if lb[int_vars_idx] == 1.0
					min_i = i
					break
				end
				if ub[int_vars_idx] == 0.0
					continue
				end
			end

			if d[i, j] < min
				min = d[i, j]
				min_i = i
			end
		end

		v[Int((j - 1) * m + min_i)] = 1.0

	end
	v = Vector(v)
	return v
end

# sblmo = FlowPolytopeBLMO(3, 4)
# d = rand(3, 4)
# lb = zeros(12)
# ub = ones(12)
# lb[5] = 1.0
# int_vars = collect(1:12)
# v = Boscia.bounded_compute_extreme_point(sblmo, d, lb, ub, int_vars)
# @show d
# @show v



"""
Computes the inface extreme point given an direction d, x, the current lower and upper bounds on the integer variables, and the set of integer variables.
"""
function Boscia.bounded_compute_inface_extreme_point(
	sblmo::FlowPolytopeBLMO,
	d,
	x,
	lb,
	ub,
	int_vars;
	kwargs...,
)
	m = sblmo.m
	n = sblmo.n
	d = reshape(d, m, n)
	x = reshape(x, m, n)
	v = spzeros(m * n)

	for j in collect(1:n)
		min = Inf
		min_i = -1
		for i in collect(1:m)
			if x[i, j] ≈ 0.0
				continue
			end

			if x[i, j] ≈ 1.0
				min_i = i
				break
			end
			idx = Int((j - 1) * m + i)
			int_vars_idx = findfirst(x -> x == idx, int_vars)
			if int_vars_idx !== nothing
				if lb[int_vars_idx] == 1.0
					min_i = i
					break
				end
				if ub[int_vars_idx] == 0.0
					continue
				end
			end

			if d[i, j] < min
				min = d[i, j]
				min_i = i
			end
		end
		v[Int((j - 1) * m + min_i)] = 1.0

	end
	v = Vector(v)
	return v


end

"""
LMO-like operation which computes a vertex minimizing in `direction` on the face defined by the current fixings.
Fixings are maintained by the oracle (or deduced from `x` itself).
"""
function Boscia.bounded_dicg_maximum_step(
	sblmo::FlowPolytopeBLMO,
	direction,
	x,
	lb,
	ub,
	int_vars;
	kwargs...,
)
	T = promote_type(eltype(x), eltype(direction))
	gamma_max = one(T)
	for idx in eachindex(x)
		if direction[idx] != 0.0
			# iterate already on the boundary
			if (direction[idx] < 0 && x[idx] ≈ 1) || (direction[idx] > 0 && x[idx] ≈ 0)
				return zero(gamma_max)
			end
			# clipping with the zero boundary
			if direction[idx] > 0
				gamma_max = min(gamma_max, x[idx] / direction[idx])
			else
				@assert direction[idx] < 0
				gamma_max = min(gamma_max, -(1 - x[idx]) / direction[idx])
			end
		end
	end
	return gamma_max

end

"""
"""
function Boscia.is_decomposition_invariant_oracle_simple(sblmo::FlowPolytopeBLMO)
	return true
end

"""
"""
function Boscia.dicg_split_vertices_set_simple(sblmo::FlowPolytopeBLMO, x, vidx)
	x0_left = copy(x)
	x0_right = copy(x)
	return x0_left, x0_right
end

"""
The sum of each row and column has to be equal to 1.
"""
function Boscia.is_simple_linear_feasible(sblmo::FlowPolytopeBLMO, v::AbstractVector)
	m = sblmo.m
	n = sblmo.n
	v = reshape(v, m, n)
	for j in 1:n


		if !isapprox(sum(v[:, j]), 1.0)
			@debug "Column sum not 1: $(sum(v[((j-1)*m+1):(j*m)]))"
			return false
		end

		# # append by column ? row sum : column sum
		# if !isapprox(sum(v[j:n:n^2]), 1.0, atol = 1e-6, rtol = 1e-3)
		# 	@debug "Row sum not 1: $(sum(v[i:n:n^2]))"
		# 	return false
		# end
	end
	return true
end


function Boscia.build_dicg_start_point(lmo)
	# We pick a random point.
	n = lmo.blmo.n
	d = ones(n)
	x0 = FrankWolfe.compute_extreme_point(lmo, d)
	return x0
end

function domain_orcal(x)
	return true
end
