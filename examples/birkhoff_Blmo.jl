## Birkhoff BLMO
using Boscia
using Hungarian
using LinearAlgebra
using SparseArrays

"""
	BikrhoffBLMO

A simple LMO that computes the extreme point given the node specific bounds on the integer variables.
Can be stateless since all of the bound management is done by the ManagedBoundedLMO.   
"""
struct BirkhoffBLMO <: Boscia.SimpleBoundableLMO
	append_by_column::Bool
	dim::Int
	int_vars::Vector{Int}
	atol::Float64
	rtol::Float64
end

BirkhoffBLMO(append_by_column, dim, int_vars) =
	BirkhoffBLMO(append_by_column, dim, int_vars, 1e-6, 1e-3)

"""
Computes the extreme point given an direction d, the current lower and upper bounds on the integer variables, and the set of integer variables.
"""
function Boscia.bounded_compute_extreme_point(sblmo::BirkhoffBLMO, d, lb, ub, int_vars; kwargs...)
	n = sblmo.dim

	if size(d, 2) == 1
		d = sblmo.append_by_column ? reshape(d, (n, n)) : transpose(reshape(d, (n, n)))
	end


	fixed_to_one_rows = Int[]
	fixed_to_one_cols = Int[]
	delete_ub = Int[]

	for idx in eachindex(int_vars)
		if lb[idx] >= 1 - eps()
			var_idx = int_vars[idx]
			if sblmo.append_by_column
				j = ceil(Int, var_idx / n)
				i = Int(var_idx - n * (j - 1))
				push!(fixed_to_one_rows, i)
				push!(fixed_to_one_cols, j)
				append!(delete_ub, union(collect((j-1)*n+1:j*n), collect(i:n:n^2)))
			else
				i = ceil(Int, var_idx / n)
				j = Int(var_idx - n * (j - 1))
				push!(fixed_to_one_rows, i)
				push!(fixed_to_one_cols, j)
				append!(delete_ub, union(collect((i-1)*n+1:i*n), collect(j:n:n^2)))
			end
		end
	end

	#=for j in 1:n
		for i in 1:n
			if lb[(j-1)*n + i] >= 1 - eps()
				if sblmo.append_by_column
					push!(fixed_to_one_rows, i)
					push!(fixed_to_one_cols, j)
					append!(delete_ub, union(collect((j-1)*n+1:j*n), collect(i:n:n^2)))
				else
					push!(fixed_to_one_rows, j)
					push!(fixed_to_one_cols, i)
					append!(delete_ub, union(collect((i-1)*n+1:i*n), collect(j:n:n^2)))
				end
			end
		end
	end=#

	sort!(delete_ub)
	unique!(delete_ub)
	nfixed = length(fixed_to_one_cols)
	nreduced = n - nfixed
	reducedub = copy(ub)
	reducedintvars = copy(int_vars)
	delete_ub_idx = findall(x -> x in delete_ub, int_vars)
	deleteat!(reducedub, delete_ub_idx)
	deleteat!(reducedintvars, delete_ub_idx)


	# stores the indices of the original matrix that are still in the reduced matrix
	index_map_rows = fill(1, nreduced)
	index_map_cols = fill(1, nreduced)
	idx_in_map_row = 1
	idx_in_map_col = 1
	for orig_idx in 1:n
		if orig_idx ∉ fixed_to_one_rows
			index_map_rows[idx_in_map_row] = orig_idx
			idx_in_map_row += 1
		end
		if orig_idx ∉ fixed_to_one_cols
			index_map_cols[idx_in_map_col] = orig_idx
			idx_in_map_col += 1
		end
	end
	type = typeof(d[1, 1])
	d2 = ones(Union{type, Missing}, nreduced, nreduced)

	for j in 1:nreduced
		for i in 1:nreduced
			idx = (index_map_cols[j] - 1) * n + index_map_rows[i]
			# interdict arc when fixed to zero
			if idx in reducedintvars
				reducedub_idx = findfirst(x -> x == idx, reducedintvars)
				if reducedub[reducedub_idx] <= eps()
					if sblmo.append_by_column
						d2[i, j] = missing
					else
						d2[j, i] = missing
					end
				else
					if sblmo.append_by_column
						d2[i, j] = d[index_map_rows[i], index_map_cols[j]]
					else
						d2[j, i] = d[index_map_rows[j], index_map_cols[i]]
					end
				end
			else
				if sblmo.append_by_column
					d2[i, j] = d[index_map_rows[i], index_map_cols[j]]
				else
					d2[j, i] = d[index_map_rows[j], index_map_cols[i]]
				end
			end
		end
	end

	#=for j in 1:nreduced
		for i in 1:nreduced
			# interdict arc when fixed to zero
			if reducedub[(j-1)*nreduced + i] <= eps()
				if sblmo.append_by_column
					d2[i,j] = missing
				else
					d2[j,i] = missing
				end
			else
				if sblmo.append_by_column
					d2[i,j] = d[index_map_rows[i], index_map_cols[j]]
				else
					d2[j,i] = d[index_map_rows[j], index_map_cols[i]]
				end
			end
		end
	end=#
	m = SparseArrays.spzeros(n, n)
	for (i, j) in zip(fixed_to_one_rows, fixed_to_one_cols)
		m[i, j] = 1
	end
	res_mat = Hungarian.munkres(d2)
	(rows, cols, vals) = SparseArrays.findnz(res_mat)
	@inbounds for i in eachindex(cols)
		m[index_map_rows[rows[i]], index_map_cols[cols[i]]] = (vals[i] == 2)
	end

	m = if sblmo.append_by_column
		reduce(vcat, Matrix(m))
	else
		reduce(vcat, LinearAlgebra.transpose(Matrix(m)))
	end

	return m
end


"""
Computes the inface extreme point given an direction d, x, the current lower and upper bounds on the integer variables, and the set of integer variables.
"""
function Boscia.bounded_compute_inface_extreme_point(
	sblmo::BirkhoffBLMO,
	direction,
	x,
	lb,
	ub,
	int_vars;
	kwargs...,
)
	n = sblmo.dim

	if size(direction, 2) == 1
		direction =
			sblmo.append_by_column ? reshape(direction, (n, n)) :
			transpose(reshape(direction, (n, n)))
	end

	if size(x, 2) == 1
		x = sblmo.append_by_column ? reshape(x, (n, n)) : transpose(reshape(x, (n, n)))
	end
	fixed_to_one_rows = Int[]
	fixed_to_one_cols = Int[]
	delete_ub = Int[]

	for idx in eachindex(int_vars)
		if lb[idx] >= 1 - eps()
			var_idx = int_vars[idx]
			if sblmo.append_by_column
				j = ceil(Int, var_idx / n)
				i = Int(var_idx - n * (j - 1))
				push!(fixed_to_one_rows, i)
				push!(fixed_to_one_cols, j)
				append!(delete_ub, union(collect((j-1)*n+1:j*n), collect(i:n:n^2)))
			else
				i = ceil(int, var_idx / n)
				j = Int(var_idx - n * (j - 1))
				push!(fixed_to_one_rows, j)
				push!(fixed_to_one_cols, i)
				append!(delete_ub, union(collect((i-1)*n+1:i*n), collect(j:n:n^2)))
			end
		end
	end

	for j in 1:n
		if j ∉ fixed_to_one_cols
			for i in 1:n
				if i ∉ fixed_to_one_rows
					if x[i, j] >= 1 - eps()
						push!(fixed_to_one_rows, i)
						push!(fixed_to_one_cols, j)
						if sblmo.append_by_column
							append!(delete_ub, union(collect((j-1)*n+1:j*n), collect(i:n:n^2)))
						else
							append!(delete_ub, union(collect((i-1)*n+1:i*n), collect(j:n:n^2)))
						end
					end
				end
			end
		end
	end

	#=for j in 1:n
		for i in 1:n
			if lb[(j-1)*n + i] >= 1 - eps()
				if sblmo.append_by_column
					push!(fixed_to_one_rows, i)
					push!(fixed_to_one_cols, j)
					append!(delete_ub, union(collect((j-1)*n+1:j*n), collect(i:n:n^2)))
				else
					push!(fixed_to_one_rows, j)
					push!(fixed_to_one_cols, i)
					append!(delete_ub, union(collect((i-1)*n+1:i*n), collect(j:n:n^2)))
				end
			end
		end
	end=#
	sort!(delete_ub)
	unique!(delete_ub)
	fixed_to_one_cols = unique!(fixed_to_one_cols)
	fixed_to_one_rows = unique!(fixed_to_one_rows)
	nfixed = length(fixed_to_one_cols)
	nreduced = n - nfixed
	reducedub = copy(ub)
	reducedintvars = copy(int_vars)
	delete_ub_idx = findall(x -> x in delete_ub, int_vars)
	deleteat!(reducedub, delete_ub_idx)
	deleteat!(reducedintvars, delete_ub_idx)
	# stores the indices of the original matrix that are still in the reduced matrix
	index_map_rows = fill(1, nreduced)
	index_map_cols = fill(1, nreduced)
	idx_in_map_row = 1
	idx_in_map_col = 1
	for orig_idx in 1:n
		if orig_idx ∉ fixed_to_one_rows
			index_map_rows[idx_in_map_row] = orig_idx
			idx_in_map_row += 1
		end
		if orig_idx ∉ fixed_to_one_cols
			index_map_cols[idx_in_map_col] = orig_idx
			idx_in_map_col += 1
		end
	end
	type = typeof(direction[1, 1])
	d2 = ones(Union{type, Missing}, nreduced, nreduced)

	for j in 1:nreduced
		for i in 1:nreduced
			idx = (index_map_cols[j] - 1) * n + index_map_rows[i]
			# interdict arc when fixed to zero
			if idx in reducedintvars
				reducedub_idx = findfirst(x -> x == idx, reducedintvars)
				if reducedub[reducedub_idx] <= eps()
					if sblmo.append_by_column
						d2[i, j] = missing
					else
						d2[j, i] = missing
					end
				else
					if sblmo.append_by_column
						if x[index_map_rows[i], index_map_cols[j]] <= eps()
							d2[i, j] = missing
						else
							d2[i, j] = direction[index_map_rows[i], index_map_cols[j]]
						end
					else
						if x[index_map_rows[i], index_map_cols[j]] <= eps()
							d2[j, i] = missing
						else
							d2[j, i] = direction[index_map_rows[j], index_map_cols[i]]
						end
					end
				end
			else
				if sblmo.append_by_column
					if x[index_map_rows[i], index_map_cols[j]] <= eps()
						d2[i, j] = missing
					else
						d2[i, j] = direction[index_map_rows[i], index_map_cols[j]]
					end
				else
					if x[index_map_rows[i], index_map_cols[j]] <= eps()
						d2[j, i] = missing
					else
						d2[j, i] = direction[index_map_rows[j], index_map_cols[i]]
					end
				end
			end
		end
	end

	#=for j in 1:nreduced
		for i in 1:nreduced
			# interdict arc when fixed to zero
			if reducedub[(j-1)*nreduced + i] <= eps()
				if sblmo.append_by_column
					d2[i,j] = missing
				else
					d2[j,i] = missing
				end
			else
				if sblmo.append_by_column
					d2[i,j] = d[index_map_rows[i], index_map_cols[j]]
				else
					d2[j,i] = d[index_map_rows[j], index_map_cols[i]]
				end
			end
		end
	end=#
	m = SparseArrays.spzeros(n, n)
	for (i, j) in zip(fixed_to_one_rows, fixed_to_one_cols)
		m[i, j] = 1
	end
	res_mat = Hungarian.munkres(d2)
	(rows, cols, vals) = SparseArrays.findnz(res_mat)
	@inbounds for i in eachindex(cols)
		m[index_map_rows[rows[i]], index_map_cols[cols[i]]] = (vals[i] == 2)
	end


	m = if sblmo.append_by_column
		reduce(vcat, Matrix(m))
	else
		reduce(vcat, LinearAlgebra.transpose(Matrix(m)))
	end

	return m
end

"""
LMO-like operation which computes a vertex minimizing in `direction` on the face defined by the current fixings.
Fixings are maintained by the oracle (or deduced from `x` itself).
"""
function Boscia.bounded_dicg_maximum_step(
	sblmo::BirkhoffBLMO,
	direction,
	x,
	lb,
	ub,
	int_vars;
	kwargs...,
)
	n = sblmo.dim

	if size(direction, 2) == 1
		direction =
			sblmo.append_by_column ? reshape(direction, (n, n)) :
			transpose(reshape(direction, (n, n)))
		x = sblmo.append_by_column ? reshape(x, (n, n)) : transpose(reshape(x, (n, n)))
	end
	return FrankWolfe.dicg_maximum_step(FrankWolfe.BirkhoffPolytopeLMO(), direction, x)
end

"""
"""
function Boscia.is_decomposition_invariant_oracle_simple(sblmo::BirkhoffBLMO)
	return true
end

"""
"""
function Boscia.dicg_split_vertices_set_simple(sblmo::BirkhoffBLMO, x, vidx)
	x0_left = copy(x)
	x0_right = copy(x)
	return x0_left, x0_right
end

"""
The sum of each row and column has to be equal to 1.
"""
function Boscia.is_simple_linear_feasible(sblmo::BirkhoffBLMO, v::AbstractVector)
	n = sblmo.dim
	for i in 1:n
		# append by column ? column sum : row sum 
		if !isapprox(sum(v[((i-1)*n+1):(i*n)]), 1.0, atol = 1e-6, rtol = 1e-3)
			@debug "Column sum not 1: $(sum(v[((i-1)*n+1):(i*n)]))"
			return false
		end
		# append by column ? row sum : column sum
		if !isapprox(sum(v[i:n:n^2]), 1.0, atol = 1e-6, rtol = 1e-3)
			@debug "Row sum not 1: $(sum(v[i:n:n^2]))"
			return false
		end
	end
	return true
end

#=function Boscia.is_simple_linear_feasible(sblmo::BirkhoffBLMO, v::AbstractMatrix) 
	n = sblmo.dim
	for i in 1:n
		# check row sum
		if !isapprox(sum(v[i, 1:n]), 1.0, atol=sblmo.atol, rtol=sblmo.rtol)
			return false
		end
		# check column sum
		if !isapprox(sum(v[1:n, i]), 1.0, atol=sblmo.atol, rtol=sblmo.rtol)
			return false
		end
	end
	return true
end=#

