using Random
using LinearAlgebra
using Kronecker


function hessian(A, D)
	# Compute the Hessian as A ⊗ D + (A ⊗ D)'
	H = kron(A, D) + kron(A, D)'
	return H
end

# Function to generate distance and flow matrices for facility layout
function generate_facility_layout_data(n::Int)
	# Seed the random number generator for reproducibility (optional)

	# Generate a random distance matrix (symmetric, with zeros on the diagonal)
	distance_matrix = zeros(n, n)
	for i in 1:n
		for j in i+1:n
			distance = rand()  # Random distance between 1 and 10
			distance_matrix[i, j] = distance
			distance_matrix[j, i] = distance
		end
		distance_matrix[i, i] = 0  # Ensure the diagonal is always zero
	end

	# Initialize the flow matrix
	flow_matrix = zeros(n, n)

	# Ensure each facility is connected to at least two other facilities, but randomly more
	for h in 1:n
		# Pick a random number of connected facilities (at least 2, up to n-1), excluding h
		num_connections = rand(2:n-1)
		# Create a list of facilities excluding h
		available_facilities = setdiff(1:n, [h])

		# Randomly select distinct facilities from the available list
		connected_facilities = available_facilities[randperm(n - 1)[1:num_connections]]

		for i in connected_facilities
			# Pick another facility to form a cycle with, excluding h and i
			remaining_facilities = setdiff(connected_facilities, [h, i])
			j = rand(remaining_facilities)

			# Assign positive random flows (non-symmetric)
			flow_h_i = rand()
			flow_i_j = rand()
			flow_j_h = rand()

			# Assign the flows if the values are zero (to avoid overwriting)
			if flow_matrix[h, i] == 0
				flow_matrix[h, i] = flow_h_i
				flow_matrix[i, h] = flow_h_i
			end
			if flow_matrix[i, j] == 0
				flow_matrix[i, j] = flow_i_j
				flow_matrix[j, i] = flow_i_j
			end
			if flow_matrix[j, h] == 0
				flow_matrix[j, h] = flow_j_h
				flow_matrix[h, j] = flow_j_h
			end
		end

		flow_matrix[h, h] = 0  # Ensure the diagonal is always zero
	end

	cost_matrix = rand(n, n)


	return flow_matrix, distance_matrix, cost_matrix
end

# Function to compute the n^2 x n^2 matrix P and the Q vector of length n^2
function generate_p_and_q_matrices(A, D)
	n = size(A, 1)  # Assuming A and D are square matrices of the same size

	# Step 1: Calculate row and column sums for A and D
	a = sum(A, dims = 2)[:]        # Row-wise sum of A
	a_prime = sum(A, dims = 1)'[:]  # Column-wise sum of A (transpose of row sum)

	d = sum(D, dims = 2)[:]        # Row-wise sum of D
	d_prime = sum(D, dims = 1)'[:]  # Column-wise sum of D (transpose of row sum)

	# Initialize the P matrix of size n^2 x n^2 and the Q vector of size n^2
	P = zeros(Float64, n^2, n^2)
	Q = zeros(Float64, n^2)

	# Step 2: Fill the matrix P and the vector Q based on the provided formulas
	for i in 1:n
		for k in 1:n
			# Compute the index for Q and P using the (i,k) pair
			idx = (i - 1) * n + k

			# Compute the Q vector entry q_{ik}
			Q[idx] = 0.5 * (a[i] * d[k] + a_prime[i] * d_prime[k])

			# Now, populate the P matrix for each (i,k) and (j,l)
			for j in 1:n
				for l in 1:n
					row_idx = (i - 1) * n + k
					col_idx = (j - 1) * n + l

					# Diagonal elements: P_{ik, ik}
					if (i == j) && (k == l)
						P[row_idx, col_idx] = 0.5 * (a[i] * d[k] + a_prime[i] * d_prime[k])
					else
						# Off-diagonal elements: P_{ik, jl} = A_{ij} * D_{kl} for i ≠ j and k ≠ l
						P[row_idx, col_idx] = A[i, j] * D[k, l]
					end
				end
			end
		end
	end

	@assert isposdef(P)


	return P, Q
end

function build_facility_objective(A, b)

	function f(x)
		return x' * A * x - b' * x
	end

	function grad!(storage, x)
		storage .= 2.0 * A * x - b
		return storage
	end
	return f, grad!
end

function build_Koopmans_Beckmann_objective(A, D, C)
	D_kronecker_A = kron(D, A)
	A_kronecker_D = kron(A, D)
	lambda_min = minimum(eigvals(D_kronecker_A))
	n = size(A, 1)
	I = Diagonal(ones(n^2))

	function f(x)
		X = reshape(x, n, n)
		return x' * (D_kronecker_A - lambda_min * I) * x + LinearAlgebra.tr(C * X')
	end

	function grad!(storage, x)
		storage .= 2.0 * (D_kronecker_A - lambda_min * I) * x + vec(C)
		return storage
	end
	return f, grad!
end

function build_start_point(A, D)
	n = size(A, 1)

	# Calculate row-wise sums of A and D
	a = sum(A, dims = 2)[:]  # Row sums of A, flattened into a vector
	d = sum(D, dims = 2)[:]  # Row sums of D, flattened into a vector

	# Sort `a` in non-increasing order and `d` in non-decreasing order
	a_indices = sortperm(a, rev = true)      # Get indices of sorted `a`
	d_indices = sortperm(d, rev = false)     # Get indices of sorted `d`

	# Initialize the start point matrix X (n x n) with zeros
	X = zeros(n, n)

	# Set X based on the indices from the sorted `a` and `d`
	for u in 1:n
		i = a_indices[u]   # The index from sorted `a`
		k = d_indices[u]   # The index from sorted `d`
		X[i, k] = 1.0        # Set X[i, k] to 1.0
	end
	# Return the flattened vector form of the matrix X (n^2 vector)
	return vec(X)
end




