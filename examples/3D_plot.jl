using PyPlot
using Random
using LinearAlgebra
using Statistics


seed = 1
Random.seed!(seed)

# function f(x)
#     return 1 / 2 * Ωi * dot(x, Mi, x) - dot(ri, x)
# end

# n = 2
# ri = rand(n)
# ai = rand(n)
# Ωi = rand()
# bi = sum(ai)
# Ai = randn(n, n)
# Ai = Ai' * Ai
# Mi = (Ai + Ai') / 2
# @assert isposdef(Mi)

mu = 10.0 * rand(Float64);

n0 = 2;
p = n0 # 5 * n0;
k = ceil(n0 / 5);
lambda_0 = rand(Float64);
lambda_2 = 10.0 * rand(Float64);
A = rand(Float64, n0, p)
y = rand(Float64, n0)
M = 2 * var(A)

function build_objective_gradient(A, y, mu)
    # just flexing with unicode
    # reusing notation from Bach 2010 Self-concordant analyis for LogReg
    ℓ(u) = log(exp(u/2) + exp(-u/2))
    dℓ(u) = -1/2 + inv(1 + exp(-u))
    n = length(y)
    invn = inv(n)
    p = size(A)[2]
    function f(x)
        xv = @view(x[1:p])
        err_term = invn * sum(eachindex(y)) do i # 1/N
            dtemp = dot(A[i,:], xv) # predicted label
            ℓ(dtemp) - y[i] * dtemp / 2
        end
        pen_term = mu * dot(xv, xv) / 2
        err_term + pen_term
    end
    function grad!(storage, x)
        storage .= 0
        xv = @view(x[1:p])
        for i in eachindex(y)
            dtemp = dot(A[i,:], xv)
            @. storage += invn * A[i] * (dℓ(dtemp) - y[i] / 2)
        end
        @. storage +=  mu * x
        storage
    end
    (f, grad!)
end

f, grad! = build_objective_gradient(A, y, mu)

dim_grid = 40
x = LinRange(0, 1, dim_grid)
meshgrid = x' .* ones(dim_grid)

zMesh = zeros(dim_grid, dim_grid)
for (idx_i, i) in enumerate(x)
    for (idx_j, j) in enumerate(x)
        zMesh[idx_i,idx_j] = f([i,j])
    end
end

fig = plt.figure(figsize=(12,6))
# ax1 = fig.add_subplot(121, projection="3d")
# surf = ax1.plot_surface(x, x, zMesh)
# fig.colorbar(surf)
# ax1.set_xlabel("x")
# ax1.set_ylabel("y")
# ax1.set_zlabel("f")

ax2 = fig.add_subplot(111)
surf2 = plt.contourf(x, x, zMesh)
fig.colorbar(surf2)
ax2.set_xlabel("x")
ax2.set_ylabel("y")

savefig("examples/csv/3D.png")
