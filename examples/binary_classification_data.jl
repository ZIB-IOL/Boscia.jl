using Distributions, LinearAlgebra
using Plots
pyplot()
using Random

using Boscia
using SCIP
import MathOptInterface
const MOI = MathOptInterface
using FrankWolfe

seed=1
Random.seed!(seed)

samples = 10
data_1 = rand(MvNormal(ones(2), [0.3,0.3]),samples)
x_1 = data_1[1,:]
x_2 = data_1[2,:]
y_1 = zeros(samples)
plot(x_1, x_2, seriestype=:scatter, label="Distribution 1", color="red", markersize = 8)

data_2 = rand(MvNormal([2,2], [0.3,0.3]),samples)
x_1 = data_2[1,:]
x_2 = data_2[2,:]
y_2 = ones(samples)
plot!(x_1, x_2, seriestype=:scatter, label="Distribution 2", color="blue", markersize = 8)
plot!([2.7,0],[0,2.7])
# A = hcat(data_1, data_2)'
# y = vcat(y_1, y_2)

# n0 = 2
# p = 2
# k = 2.0 #ceil(n0/2)

# const mu = 10.0 * rand(Float64);
# const M = 2 * var(A)

# o = SCIP.Optimizer()
# MOI.set(o, MOI.Silent(), true)
# MOI.empty!(o)
# x = MOI.add_variables(o, 2p)
# for i in p+1:2p
#     MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
#     MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
#     MOI.add_constraint(o, x[i], MOI.ZeroOne()) 
# end
# for i in 1:p
#     MOI.add_constraint(
#         o,
#         MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, M], [x[i], x[i+p]]), 0.0),
#         MOI.GreaterThan(0.0),
#     )
#     MOI.add_constraint(
#         o,
#         MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -M], [x[i], x[i+p]]), 0.0),
#         MOI.LessThan(0.0),
#     )
# end
# MOI.add_constraint(
#     o,
#     MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p), x[p+1:2p]), 0.0),
#     MOI.LessThan(k),
# )
# lmo = FrankWolfe.MathOptLMO(o)

# function build_objective_gradient(A, y, mu)
#     # just flexing with unicode
#     # reusing notation from Bach 2010 Self-concordant analyis for LogReg
#     ℓ(u) = log(exp(u/2) + exp(-u/2))
#     dℓ(u) = -1/2 + inv(1 + exp(-u))
#     n = length(y)
#     invn = inv(n)
#     function f(x)
#         xv = @view(x[1:p])
#         err_term = invn * sum(eachindex(y)) do i # 1/N
#             dtemp = dot(A[i,:], xv) # predicted label
#             ℓ(dtemp) - y[i] * dtemp / 2
#         end
#         pen_term = mu * dot(xv, xv) / 2
#         err_term + pen_term
#     end
#     function grad!(storage, x)
#         storage .= 0
#         xv = @view(x[1:p])
#         for i in eachindex(y)
#             dtemp = dot(A[i,:], xv)
#             @. storage += invn * A[i] * (dℓ(dtemp) - y[i] / 2)
#         end
#         @. storage += mu * x
#         storage
#     end
#     (f, grad!)
# end

# f, grad! = build_objective_gradient(A, y, mu)
# x, _, result = Boscia.solve(f, grad!, lmo, verbose=false, fw_epsilon=1e-3, print_iter=10)
# @show x
# @show f(x) 