using Boscia
using FrankWolfe
using Test
using Random
using HiGHS
# using Statistics
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface

# Poisson sparse regression

# For bug hunting:
# seed = rand(UInt64)
# @show seed
# #seed = 0xfe03ee83ca373eab   
# Random.seed!(seed)

# min_{w, b, z} ∑_i exp(w x_i + b) - y_i (w x_i + b) + α norm(w)^2
# s.t. -N z_i <= w_i <= N z_i
# b ∈ [-N, N]
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p

# y_i    - data points, poisson distributed 
# X_i, b - coefficient for the linear estimation of the expected value of y_i
# w_i    - continuous variables
# z_i    - binary variables s.t. z_i = 0 => w_i = 0
# k      - max number of non zero entries in w

# In a poisson regression, we want to model count data.
# It is assumed that y_i is poisson distributed and that the log 
# of its expected value can be computed linearly.

# n = 20
# p = n

# # underlying true weights
# const ws = rand(Float64, p)
# # set 50 entries to 0
# for _ in 1:20
#     ws[rand(1:p)] = 0
# end
# const bs = rand(Float64)
# const Xs = randn(Float64, n, p)
# const ys = map(1:n) do idx
#     a = dot(Xs[idx, :], ws) + bs
#     return rand(Distributions.Poisson(exp(a)))
# end
# Ns = 0.10

# TODO: document better



function build_function(seed, n; Ns=0.0, use_scale=false)
    Random.seed!(seed)
    p = n

    # underlying true weights
    ws = rand(Float64, p)
    # set 50 entries to 0
    for _ in 1:20
        ws[rand(1:p)] = 0
    end
    bs = rand(Float64)
    Xs = randn(Float64, n, p)
    for j in 1:p 
        Xs[:,j] ./= (maximum(Xs[:,j]) - minimum(Xs[:,j]))
        if Ns == 10.0
            Xs[:,j] .*= 0.1
        end
    end
    ys = map(1:n) do idx
        a = dot(Xs[idx, :], ws) + bs
        return rand(Distributions.Poisson(exp(a)))
    end

    α = 1.3
    scale = exp(n/2)
    function f(θ)
        #θ = BigFloat.(θ)
        w = @view(θ[1:p])
        b = θ[end]
        s = sum(1:n) do i
            a = dot(w, Xs[:, i]) + b
            return 1 / n * (exp(a) - ys[i] * a)
        end
        if use_scale
            return 1/scale * (s + α * norm(w)^2)
        end
        return s + α * norm(w)^2
    end
    function grad!(storage, θ)
        #θ = BigFloat.(θ)
        w = @view(θ[1:p])
        b = θ[end]
        storage[1:p] .= 2α .* w
        storage[p+1:2p] .= 0
        storage[end] = 0
        for i in 1:n
            xi = @view(Xs[:, i])
            a = dot(w, xi) + b
            storage[1:p] .+= 1 / n * xi * exp(a)
            storage[1:p] .-= 1 / n * ys[i] * xi
            storage[end] += 1 / n * (exp(a) - ys[i])
        end
        if use_scale
            storage .*= 1/scale
        end
        return storage
    end
    # @show bs, Xs, ys, ws

    return f, grad!, p, α, bs, Xs, ys, ws
end

function build_optimizer(o, p, k, Ns)
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    w = MOI.add_variables(o, p)
    z = MOI.add_variables(o, p)
    b = MOI.add_variable(o)
    # z_i ∈ {0,1} for i = 1,..,p
    for i in 1:p
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())
    end
    for i in 1:p
        # s.t. -N z_i <= w_i <= N z_i
        MOI.add_constraint(o, Ns * z[i] + w[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, -Ns * z[i] + w[i], MOI.LessThan(0.0))
        # Indicator: z[i] = 1 => -N <= w[i] <= N
        #=gl = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),],
            [0.0, 0.0], )
        gg = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),],
            [0.0, 0.0], )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(Ns)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-Ns))) =#
    end
    # ∑ z_i <= k 
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(o, sum(z, init=0.0), MOI.GreaterThan(1.0))
    # b ∈ [-N, N]
    MOI.add_constraint(o, b, MOI.LessThan(Ns))
    MOI.add_constraint(o, b, MOI.GreaterThan(-Ns))
    lmo = Boscia.MathOptBLMO(o)
    return lmo, (w,z,b)
end


############ Decide which strategies to run #####################
strategies = Any[
    "Strong_Branching","MOST_INFEASIBLE"
]

for iterations_stable in Int64[5]
    for decision_function in [
        "product", 
        "weighted_sum"
        ]
        if decision_function == "product"
            μ = 1e-6
            push!(strategies, Dict(:iterations_stable => iterations_stable, :μ => μ, :decision_function => decision_function))
        else
            for μ in [0.7]
                push!(strategies, Dict(:iterations_stable => iterations_stable, :μ => μ, :decision_function => decision_function))
            end
        end
    end
end


############## Example sizes ######################

example_dimensions = [30]

seeds = rand(UInt64, 3)
Ns = 0.1


############## Set Parameters for all runs ######################
verbose = true
print_iter = 100
time_limit = 600
rel_dual_gap=1e-2
# Set parameters for saving results
file_name = "poisson_reg_examples_a_c"

#################################################################
f, grad!, p, α, bs, Xs, ys, ws = build_function(1, 10; Ns=Ns)
    k = 10/2
    o = HiGHS.Optimizer()
    lmo, _ = build_optimizer(o, p, k, Ns)
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)


for seed in seeds
    for dim in example_dimensions
        n = dim

        example_name = string("poisson_reg_n_", n, "_p_", n)
        for branching_strategy in strategies
            f, grad!, p, α, bs, Xs, ys, ws = build_function(seed, n; Ns=Ns)
            k = n/2
            o = HiGHS.Optimizer()
            lmo, _ = build_optimizer(o, p, k, Ns)
            Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)


            #lmo, f, grad! = poisson_reg(n, p, 10, seed, 0.10, 1.3)
            if branching_strategy == "Strong_Branching"

                blmo = Boscia.MathOptBLMO(HiGHS.Optimizer())
                branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
                MOI.set(branching_strategy.bounded_lmo.o, MOI.Silent(), true)
                x, _, result =
                    Boscia.solve(
                        f, 
                        grad!, 
                        lmo,  
                        branching_strategy=branching_strategy,
                        verbose=verbose,
                        print_iter=print_iter, 
                        time_limit=time_limit,
                        rel_dual_gap=rel_dual_gap
                    )
                settings = "Strong_Branching"
                Boscia.save_results(result, settings, example_name, seed, file_name, false) 
            
            elseif branching_strategy == "MOST_INFEASIBLE"
                x, _, result = Boscia.solve(
                    f, 
                    grad!, 
                    lmo, 
                    verbose=verbose, 
                    print_iter=print_iter, 
                    time_limit=time_limit,
                    rel_dual_gap=rel_dual_gap
                    )
                settings = "MOST_INFEASIBLE"
                Boscia.save_results(result, settings, example_name, seed, file_name, false) 
            else
                iterations_stable = branching_strategy[:iterations_stable]
                decision_function = branching_strategy[:decision_function]
                μ = branching_strategy[:μ]
                x, _, result = Boscia.solve(
                    f, 
                    grad!, 
                    lmo,
                    branching_strategy=Boscia.PSEUDO_COST(iterations_stable,false, lmo, μ, decision_function),
                    verbose=verbose, 
                    print_iter=print_iter, 
                    time_limit=time_limit,
                    rel_dual_gap=rel_dual_gap
                    )
                    settings = "PSEUDO_COST_" * decision_function * "_" * string(iterations_stable) * "_μ=" * string(μ)
                Boscia.save_results(result, settings, example_name, seed, file_name, false)
            end
        end
    end
end


