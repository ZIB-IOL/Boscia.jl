using Statistics
using Random
using Distributions
using LinearAlgebra
import HiGHS
using SCIP
using MathOptInterface
MOI = MathOptInterface
using FrankWolfe
using Boscia
using Bonobo
using DataFrames
using CSV


# Integer sparse regression

# min norm(y-A x)² 
# s.t. 0 <= x_i <= r
# ∑ x_i <= k 
# x_i ∈ Z for i = 1,..,n

# There A represents the collection of data points and 
# is a very tall matrix, i.e. number of rows = m >> number of columns = n.
# y - is the vector of results.
# r - controls how often we have to maximal split on a index.
# k - is the sparsity parameter. We only want a few non zero entries.

function build_int_reg(dim, fac, seed, use_indicator, time_limit, rtol)
    example = "int_reg"
    Random.seed!(seed)

    p = dim
    m = 3*p
    l = 10
    k = ceil(p/fac)

    sol_x = rand(1:l, p)
    for _ in 1:(p-k)
        sol_x[rand(1:p)] = 0
    end

    k = count(i -> i != 0, sol_x)

    D = rand(m, p)
    y_d = D * sol_x
    M = 1.0*l


    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, p)
    z = MOI.add_variables(o, p)
    for i in 1:p
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0 * l))
        MOI.add_constraint(o, x[i], MOI.Integer())

        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne()) 
    end 
    for i in 1:p
        if use_indicator 
            # Indicator: x[i+p] = 1 => x[i] = 0
            # Beware: SCIP can only handle MOI.ACTIVATE_ON_ONE with LessThan constraints.
            # Hence, in the indicator formulation, we ahve zi = 1 => xi = 0. (In bigM zi = 0 => xi = 0)
            gl = MOI.VectorAffineFunction(
                [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                    MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[i])),],
                [0.0, 0.0], )
            gg = MOI.VectorAffineFunction(
                [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                    MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, x[i])),],
                [0.0, 0.0], )
            MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(0.0)))
            MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(0.0)))
        else
            # big M
            MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,M], [x[i], z[i]]), 0.0), MOI.GreaterThan(0.0))
            MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-M], [x[i], z[i]]), 0.0), MOI.LessThan(0.0))
        end
        
    end
    if use_indicator
        MOI.add_constraint(o, sum(z, init=0.0), MOI.GreaterThan(1.0 * (p-k))) # we want less than k zeros
    else
        MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    end
    lmo = FrankWolfe.MathOptLMO(o)


    function f(x)
        xv = @view(x[1:p])
        return 1 / 2 * sum(abs2, y_d - D * xv)  
    end

    function grad!(storage, x)
        storage .= 0
        @view(storage[1:p]) .= transpose(D) * (D * @view(x[1:p]) - y_d)
        return storage
    end

    iter =1
    x = zeros(2p)
    for i in 1:iter
        indicator = use_indicator ? "indicator" : "bigM"
        data = @timed x, time_lmo, result = Boscia.solve(f, grad!, lmo; print_iter = 100, verbose=true, time_limit = time_limit, rel_dual_gap = rtol, dual_gap = 1e-4, use_postsolve = false, fw_epsilon = 1e-2, min_node_fw_epsilon =1e-5)
        df = DataFrame(seed=seed, dimension=dim, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size])
        file_name = "experiments/csv/bigM_vs_indicator_" * example * "_" * indicator * "_" * string(dim) * "_" * string(fac) * "_" * string(seed) * ".csv"
        CSV.write(file_name, df, append=false)
    end
    return x, f(x)
end