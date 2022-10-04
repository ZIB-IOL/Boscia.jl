using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
using Distributions
import MathOptInterface
MOI = MathOptInterface

function build_lmo(dimension, seed)
    #seed = 3 # 4 # 3 freezes ? after rens # 1 too slow
    Random.seed!(seed)

    n = dimension
    ri = rand(n)
    ai = rand(n)
    Ωi = rand(Float64)
    bi = sum(ai)
    Ai = randn(n, n)
    Ai = Ai' * Ai
    Mi = (Ai + Ai') / 2
    @assert isposdef(Mi)

    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    I = collect(1:n) #rand(1:n0, Int64(floor(n0/2)))
    for i in 1:n
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        if i in I
            MOI.add_constraint(o, x[i], MOI.Integer())
        end
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai, x), 0.0),
        MOI.LessThan(bi),
    )
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.GreaterThan(1.0),
    )
    lmo = FrankWolfe.MathOptLMO(o)
    return lmo
end

function f(x)
    return 1 / 2 * Ωi * dot(x, Mi, x) - dot(ri, x)
end
function grad!(storage, x)
    mul!(storage, Mi, x, Ωi, 0)
    storage .-= ri
    return storage
end

# x, _, result = Boscia.solve(f, grad!, lmo, verbose=true)
# @test dot(ai, x) <= bi + 1e-6
# @test f(x) <= f(result[:raw_solution]) + 1e-6
# @show MOI.get(o, MOI.SolveTimeSec())

seeds = [1,2]
dimensions = [10,15]

iter = 1#3

for (seed_idx, seed_val) in enumerate(seeds)
    for (dim_idx,dim_val) in enumerate(dimensions)
        for i in 1:iter
            seed = seed_val
            dimension = dim_val
            lmo = build_lmo(dimension, seed)
            _, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true)
            df = DataFrame(seed=seed, dimension=n, time_boscia=result[:total_time_in_sec], memory_boscia=data[3])
            file_name = "experiments/csv/boscia_vs_scip.csv"
            CSV.write(file_name, df, append=false)
        end
    end
end 

mutable struct GradientCutHandler{F, G, XT} <: SCIP.AbstractConstraintHandler
    o::SCIP.Optimizer
    f::F
    grad!::G
    storage::XT
    epivar::MOI.VariableIndex
    vars::Vector{MOI.VariableIndex}
    ncalls::Int
end

function SCIP.check(ch::GradientCutHandler, constraints::Vector{Ptr{SCIP.SCIP_CONS}}, sol::Ptr{SCIP.SCIP_SOL}, checkintegrality::Bool, checklprows::Bool, printreason::Bool, completely::Bool)
    @assert length(constraints) == 0
    values = SCIP.sol_values(ch.o, ch.vars, sol)
    zval = SCIP.sol_values(ch.o, [ch.epivar], sol)[1]
    if zval < ch.f(values)
        return SCIP.SCIP_INFEASIBLE
    end
    return SCIP.SCIP_FEASIBLE
end

function enforce_epigraph(ch::GradientCutHandler)
    values = SCIP.sol_values(ch.o, ch.vars)
    zval = SCIP.sol_values(ch.o, [ch.epivar])[1]
    fx = ch.f(values)
    ch.grad!(ch.storage, values)
    # f(x̂) + dot(∇f(x̂), x-x̂) - z ≤ 0 <=>
    # dot(∇f(x̂), x) - z ≤ dot(∇f(x̂), x̂) - f(x̂)
    if zval < fx - 1e-10
        # println(fx - zval)
        f = dot(ch.storage, ch.vars) - ch.epivar
        s = MOI.LessThan(dot(ch.storage, values) - fx)
        fval = MOI.Utilities.eval_variables(vi -> SCIP.sol_values(ch.o, [vi])[1],  f)
        # @show fval - s.upper
        # @assert fval > s.upper - 1e-10
        MOI.add_constraint(
            ch.o,
            dot(ch.storage, ch.vars) - ch.epivar,
            MOI.LessThan(dot(ch.storage, values) - fx),
        )
        ch.ncalls += 1
        # @show ch.ncalls
        return SCIP.SCIP_CONSADDED
    end
    return SCIP.SCIP_FEASIBLE
end

function SCIP.enforce_lp_sol(ch::GradientCutHandler, constraints, nusefulconss, solinfeasible)
    @assert length(constraints) == 0
    return enforce_epigraph(ch)
end

function SCIP.enforce_pseudo_sol(
        ch::GradientCutHandler, constraints, nusefulconss,
        solinfeasible, objinfeasible,
    )
    @assert length(constraints) == 0
    return enforce_epigraph(ch)
end

function SCIP.lock(ch::GradientCutHandler, constraint, locktype, nlockspos, nlocksneg)
    z::Ptr{SCIP.SCIP_VAR} = SCIP.var(ch.o, ch.epivar)
    if z != C_NULL
        SCIP.@SCIP_CALL SCIP.SCIPaddVarLocksType(ch.o, z, SCIP.SCIP_LOCKTYPE_MODEL, nlockspos, nlocksneg)
    end
    for x in ch.vars
        xi::Ptr{SCIP.SCIP_VAR} = SCIP.var(ch.o, x)
        xi == C_NULL && continue
        SCIP.@SCIP_CALL SCIP.SCIPaddVarLocksType(ch.o, xi, SCIP.SCIP_LOCKTYPE_MODEL, nlockspos + nlocksneg, nlockspos + nlocksneg)
    end
end

o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
MOI.empty!(o)
x = MOI.add_variables(o, n)
I = collect(1:n) #rand(1:n0, Int64(floor(n0/2)))
for i in 1:n
    MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
    if i in I
        MOI.add_constraint(o, x[i], MOI.Integer())
    end
end
MOI.add_constraint(
    o,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai, x), 0.0),
    MOI.LessThan(bi),
)
MOI.add_constraint(
    o,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
    MOI.GreaterThan(1.0),
)

z = MOI.add_variable(o)
MOI.add_constraint(o, z, MOI.GreaterThan(0.0))

epigraph_ch = GradientCutHandler(o, f, grad!, zeros(length(x)), z, x, 0)
SCIP.include_conshdlr(o, epigraph_ch; needs_constraints=false, name="handler_gradient_cuts")

MOI.set(o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z)

values = [0.9, 1.0]
fw_epsilon_values = [1e-3, 5e-3, 1e-4, 1e-7]
min_num_lower_values = [20, 40, 60, 80, 100, 200, Inf]
seeds = [1]#,2,3]

iter = 1#3

for (seed_idx, seed_val) in enumerate(seeds)
    for (index,value) in enumerate(values)
        for i in 1:iter
            for (idx,eps) in enumerate(fw_epsilon_values)
                for (idx2, min_num_lower_val) in enumerate(min_num_lower_values)
                    dual_gap_decay_factor = value
                    min_number_lower = min_num_lower_val
                    fw_epsilon = eps
                    seed = seed_val
                    data = @timed _, time_lmo, result = Boscia.solve(f, grad!, lmo; verbose=true, dual_gap_decay_factor=dual_gap_decay_factor, min_number_lower=min_number_lower, fw_epsilon = fw_epsilon, print_iter=1)
                    df = DataFrame(seed=seed, dimension=n, min_number_lower=min_number_lower, adaptive_gap=dual_gap_decay_factor, iteration=result[:number_nodes], time=result[:total_time_in_sec]*1000, memory=data[3], lb=result[:list_lb], ub=result[:list_ub], list_time=result[:list_time], list_num_nodes=result[:list_num_nodes], list_lmo_calls=result[:list_lmo_calls_acc], active_set_size=result[:list_active_set_size], discarded_set_size=result[:list_discarded_set_size])
                    file_name = "experiments/csv/early_stopping_" * example * "_" * string(n) * "_" * string(k) * "_" * string(seed) * "_" * string(min_number_lower) * "_" * string(dual_gap_decay_factor) * "_" * string(fw_epsilon) * "_" * string(i) *".csv"
                    CSV.write(file_name, df, append=false)
                end
            end
        end
    end
end  

MOI.optimize!(o)
println("SCIP")
@show MOI.get(o, MOI.ObjectiveValue())
@show MOI.get(o, MOI.SolveTimeSec())

