using SCIP
import MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities
using LinearAlgebra
using Boscia
using FrankWolfe
using JSON
using SparseArrays
using HiGHS

all_instances = JSON.parsefile(joinpath(@__DIR__, "../filtered_instances.json"))

idx = findfirst(d -> occursin("2880.lp", d["path"]), all_instances)
instance_info = all_instances[idx]

function build_and_solve_scip(instance_info; timelimit=3600)
    path = instance_info["path"]
    @info "Instance $(basename(instance_info["path"]))"
    o_temp = MOIU.Model{Float64}()
    MathOptInterface.read_from_file(o_temp, path)
    f = MOI.get(o_temp, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
    if instance_info["added_binary"]
        MOI.add_constraint.(o_temp, MOI.get(o_temp, MOI.ListOfVariableIndices()), MOI.ZeroOne())
    end
    z = MOI.add_variable(o_temp)
    MOI.add_constraint(o_temp, f - z, MOI.LessThan(0.0))
    MOI.set(o_temp, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z)
    MOI.set(o_temp, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    o = HiGHS.Optimizer()
    MOI.copy_to(o, o_temp)
    MOI.set(o, MOI.TimeLimitSec(), timelimit)
    MOI.optimize!(o)
    runtime = MOI.get(o, MOI.SolveTimeSec())
    status = MOI.get(o, MOI.TerminationStatus())
    primal_bound = MOI.get(o, MOI.ObjectiveValue())
    dual_bound = MOI.get(o, MOI.ObjectiveBound())
    return (; runtime, status, primal_bound, dual_bound)
end

function solve_boscia(instance_info; timelimit=3600)
    path = instance_info["path"]
    o_temp = MOIU.Model{Float64}()
    MathOptInterface.read_from_file(o_temp, path)
    v_indices = MOI.get(o_temp, MOI.ListOfVariableIndices())
    varindex_list = MOI.get.(o_temp, MOI.VariableName(), v_indices)

    if instance_info["added_binary"]
        MOI.add_constraint.(o_temp, MOI.get(o_temp, MOI.ListOfVariableIndices()), MOI.ZeroOne())
    end
    Q = spzeros(length(varindex_list), length(varindex_list))
    q = zeros(length(varindex_list))
    f = MOI.get(o_temp, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
    for term in f.quadratic_terms
        Q[term.variable_1.value, term.variable_2.value] += term.coefficient
        Q[term.variable_2.value, term.variable_1.value] += term.coefficient
    end
    for term in f.affine_terms
        q[term.variable.value] += term.coefficient
    end

    bin_cons = MOI.get(o_temp, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.ZeroOne}())
    bin_vars = MOI.VariableIndex.(getproperty.(bin_cons, :value))
    bin_mask = BitVector((vidx ∈ bin_vars for vidx in v_indices))

    λmin = eigmin(Q)
    diag_term = Diagonal((abs(λmin) + 1e-6) * bin_mask)

    Q .+= diag_term
    q .-= (abs(λmin) + 1e-6) * bin_mask
    MOI.set(o_temp, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * v_indices[1])

    function objective(x)
        1/2 * dot(x, Q, x) + dot(q, x)
    end

    function grad!(storage, x)
        storage .= q
        mul!(storage, Q, x, 1.0, 1.0)
        return storage
    end
    o = SCIP.Optimizer()
    MOI.copy_to(o, o_temp)
    MOI.set(o, MOI.Silent(), true)
    lmo = FrankWolfe.MathOptLMO(o)
    x, _, result = Boscia.solve(objective, grad!, lmo, verbose=true, time_limit=timelimit)
    runtime = result[:total_time_in_sec]
    status = result[:status]
    primal_bound = result[:primal_objective]
    dual_bound = result[:dual_bound]
    (; runtime, status, primal_bound, dual_bound)
end
