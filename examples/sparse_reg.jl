using Boscia
using FrankWolfe
using Random
using SCIP
# using Statistics
using LinearAlgebra
using Distributions
import MathOptInterface
#MOI = MathOptInterface
using CSV
using DataFrames
include("boscia_vs_scip.jl")

function sparse_reg(seed=1, n=20, iter = 1; bo_mode)
    limit = 1800

    Random.seed!(seed)
    p = 5 * n;
    k = ceil(n / 5);
    lambda_0 = rand(Float64);
    lambda_2 = 10.0 * rand(Float64);
    A = rand(Float64, n, p)
    y = rand(Float64, n)
    M = 2 * var(A)

    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, 2p)
    for i in p+1:2p
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, x[i], MOI.ZeroOne()) # or MOI.Integer()
    end
    for i in 1:p
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, M], [x[i], x[i+p]]), 0.0),
            MOI.GreaterThan(0.0),
        )
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -M], [x[i], x[i+p]]), 0.0),
            MOI.LessThan(0.0),
        )
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p), x[p+1:2p]), 0.0),
        MOI.LessThan(k),
    )
    lmo = FrankWolfe.MathOptLMO(o)

    function f(x)
        xv = @view(x[1:p])
        return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
    end

    function grad!(storage, x)
        storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
        storage[p+1:2p] .= lambda_0
        return storage
    end

    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)

    for i in 1:iter
        if bo_mode == "afw"
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=true, time_limit=limit, afw=true)
        elseif bo_mode == "as_ss"
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=false, warmstart_shadow_set=false)
        elseif bo_mode == "as"
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=false, warmstart_shadow_set=true)
        elseif bo_mode == "ss"
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit, warmstart_active_set=true, warmstart_shadow_set=false)
        elseif bo_mode == "boscia"
            x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=limit)
        end             
        # @show x, f(x)
        # @test dot(ai, x) <= bi + 1e-6
        # @test f(x) <= f(result[:raw_solution]) + 1e-6
        total_time_in_sec=result[:total_time_in_sec]
        status = result[:status]
        if occursin("Optimal", result[:status])
            status = "OPTIMAL"
        end
        df = DataFrame(seed=seed, dimension=n, p=p, k=k, time=total_time_in_sec, solution=result[:primal_objective], termination=status, ncalls=result[:lmo_calls])
        if bo_mode ==  "afw"
            file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_sparse_reg.csv")
        elseif bo_mode == "boscia"
            file_name = joinpath(@__DIR__, "csv/" * bo_mode * "_sparse_reg.csv")
        else 
            file_name = joinpath(@__DIR__,"csv/no_warm_start_" * bo_mode * "_sparse_reg.csv")
        end
        if !isfile(file_name)
            CSV.write(file_name, df, append=true, writeheader=true)
        else 
            CSV.write(file_name, df, append=true)
        end
        # display(df)
    end
end

function sparse_reg_scip(seed=1, n=20, iter = 1;)
    limit = 1800

    Random.seed!(seed)
    p = 5 * n;
    k = ceil(n / 5);
    lambda_0 = rand(Float64);
    lambda_2 = 10.0 * rand(Float64);
    A = rand(Float64, n, p)
    y = rand(Float64, n)
    M = 2 * var(A)

    function build_function()
        function f(x)
            xv = @view(x[1:p])
            return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
        end
    
        function grad!(storage, x)
            storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
            storage[p+1:2p] .= lambda_0
            return storage
        end
        return f, grad!
    end
    f, grad! = build_function()
    function build_scip_optimizer()
        o = SCIP.Optimizer()
        MOI.set(o, MOI.Silent(), true)
        MOI.empty!(o)
        x = MOI.add_variables(o, 2p)
        for i in p+1:2p
            MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
            MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
            MOI.add_constraint(o, x[i], MOI.ZeroOne()) # or MOI.Integer()
        end
        for i in 1:p
            MOI.add_constraint(
                o,
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, M], [x[i], x[i+p]]), 0.0),
                MOI.GreaterThan(0.0),
            )
            MOI.add_constraint(
                o,
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -M], [x[i], x[i+p]]), 0.0),
                MOI.LessThan(0.0),
            )
        end
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p), x[p+1:2p]), 0.0),
            MOI.LessThan(k),
        )        
        z_i = MOI.add_variable(o)
        
        epigraph_ch = GradientCutHandler(o, f, grad!, zeros(length(x)), z_i, x, 0)
        SCIP.include_conshdlr(o, epigraph_ch; needs_constraints=false, name="handler_gradient_cuts")
        
        MOI.set(o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
        # println("SCIP MODEL")
        # print(o)
        return o, epigraph_ch, x
    end

    for i in 1:iter
        o, epigraph_ch, x = build_scip_optimizer()
        MOI.set(o, MOI.TimeLimitSec(), limit)
        # MOI.set(o, MOI.AbsoluteGapTolerance(), 1.000000e-06) #AbsoluteGapTolerance not defined
        # MOI.set(o, MOI.RelativeGapTolerance(), 1.000000e-02)
        MOI.optimize!(o)
        time_scip = MOI.get(o, MOI.SolveTimeSec())
        vars_scip = MOI.get(o, MOI.VariablePrimal(), x)
        #@assert sum(ai.*vars_scip) <= bi + 1e-6 # constraint violated
        solution_scip = f(vars_scip)
        termination_scip = String(string(MOI.get(o, MOI.TerminationStatus())))
        ncalls_scip = epigraph_ch.ncalls

        df = DataFrame(seed=seed, dimension=n, p=p, k=k, time=time_scip, solution=solution_scip, termination=termination_scip, calls=ncalls_scip)
        file_name = joinpath(@__DIR__,"csv/scip_oa_sparse_reg.csv")
        if !isfile(file_name)
            CSV.write(file_name, df, append=true, writeheader=true)
        else 
            CSV.write(file_name, df, append=true)
        end
    end
end

sparse_reg(1, 1, bo_mode="afw")
sparse_reg_scip(1, 1)
