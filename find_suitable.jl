using LinearAlgebra, SparseArrays
using JuMP
using Arpack
using JSON

# qp_instances = readdir("/nfs/optimi/kombadon/MINLP/instances/qplib/lp/", join=true)
qp_instances = ["QPLIB_9030.lp.gz",  "QPLIB_1703.lp", "QPLIB_3913.lp", "QPLIB_8785.lp.gz"]

function process_instance(o)
    all_vars = MOI.get(o, MOI.ListOfVariableIndices())
    bin_cons = MOI.get(o, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.ZeroOne}())
    bin_vars = MOI.VariableIndex.(getproperty.(bin_cons, :value))
    nbin = length(bin_vars)
    other_vars = [v for v in all_vars if v ∉ bin_vars]
    nother = length(other_vars)
    # pure binary form for all continuous QPs
    add_binary = false
    if nbin + length(MOI.get(o, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.Integer}())) == 0
        for v in all_vars
            MOI.add_constraint(o, v, MOI.ZeroOne())
        end
        bin_cons = MOI.get(o, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.ZeroOne}())
        bin_vars = MOI.VariableIndex.(getproperty.(bin_cons, :value))
        nbin = length(all_vars)
        @assert length(bin_vars) == nbin
        other_vars = MOI.VariableIndex[]
        nother = 0
        add_binary = true
    end
    ftype = MOI.get(o, MOI.ObjectiveFunctionType())
    Q = spzeros(nbin, nbin)
    S = spzeros(nother, nother)
    q = Dict{String, Float64}()
    s = Dict{String, Float64}()
    Qdict = Dict{Tuple{String, String}, Float64}()
    Sdict = Dict{Tuple{String, String}, Float64}()
    if !(ftype <: MOI.ScalarQuadraticFunction)
        @info "non-quadratic objective"
        return false, Qdict, Sdict, q, s, add_binary
    end
    f = MOI.get(o, MOI.ObjectiveFunction{ftype}())
    for term in f.quadratic_terms
        if term.coefficient == 0.0
           @warn "zero coeff"
           continue
        end
        if (term.variable_1 ∈ bin_vars) != (term.variable_2 ∈ bin_vars)
            @info "joint quadratic term between a binary and non-binary"
           return false, Qdict, Sdict, q, s, add_binary
        end
        if term.variable_1 ∈ bin_vars
            ia = findfirst(==(term.variable_1), bin_vars)
            ib = findfirst(==(term.variable_1), bin_vars)
            Q[ia,ib] = term.coefficient
            Q[ib,ia] = term.coefficient
            Qdict[(
                MOI.get(o, MOI.VariableName(), term.variable_1),
                MOI.get(o, MOI.VariableName(), term.variable_2),
            )] = term.coefficient
        else
            ia = findfirst(==(term.variable_1), other_vars)
            ib = findfirst(==(term.variable_1), other_vars)
            S[ia,ib] = S[ib,ia] = term.coefficient
            Sdict[(
                MOI.get(o, MOI.VariableName(), term.variable_1),
                MOI.get(o, MOI.VariableName(), term.variable_2),
            )] = term.coefficient
        end
    end
    for term in f.affine_terms
        varname = MOI.get(o, MOI.VariableName(), term.variable)
        if term.variable ∈ bin_vars
            q[varname] = term.coefficient
        else
            s[varname] = term.coefficient
        end
    end
    # non-PSD on other variables
    if !isposdef(S)
        @info "nonposdef nonbins"
        return false, Qdict, Sdict, q, s, add_binary
    end
    if length(Q) == 0
        @info "no binary"
        return true, Qdict, Sdict, q, s, add_binary
    end
    if norm(Q) <= 1e-5
        @info "no quadratic term"
        return false, Qdict, Sdict, q, s, add_binary
    end
    if isdiag(Q)
        λmin = minimum(diag(Q))
    else
        Q_eigvals = Arpack.eigs(Q, nev=1, which=:SR, check=1)
        if isempty(Q_eigvals[1])
            λmin = 0.0
        else
            λmin::Float64 = Q_eigvals[1][1]
        end
    end
    if λmin >= -1e-7
        @info "posdef bins"
        return true, Qdict, Sdict, q, s, add_binary
    end
    @info "adding diagonal $(-λmin)I"
    for v in bin_vars
        varname = MOI.get(o, MOI.VariableName(), v)
        Qdict[(varname, varname)] = get(Qdict, (varname, varname), 0.0) - λmin + 1e-6
    end
    return true, Qdict, Sdict, q, s, add_binary
end

# too large for reasonable machine
nok_instances = String[]
ok_instances = []

for instance in qp_instances
    if instance in nok_instances
        continue
    end
    @info "$instance"
    o = MOI.Utilities.Model{Float64}()
    try
        MOI.read_from_file(o, instance)
    catch e
        @warn "Exception \n$e\n when solving $instance"
        push!(nok_instances, instance)
        continue
    end
    worked, Qdict, Sdict, q, s, added_binary = process_instance(o)
    if worked
        push!(
            ok_instances,
            (; path=instance, Qdict, Sdict, q, s, added_binary=added_binary),
        )
    end
end

open("nok_instances.txt", "w") do f
    write(f, join(nok_instances, '\n'))
end

open("filtered_instances.json", "w") do f
    write(f, JSON.json(ok_instances))
end
