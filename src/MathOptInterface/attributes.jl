# Optimizer attributes of Boscia.Optimizer: settings the user can query or change
# before solving (name, verbosity, limits, tolerances, ...).

# Each attribute has up to three methods:
#   MOI.supports(model, attr)        -> is this attribute supported?    
#   MOI.get(model, attr)             -> current value                   
#   MOI.set(model, attr, value)      -> change the value                


# SolverName
function MOI.get(model::Optimizer, ::MOI.SolverName)
    return "Boscia"
end

# SolverVersion
function MOI.get(model::Optimizer, ::MOI.SolverVersion)
    return
end

# Name 
function MOI.get(model::Optimizer, attr::MOI.Name)
    return MOI.get(model.model, attr)
end

function MOI.set(model::Optimizer, attr::MOI.Name, v::String)
    return MOI.set(model.model, attr, v)
end

MOI.supports(::Optimizer, ::MOI.Name) = true

# Silent 
function MOI.get(model::Optimizer, ::MOI.Silent)
    return model.silent
end

function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    model.silent = value
    return
end

MOI.supports(::Optimizer, ::MOI.Silent) = true

# TimeLimitSec
function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return model.timeout
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Union{Nothing,Float64})
    model.timeout = value
    return
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
