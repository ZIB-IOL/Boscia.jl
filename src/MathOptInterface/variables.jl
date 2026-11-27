function MOI.add_variable(optimizer::Optimizer)
    return MOI.add_variable(optimizer.model)
end

function MOI.get(optimizer::Optimizer, ::MOI.NumberOfVariables)
    return MOI.get(optimizer.model, MOI.NumberOfVariables())
end

function MOI.get(optimizer::Optimizer, ::MOI.ListOfVariableIndices)
    return MOI.get(optimizer.model, MOI.ListOfVariableIndices())
end
