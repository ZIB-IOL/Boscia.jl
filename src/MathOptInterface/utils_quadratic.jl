"""
    build_objective_oracles(optimizer::Optimizer)

Build the Frank-Wolfe oracles `f(x)` and `grad!(storage, x)` from the
objective stored in `optimizer`.

- `ScalarQuadraticFunction`: written as `0.5 * x' * Q * x + q' * x + c`,
  with `f` and `grad!` computed directly from `Q`, `q` and `c`.
- `ScalarNonlinearFunction`: `f` and `grad!` are computed by MOI's automatic
  differentiation (`MOI.Nonlinear.Evaluator` with `SparseReverseMode`).

For quadratic objectives, convexity is checked: an error is thrown if
`sign * Q` has a negative eigenvalue, where `sign` is `-1` for
`MOI.MAX_SENSE` and `1` otherwise. For nonlinear objectives convexity
cannot be checked in general.
"""
function build_objective_oracles(optimizer::Optimizer)
    typ = MOI.get(optimizer, MOI.ObjectiveFunctionType())
    obj = MOI.get(optimizer, MOI.ObjectiveFunction{typ}())
    n = MOI.get(optimizer, MOI.NumberOfVariables())
    sense = MOI.get(optimizer, MOI.ObjectiveSense())

    if sense == MOI.MIN_SENSE
        sign = 1.0
    elseif sense == MOI.MAX_SENSE
        sign = -1.0
    else
        error("Unsupported objective sense: $sense")
    end

    if typ == MOI.ScalarQuadraticFunction{Float64}

        Q = zeros(Float64, n, n)
        q = zeros(Float64, n)

        # Linear part
        for term in obj.affine_terms
            q[term.variable.value] += term.coefficient
        end

        # Quadratic part 
        for term in obj.quadratic_terms
            i = term.variable_1.value
            j = term.variable_2.value
            a = term.coefficient

            if i == j
                Q[i, i] += a
            else
                Q[i, j] += a
                Q[j, i] += a
            end
        end

        c = obj.constant

        # check convexity
        @assert issymmetric(Q)
        λmin = eigmin(Symmetric(sign * Q))
        if λmin < -1e-8
            error(
                "The quadratic objective is not convex (concave for MAX_SENSE); Boscia requires a convex objective.",
            )
        end


        f = x -> sign * (0.5 * dot(x, Q, x) + dot(q, x) + c)

        # Gradient of 0.5 * x' * Q * x + q' * x + c is Q * x + q 
        grad! = (storage, x) -> begin
            mul!(storage, Q, x)
            storage .+= q
            storage .*= sign
            return storage
        end

        return f, grad!

    elseif typ == MOI.ScalarNonlinearFunction
        nlp = MOI.Nonlinear.Model()
        for (name, arity, args) in optimizer.user_defined_functions
            MOI.Nonlinear.register_operator(nlp, name, arity, args...)
        end
        MOI.Nonlinear.set_objective(nlp, obj)
        vars = MOI.get(optimizer, MOI.ListOfVariableIndices())
        evaluator = MOI.Nonlinear.Evaluator(nlp, MOI.Nonlinear.SparseReverseMode(), vars)
        MOI.initialize(evaluator, [:Grad])
        f = x -> sign * MOI.eval_objective(evaluator, x)

        grad! = (storage, x) -> begin
            MOI.eval_objective_gradient(evaluator, storage, x)
            storage .*= sign
            return storage
        end

        return f, grad!

    else
        error("Unsupported objective type: $typ")
    end
end
