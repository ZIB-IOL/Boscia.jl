using Statistics
using Distributions
using LinearAlgebra

# min h(sqrt(y' * M * y)) - r' * y
# s.t. a' * y <= b 
#           y >= 0
#           y_i in Z for i in I

n0 = 30
const r = 10 * rand(n0)
const a = rand(n0)
const Ω = 3 * rand(Float64)
const b = sum(a)
A1 = randn(n0, n0)
A1 = A1' * A1
const M1 = (A1 + A1') / 2
@assert isposdef(M1)


@testset "Buchheim et. al. mean risk" begin
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n0)
    I = collect(1:n0) #rand(1:n0, Int64(floor(n0/2)))
    for i in 1:n0
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        if i in I
            MOI.add_constraint(o, x[i], MOI.Integer())
        end
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(a, x), 0.0),
        MOI.LessThan(b),
    )
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n0), x), 0.0),
        MOI.GreaterThan(1.0),
    )
    lmo = FrankWolfe.MathOptLMO(o)

    # Define the root of the tree
    # we fix the direction so we can actually find a veriable to split on later!
    direction = Vector{Float64}(undef, n0)
    Random.rand!(direction)
    v = compute_extreme_point(time_lmo, direction)
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(v)[], 1)

    function f(x)
        return Ω * (x' * M1 * x) - r' * x
    end
    function grad!(storage, x)
        storage .= 2 * Ω * M1 * x - r
        return storage
    end

    x_,result = Boscia.solve(f, grad!, lmo, verbose)
    
    @test dot(a,x) <= b + 1e-6
    @test f(x) <= f(result[:raw_solution]) + 1e-6
end
