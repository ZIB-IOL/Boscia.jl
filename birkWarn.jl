using SCIP
using FrankWolfe
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface

gradient1 = [6.379211273943586e-7, 5.295988958209374e-7, 1.0386358745839708e-6, 1.0514506535441726e-6, 8.608421032396674e-7, 1.8074067472451105e-5, 1.7324152694869754e-5, 1.8189859484818438e-5, 1.8542007474288413e-5, 1.8037831183648212e-5, 1.1105130827693266e-6, 9.72334610263234e-7, 1.1516701333980084e-6, 1.1056714399604317e-6, 1.2613348674833658e-6, 1.4153105305225733e-6, 1.8928261615869246e-6, 1.6780452606335317e-6, 2.064790638633962e-6, 1.984840338184468e-6, 1.2784076545008238e-6, 1.170739178102398e-6, 1.6008103024336973e-6, 1.344224123012161e-6, 1.2027104786538345e-6, 6.379211273943586e-7, 5.295988958209374e-7, 1.0386358745839708e-6, 1.0514506535441726e-6, 8.608421032396674e-7, 1.8074067472451105e-5, 1.7324152694869754e-5, 1.8189859484818438e-5, 1.8542007474288413e-5, 1.8037831183648212e-5, 1.1105130827693266e-6, 9.72334610263234e-7, 1.1516701333980084e-6, 1.1056714399604317e-6, 1.2613348674833658e-6, 1.4153105305225733e-6, 1.8928261615869246e-6, 1.6780452606335317e-6, 2.064790638633962e-6, 1.984840338184468e-6, 1.2784076545008238e-6, 1.170739178102398e-6, 1.6008103024336973e-6, 1.344224123012161e-6, 1.2027104786538345e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
gradient2 = [6.379211273943586e-7, 2.295988958209374e-7, 1.0386358745839708e-6, 1.0514806535441726e-6, 1.608421032396674e-7, 3.8074067472451105e-5, 1.7324152694869754e-5, 1.0000059484818438e-5, 1.8542007474288413e-5, 6.8037831183648212e-5, 1.1105130827693266e-6, 9.72334610263234e-7, 1.1516701333980084e-6, 1.1056714399604317e-6, 1.2613348674833658e-6, 1.4153105305225733e-6, 1.8928261615869246e-6, 1.6780452606335317e-6, 2.064790638633962e-6, 1.984840338184468e-6, 1.2784076545008238e-6, 1.170739178102398e-6, 1.6008103024336973e-6, 1.344224123012161e-6, 1.2027104786538345e-6, 6.379211273943586e-7, 5.295988958209374e-7, 1.0386358745839708e-6, 1.0514506535441726e-6, 8.608421032396674e-7, 1.8074067472451105e-5, 1.7324152694869754e-5, 4.8189859484818438e-5, 1.8542007474288413e-5, 1.8037831183648212e-5, 1.1105130827693266e-6, 7.72334610263234e-7, 1.1516701333980084e-6, 1.1056714399604317e-6, 1.2613348674833658e-6, 1.4153105305225733e-6, 1.8928261615869246e-6, 1.6780452606335317e-6, 2.064790638633962e-6, 1.984840338184468e-6, 1.2784076545008238e-6, 1.170739178102398e-6, 1.6008103024336973e-6, 1.344224123012161e-6, 1.2027104786538345e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

""" building the lmo """
n = 5
k = 2

o = SCIP.Optimizer()#MOI.Utilities.Model{Float64}()

    Y = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
    X = [reshape(MOI.add_variables(o, n^2), n, n) for _ in 1:k]
    theta = MOI.add_variables(o, k)

    for i in 1:k
        MOI.add_constraint.(o, Y[i], MOI.GreaterThan(0.0))
        MOI.add_constraint.(o, Y[i], MOI.LessThan(1.0))
        MOI.add_constraint.(o, X[i], MOI.ZeroOne())

        MOI.add_constraint(o, theta[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, theta[i], MOI.LessThan(1.0))
        # doubly stochastic constraints
        MOI.add_constraint.(
            o, vec(sum(X[i], dims=1, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
            MOI.EqualTo(1.0),
        )
        MOI.add_constraint.(
            o, vec(sum(X[i], dims=1, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
            MOI.EqualTo(1.0),
        )
        # 0 ≤ Y_i ≤ X_i
        MOI.add_constraint.(
            o, 1.0 * Y[i] - X[i],
            MOI.LessThan(0.0),
        )
        # 0 ≤ θ_i - Y_i ≤ 1 - X_i
        MOI.add_constraint.(
            o, 1.0 * theta[i] .- Y[i] .+ X[i],
            MOI.LessThan(1.0),
        )
    end
    MOI.add_constraint(o, sum(theta, init=0.0), MOI.EqualTo(1.0))
	MOI.add_constraint(o, X[1][2,1], MOI.GreaterThan(1.0))
	MOI.add_constraint(o, X[1][4,5], MOI.GreaterThan(1.0))

	MOI.add_constraint(o, X[1][1,2], MOI.LessThan(0.0))
	MOI.add_constraint(o, X[1][5,2], MOI.LessThan(0.0))
	MOI.add_constraint(o, X[1][2,5], MOI.LessThan(0.0))
	MOI.add_constraint(o, X[1][5,5], MOI.LessThan(0.0))

MOI.set(
           o,
           MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
           MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(gradient1, MOI.get(o, MOI.ListOfVariableIndices())), 0.0),
       );

MOI.optimize!(o)


MOI.set(
    o,
    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(gradient2, MOI.get(o, MOI.ListOfVariableIndices())), 0.0),
);

MOI.optimize!(o)

MOI.set(
    o,
    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(gradient1, MOI.get(o, MOI.ListOfVariableIndices())), 0.0),
);

MOI.optimize!(o)


""" using the MPS file """
#=src = MOI.FileFormats.Model(filename="file1.mps")
MOI.read_from_file(src, joinpath(@__DIR__, "file1.mps"))

o = SCIP.Optimizer()
index_map = MOI.copy_to(o, src);

MOI.optimize!(o)

ind = []
for i in 1:102
    push!(ind, MOI.getindex(index_map, MOI.VariableIndex(i)).value)
end

MOI.set(
    o,
    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(gradient2[ind], MOI.get(o, MOI.ListOfVariableIndices())), 0.0),
);

MOI.optimize!(o)

MOI.set(
    o,
    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(gradient1[ind], MOI.get(o, MOI.ListOfVariableIndices())), 0.0),
);

MOI.optimize!(o)=#