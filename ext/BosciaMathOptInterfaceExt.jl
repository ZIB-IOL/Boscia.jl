module BosciaMathOptInterfaceExt

using Boscia
using FrankWolfe
using MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities

import MathOptSetDistances as MOD

include("MOI_bounded_oracle.jl")

end # module