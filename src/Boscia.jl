module Boscia

using FrankWolfe
using Random
using SCIP
import MathOptInterface
import Bonobo
using Printf
using Dates
const MOI = MathOptInterface
const MOIU = MOI.Utilities

import MathOptSetDistances as MOD

include("defs.jl")
include("time_tracking_lmo.jl")
include("bounds.jl")
include("frank_wolfe_variants.jl")
include("node.jl")
include("custom_bonobo.jl")
include("callbacks.jl")
include("problem.jl")
include("infeasible_pairwise.jl")
include("heuristics.jl")
include("strong_branching.jl")
include("utilities.jl")
include("interface.jl")

end # module
