module Boscia

using FrankWolfe
using Random
using SCIP
import MathOptInterface as MOI
const MOIU = MOI.Utilities

import Bonobo
using Printf
using Dates

import MathOptSetDistances as MOD

const XREF = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

include("time_tracking_lmo.jl")
include("bounds.jl")
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
