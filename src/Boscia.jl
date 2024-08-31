module Boscia

using FrankWolfe
import FrankWolfe: compute_extreme_point
export compute_extreme_point
using Random
import Bonobo
using Printf
using Dates
using MathOptInterface
using SparseArrays
using CSV
using DataFrames
const MOI = MathOptInterface
const MOIU = MOI.Utilities
import MathOptSetDistances as MOD

include("integer_bounds.jl")
include("blmo_interface.jl")
include("time_tracking_lmo.jl")
include("frank_wolfe_variants.jl")
include("build_lmo.jl")
include("node.jl")
include("custom_bonobo.jl")
include("callbacks.jl")
include("problem.jl")
include("heuristics.jl")
include("strong_branching.jl")
include("pseudo_branching.jl")
include("utilities.jl")
include("interface.jl")
include("managed_blmo.jl")
include("MOI_bounded_oracle.jl")
include("polytope_blmos.jl")

# For extensions
if !isdefined(Base, :get_extension)
    include("../ext/BosciaSCIPExt.jl")
    include("../ext/BosciaHiGHSExt.jl")
end

end # module
