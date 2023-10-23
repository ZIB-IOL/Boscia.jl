module Boscia

using FrankWolfe
import FrankWolfe: compute_extreme_point
export compute_extreme_point
using Random
import Bonobo
using Printf
using Dates

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
include("utilities.jl")
include("interface.jl")
include("cube_blmo.jl")

# For extensions
if !isdefined(Base, :get_extension)
    include("../ext/BosciaMathOptInterfaceExt.jl")
    include("../ext/BosciaSCIPExt.jl")
    include("../ext/BosciaHiGHSExt.jl")
  end

end # module
