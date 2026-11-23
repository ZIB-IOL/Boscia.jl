module Boscia

using FrankWolfe
import FrankWolfe: compute_extreme_point
export compute_extreme_point
import FrankWolfe: LinearMinimizationOracle
export LinearMinimizationOracle
import FrankWolfe: is_decomposition_invariant_oracle
export is_decomposition_invariant_oracle

import FrankWolfe: compute_inface_extreme_point
export compute_inface_extreme_point

import FrankWolfe: dicg_maximum_step
export dicg_maximum_step

import FrankWolfe: UnitSimplexLMO
export UnitSimplexLMO

import FrankWolfe: ZeroOneHypercubeLMO
export ZeroOneHypercubeLMO

import FrankWolfe: ProbabilitySimplexLMO
export ProbabilitySimplexLMO

import FrankWolfe: BoxLMO
export BoxLMO
using Random
using LinearAlgebra
using Printf
using Dates
using MathOptInterface
using SparseArrays
using DataStructures
using NamedTupleTools
import Statistics: mean
const MOI = MathOptInterface
const MOIU = MOI.Utilities
import MathOptSetDistances as MOD

include("integer_bounds.jl")
include("defs.jl")
include("node.jl")
include("bnb_tree_operate.jl")
include("bnb_tree.jl")
include("blmo_interface.jl")
include("time_tracking_lmo.jl")
include("frank_wolfe_variants.jl")
include("build_lmo.jl")
include("tightenings.jl")
include("callbacks.jl")
include("problem.jl")
include("heuristics.jl")
include("strong_branching.jl")
include("branching_strategies.jl")
include("utilities.jl")
include("settings.jl")
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
