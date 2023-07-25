"""
    MILMO

Supertype for the Mixed Integer Linear Minimization Oracles
"""
abstract type MixedIntegerLinearMinimizationOracle end 

"""
Multiple dispatch of FrankWolfe.compute_extreme_point
"""
function compute_extreme_point end 

"""
"""
function optimize! end 

function get_lower_bounds end

function get_upper_bounds end 

function set_lower_bounds end

function set_upper_bounds end

