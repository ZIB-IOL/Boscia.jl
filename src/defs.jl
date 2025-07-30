"""
Enum for the solving stage.
"""
@enum Solve_Stage::Int32 begin
    SOLVING = 0
    OPT_GAP_REACHED = 1
    OPT_TREE_EMPTY = 2
    TIME_LIMIT_REACHED = 3
    NODE_LIMIT_REACHED = 4
    USER_STOP = 5
end

"""
Enum for the different modes.
"""
@enum Mode::Int32 begin
    DEFAULT = 0
    HEURISTIC = 1
end

"""
Trivial domain function.
"""
_trivial_domain(x) = true

"""
Trivial domain point function.
"""
_trivial_domain_point(local_bounds::IntegerBounds) = nothing
