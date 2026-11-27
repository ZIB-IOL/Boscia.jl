"""
    ZeroOneHypercube(dimension::Int)

The set ``\\{x \\in \\mathbb{R}^n : 0 \\le x_i \\le 1 \\text{ for all } i\\}``,
where `n = dimension`.
"""
struct ZeroOneHypercube <: MOI.AbstractVectorSet
    dimension::Int
end

"""
    ProbabilitySimplex(dimension::Int, N::Real)

The set ``\\{x \\in \\mathbb{R}^n : x_i \\ge 0 \\text{ for all } i, \\sum_i x_i = N\\}``,
where `n = dimension`.
"""
struct ProbabilitySimplex <: MOI.AbstractVectorSet
    dimension::Int
    N::Float64
end

"""
    UnitSimplex(dimension::Int, N::Real)

The set ``\\{x \\in \\mathbb{R}^n : x_i \\ge 0 \\text{ for all } i, \\sum_i x_i \\le N\\}``,
where `n = dimension`.
"""
struct UnitSimplex <: MOI.AbstractVectorSet
    dimension::Int
    N::Float64
end

"""
    HyperSimplex(dimension::Int, K::Int, radius::Float64)

The set ``\\{x \\in \\mathbb{R}^n : 0 \\le x_i \\le \\tau \\text{ for all } i, \\sum_i x_i = K\\}``,
where `n = dimension` and `τ = radius`.
"""
struct HyperSimplex <: MOI.AbstractVectorSet
    dimension::Int
    K::Int
    radius::Float64
end

"""
    UnitHyperSimplex(dimension::Int, K::Int, radius::Float64)

The set ``\\{x \\in \\mathbb{R}^n : 0 \\le x_i \\le \\tau \\text{ for all } i, \\sum_i x_i \\le K\\}``,
where `n = dimension` and `τ = radius`.
"""
struct UnitHyperSimplex <: MOI.AbstractVectorSet
    dimension::Int
    K::Int
    radius::Float64
end

"""
    ReverseKnapsack(dimension::Int, N::Real, upper::Real)

The set ``\\{x \\in \\mathbb{R}^n : 0 \\le x_i \\le u \\text{ for all } i, \\sum_i x_i \\ge N\\}``,
where `n = dimension` and `u = upper`.
"""
struct ReverseKnapsack <: MOI.AbstractVectorSet
    dimension::Int
    N::Float64
    upper::Float64
end


# CustomSet 
const CustomSet = Union{
    ZeroOneHypercube,
    ProbabilitySimplex,
    UnitSimplex,
    HyperSimplex,
    UnitHyperSimplex,
    ReverseKnapsack,
}
