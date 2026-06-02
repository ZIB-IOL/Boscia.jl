using LinearAlgebra
using Random
using Boscia
using StableRNGs
using FrankWolfe


seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

# ── data ──────────────────────────────────────────
n = 8
m = 70000

const X = rand(rng, Float64, n, m)
const Y = rand(rng, Float64, n, m)

λ_min = minimum(eigvals(X * X'))
@show λ_min

int_vars = collect(1:n*n)
lbs = fill(-8.0, n*n)
ubs = fill(7.0, n*n)

blmo = Boscia.CubeLMO(lbs, ubs, int_vars)
lmo = Boscia.ManagedBoundedLMO(blmo, lbs, ubs, int_vars, n*n)

# ── obj ──────────────────────────────────────
function f(w)
    W = reshape(w, n, n)
    return norm(W * X .- Y)^2
end

function grad!(storage, w)
    R_buf = zeros(Float64, n, m)
    mul!(R_buf, reshape(w, n, n), X)
    R_buf .-= Y
    mul!(reshape(storage, n, n), R_buf, X')
    @. storage *= 2f0
end

variant = Boscia.PairwiseFrankWolfe()

# ── Solve ──────────────────────────────────────────────
settings = Boscia.create_default_settings()
settings.branch_and_bound[:verbose] = true
settings.branch_and_bound[:print_iter] = 50
settings.branch_and_bound[:time_limit] = 3600
settings.frank_wolfe[:variant] = variant
settings.frank_wolfe[:fw_verbose] = true
settings.frank_wolfe[:max_fw_iter] = 2000

x, _, _ = Boscia.solve(f, grad!, lmo, settings=settings)
