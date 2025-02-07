using Boscia

"""
MIPLIB instances.
"""

include("examples/mps-example.jl")
# Boscia
miplib_boscia("22433", 1, 4)
# BnB + Ipopt
miplib_ipopt("22433", 1, 4)
# SCIP OA 
miplib_scip(1, 4, example="22433")
# Pavito 
miplib_pavito(1, 4, example="22433")
# SHOT
miplib_shot(1, 4, example="22433") 



"""
Portfolio instances.
"""

include("examples/portfolio.jl")
## Pure Integer
# Boscia
portfolio_boscia(seed=1, dimension=20, mode="integer")
# BnB + Ipopt
portfolio_ipopt(seed=1, dimension=20, mode="integer")
# SCIP  OA 
portfolio_scip(seed=1,dimension=20, mode="integer")
# Pavito 
portfolio_pavito(seed=1, dimension=20, mode="integer")
# SHOT
portfolio_shot(seed=1, dimension=20, mode="integer")

## Mixed Integer 
# Boscia
portfolio_boscia(seed=1, dimension=20, mode="mixed")
# BnB + Ipopt
portfolio_ipopt(seed=1, dimension=20, mode="mixed")
# SCIP  OA 
portfolio_scip(seed=1,dimension=20, mode="mixed")
# Pavito 
portfolio_pavito(seed=1, dimension=20, mode="mixed")
# SHOT
portfolio_shot(seed=1, dimension=20, mode="mixed")



"""
Regression instances.
"""

## Poisson Regression
include("examples/poisson_reg.jl")
# Boscia
poisson_reg_boscia(seed=1, n=50, Ns=0.1)
# BnB + Ipopt
poisson_reg_ipopt(seed=1, n=50, Ns=0.1)
# SCIP  OA 
poisson_reg_scip(seed=1, n=50, Ns=0.1)
# Pavito 
poisson_reg_pavito(seed=1, n=50, Ns=0.1)
# SHOT
poisson_reg_shot(seed=1, n=50, Ns=0.1) 

## Sparse Regression
include("examples/sparse_reg.jl")
# Boscia
sparse_reg_boscia(seed=1, n=15)
# BnB + Ipopt
sparse_reg_ipopt(seed=1, n=15)
# SCIP  OA 
sparse_reg_scip(seed=1, n=15)
# Pavito 
sparse_reg_pavito(seed=1, n=15)
# SHOT
sparse_reg_shot(seed=1, n=15) 

## Sparse Log Regression
include("examples/sparse_log_reg.jl")
# Boscia
sparse_log_reg_boscia(seed=1, dimension=5, M=0.1, var_A=1.0)
# BnB + Ipopt
sparse_log_reg_ipopt(seed=1, n=5, M=0.1, var_A=1.0)
# SCIP  OA 
sparse_log_reg_scip(seed=1, dimension=5, M=0.1, var_A=1.0)
# Pavito 
sparse_log_reg_pavito(seed=1, dimension=5, M=0.1, var_A=1.0)
# SHOT
sparse_log_reg_shot(seed=1, dimension=5, M=0.1, var_A=1.0) 



"""
Tailed Regression instances.
"""

## Tailed Sparse Regression
include("examples/tailed_cardinality.jl")
# Boscia
tailed_cardinality_sparse_reg_boscia(seed=1, dimension=15)
# SCIP  OA 
tailed_cardinality_sparse_reg_scip(seed=1, dimension=15)

## Tailed Log Sparse Regression
include("examples/tailed_cardinality_sparse_log_reg.jl")
# Boscia
tailed_cardinality_sparse_log_reg_boscia(seed=1, dimension=5, M=0.1, var_A=1.0)
# SCIP  OA 
tailed_cardinality_sparse_log_reg_scip(seed=1, dimension=5, M=0.1, var_A=1.0)
