mode = "integer"

# for dimension in [20:5:120;]
#     for seed in 2:10
#         @show seed, dimension
#         run(`sbatch batch_portfolio.sh $seed $dimension $mode`)
#     end
# end
run(`sbatch batch_portfolio.sh $4 $30 $mode`)
run(`sbatch batch_portfolio.sh $10 $30 $mode`)
run(`sbatch batch_portfolio.sh $2 $35 $mode`)
run(`sbatch batch_portfolio.sh $6 $35 $mode`)
run(`sbatch batch_portfolio.sh $6 $40 $mode`)
run(`sbatch batch_portfolio.sh $10 $80 $mode`)
run(`sbatch batch_portfolio.sh $10 $115 $mode`)

mode = "mixed"

# for dimension in [20:5:120;]
#     for seed in 2:10
#         @show seed, dimension
#         run(`sbatch batch_portfolio.sh $seed $dimension $mode`)
#     end
# end
run(`sbatch batch_portfolio.sh $1 $115 $mode`)
run(`sbatch batch_portfolio.sh $2 $20 $mode`)
run(`sbatch batch_portfolio.sh $7 $25 $mode`)
run(`sbatch batch_portfolio.sh $10 $25 $mode`)
run(`sbatch batch_portfolio.sh $4 $30 $mode`)
run(`sbatch batch_portfolio.sh $10 $30 $mode`)
run(`sbatch batch_portfolio.sh $2 $35 $mode`)
run(`sbatch batch_portfolio.sh $5 $55 $mode`)
run(`sbatch batch_portfolio.sh $10 $80 $mode`)
run(`sbatch batch_portfolio.sh $4 $85 $mode`)
run(`sbatch batch_portfolio.sh $9 $100 $mode`)
run(`sbatch batch_portfolio.sh $10 $100 $mode`)
run(`sbatch batch_portfolio.sh $3 $105 $mode`)
run(`sbatch batch_portfolio.sh $5 $105 $mode`)
run(`sbatch batch_portfolio.sh $9 $105 $mode`)
run(`sbatch batch_portfolio.sh $8 $110 $mode`)
run(`sbatch batch_portfolio.sh $9 $110 $mode`)
run(`sbatch batch_portfolio.sh $10 $110 $mode`)
run(`sbatch batch_portfolio.sh $6 $115 $mode`)
run(`sbatch batch_portfolio.sh $8 $115 $mode`)
run(`sbatch batch_portfolio.sh $9 $115 $mode`)
run(`sbatch batch_portfolio.sh $10 $115 $mode`)
run(`sbatch batch_portfolio.sh $2 $120 $mode`)
run(`sbatch batch_portfolio.sh $6 $120 $mode`)




