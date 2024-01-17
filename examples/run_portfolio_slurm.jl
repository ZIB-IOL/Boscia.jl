mode = "integer"

for dimension in [20:5:120;]
    for seed in 1:10
        @show seed, dimension
        run(`sbatch batch_portfolio.sh $seed $dimension $mode`)
    end
end

# mode = "mixed"

# for dimension in [20:5:120;]
#     for seed in 1:10
#         @show seed, dimension
#         run(`sbatch batch_portfolio.sh $seed $dimension $mode`)
#     end
# end
