mode = "no_tightening"

for dimension in 15:30
    for seed in 1:10
        @show seed, dimension
        run(`sbatch batch_tailed_cardinality_sparse_reg.sh $seed $dimension $mode`)
    end
end
