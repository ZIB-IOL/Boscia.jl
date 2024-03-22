for dimension in 15:30
    for seed in 2:10
        @show seed, dimension
        run(`sbatch batch_sparse_reg.sh $seed $dimension`)
    end
end
