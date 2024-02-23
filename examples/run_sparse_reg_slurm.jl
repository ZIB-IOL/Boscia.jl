for dimension in 15:30
    for seed in 1:1#10
        @show seed, dimension
        run(`sbatch batch_sparse_reg.sh $seed $dimension`)
    end
end
