for dimension in 16:30
    for seed in 1:2#10
        @show seed, dimension
        run(`sbatch batch_sparse_reg.sh $seed $dimension`)
    end
end
