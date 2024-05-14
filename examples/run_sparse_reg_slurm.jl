
modes = ["no_tightening", "global_tightening", "local_tightening", "afw", "no_ss", "no_as", "no_as_no_ss", "strong_branching", "hybrid_branching"]

for mode in modes
    if mode == "hybrid_branching"
        depths = [1, 2, 5, 10, 20]
    else
        depths = [1]
    end

    for depth in depths
        for dimension in 15:30
            for seed in 1:10
                @show seed, dimension
                run(`sbatch batch_sparse_reg.sh $seed $dimension $mode $depth`)
            end
        end
    end
end
