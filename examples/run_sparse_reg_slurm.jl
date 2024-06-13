
#modes = ["no_tightening", "global_tightening", "local_tightening", "afw", "no_ss", "no_as", "no_as_no_ss", "strong_branching"]

#modes = ["no_as","no_ss","no_as_no_ss"]
modes = ["default","no_tightening", "global_tightening", "local_tightening", "hybrid_branching", "strong_branching"]
modes = ["dual_gap_decay_factor"]
for mode in modes
    if mode == "hybrid_branching"
        depths = [20]#[1, 2, 5, 10, 20]
    else
        depths = [1]
    end

    if mode == "dual_gap_decay_factor"
        dual_gap_decay_factors = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
        epsilons = [1e-2, 1e-3, 5e-3, 1e-4]
        seeds = 1:3
    else
        dual_gap_decay_factors = [0.0]
        epsilons = [0.0]
        seeds = 1:10
    end

    for depth in depths
        for dimension in 15:30
            for seed in seeds
                for factor in dual_gap_decay_factors
                    for epsilon in epsilons
                        @show seed, dimension
                        run(`sbatch batch_sparse_reg.sh $seed $dimension $mode $depth $factor $epsilon`)
                    end 
                end
            end
        end
    end
end
