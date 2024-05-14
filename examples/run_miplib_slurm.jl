modes = ["no_tightening", "global_tightening", "local_tightening", "afw", "no_ss", "no_as", "no_as_no_ss", "strong_branching", "hybrid_branching", "strong_convexity"]

for mode in modes
    if mode == "hybrid_branching"
        depths = [1, 2, 5, 10, 20]
    else
        depths = [1]
    end

    for depth in depths
        for example in ["neos5", "ran14x18-disj-8", "pg5_34", "22433"]
            for num_v in 4:8
                for seed in 1:3
                    run(`sbatch batch_miplib.sh $example $num_v $seed $mode $depth`)
                end
            end
        end
    end
end
