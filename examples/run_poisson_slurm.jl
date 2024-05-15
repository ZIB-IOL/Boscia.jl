#modes = ["no_tightening", "global_tightening", "local_tightening", "afw", "no_ss", "no_as", "no_as_no_ss"]
modes = ["no_ss","no_as","no_as_no_ss"]
#modes = ["strong_branching", "hybrid_branching"]

for mode in modes
    @show mode
    if mode == "hybrid_branching"
        depths = [1, 2, 5, 10, 20]
    else
        depths = [1]
    end

    for depth in depths
        for dimension in [50:20:100;]
            for seed in 1:10
                for ns in [0.1,1,5,10]
                    @show seed, dimension, ns
                    run(`sbatch batch_poisson.sh $seed $dimension $ns $mode $depth`)
                end
            end
        end
    end
end
