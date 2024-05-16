

#modes = ["no_tightening", "global_tightening", "local_tightening", "afw", "no_ss", "no_as", "no_as_no_ss"]

modes = ["strong_branching", "hybrid_branching"]

#modes = ["no_ss","no_as","no_as_no_ss"]
for mode in modes
    @show mode
    if mode == "hybrid_branching"
        depths = [1, 2, 5, 10, 20]
    else
        depths = [1]
    end

    for depth in depths
        set = "integer"

        for dimension in [20:5:120;]
            for seed in 1:10
                @show seed, dimension
                run(`sbatch batch_portfolio.sh $seed $dimension $set $mode $depth`)
            end
        end

        set = "mixed"

        for dimension in [20:5:120;]
            for seed in 1:10
                @show seed, dimension
                run(`sbatch batch_portfolio.sh $seed $dimension $set $mode $depth`)
            end
        end
    end
end
