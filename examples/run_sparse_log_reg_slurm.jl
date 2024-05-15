include("sparse_log_reg.jl")   

modes = ["no_tightening", "global_tightening", "local_tightening", "afw", "no_ss", "no_as", "no_as_no_ss"]

#modes = ["strong_branching", "hybrid_branching"]

for mode in modes
    if mode == "hybrid_branching"
        depths = [1, 2, 5, 10, 20]
    else
        depths = [1]
    end

    for depth in depths

        for dimension in [5:5:20;]
            for seed in 1:10
            @show seed, dimension
                for M in [0.1,1]
                    for var_A in [1,5]
                        k = Float64(dimension)
                        run(`sbatch batch_sparse_log_reg.sh $seed $dimension $M $k $var_A $mode $depth`)
                    end
                end
            end
        end
    end
end

