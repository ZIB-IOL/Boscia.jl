modes = ["default", "no_tightening", "global_tightening", "local_tightening", "afw", "no_ss", "no_as", "no_as_no_ss", "strong_convexity", "strong_branching", "hybrid_branching"]
solvers = ["Boscia", "SCIP", "Ipopt", "Pavito", "SHOT"]

for solver in solvers
    for mode in modes
        @show mode
        if mode == "hybrid_branching"
            depths = [1, 2, 5, 10, 20]
        else
            depths = [1]
        end

        for depth in depths
            for dimension in [50:20:100;]
                for seed in [1,5,10]#1:10
                    for ns in [0.1,1,5,10]
                        @show seed, dimension, ns
                        run(`sbatch batch_poisson.sh $seed $dimension $ns $mode $depth $solver`)
                    end
                end
            end
        end
    end
end 

