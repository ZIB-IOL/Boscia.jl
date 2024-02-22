for dimension in [50]#:20:100;]
    for seed in 1:1#10
        for ns in [0.1,1,5,10]
            @show seed, dimension, ns
            run(`sbatch batch_poisson.sh $seed $dimension $ns`)
        end
    end
end
