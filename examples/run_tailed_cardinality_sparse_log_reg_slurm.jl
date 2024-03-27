include("sparse_log_reg.jl")    

for dimension in [5:5:20;]
    for seed in 1:10
        @show seed, dimension
        for M in [0.1,1]
            for var_A in [1,5]
                run(`sbatch batch_tailed_cardinality_sparse_log_reg.sh $seed $dimension $M $var_A`)
            end
        end
    end
end

