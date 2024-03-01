# for dimension in [50:20:100;]
#    for seed in 1:10
#        for ns in [0.1,1,5,10]
#            @show seed, dimension, ns
#            run(`sbatch batch_poisson.sh $seed $dimension $ns`)
#        end
#    end
# end

run(`sbatch batch_poisson.sh $1 $50 $5.0`)
run(`sbatch batch_poisson.sh $1 $50 $10.0`)
run(`sbatch batch_poisson.sh $1 $70 $5.0`)
run(`sbatch batch_poisson.sh $1 $70 $10.0`)
run(`sbatch batch_poisson.sh $1 $90 $1.0`)
run(`sbatch batch_poisson.sh $1 $90 $5.0`)
run(`sbatch batch_poisson.sh $1 $90 $10.0`)
run(`sbatch batch_poisson.sh $3 $70 $5.0`)
run(`sbatch batch_poisson.sh $3 $70 $10.0`)
run(`sbatch batch_poisson.sh $7 $70 $5.0`)
run(`sbatch batch_poisson.sh $7 $70 $10.0`)
run(`sbatch batch_poisson.sh $1 $90 $5.0`)
run(`sbatch batch_poisson.sh $2 $90 $1.0`)
run(`sbatch batch_poisson.sh $3 $90 $5.0`)
run(`sbatch batch_poisson.sh $4 $90 $1.0`)
run(`sbatch batch_poisson.sh $5 $90 $5.0`)
run(`sbatch batch_poisson.sh $8 $90 $1.0`)
run(`sbatch batch_poisson.sh $9 $90 $1.0`)
