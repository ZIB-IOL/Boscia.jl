mode = "integer"

for example in ["neos5", "ran14x18-disj-8", "pg5_34", "neos5", "22433"]
    for num_v in 4:8
        for seed in 1:3
            run(`sbatch batch_miplib.sh $example $num_v $seed`)
        end
    end
end
