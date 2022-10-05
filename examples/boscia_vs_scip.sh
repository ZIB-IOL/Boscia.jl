#!/usr/bin/env zsh
for SEED in {1..3}
do
    julia --project examples/boscia_vs_scip.jl "$(SEED) 10 2"
    # for DIMENSION [in 20 30 40 50]
    # do 
    #     julia --project examples/boscia_vs_scip.jl SEED DIMENSION 2
    # done 
done

#julia --project examples/boscia_vs_scip.jl 5 5 2