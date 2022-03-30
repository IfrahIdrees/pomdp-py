#!/bin/bash

# for agent_type in "standard"
#for agent_type in "random" "heuristic"
# "standard"
for agent_type in "fixed_always_ask" "standard"

do
    for maxsteps in 10 #50
    do
        # for give_next_instr_penalty in -5 -10 -15 -20 -30
        for num_sims in 10 #2 50 100 500 #20 #-25
        do
            for e in 10 #15
            do
                ./coachdial.sh -a ${agent_type} \
                    -m ${maxsteps} \
                    -n ${num_sims} \
                    -d 0.95 \
                    -e ${e} \
                    -g 10 \
                    -w -1 \
                    -r 5 \
                    -p -5  
            done
        done
    done
done