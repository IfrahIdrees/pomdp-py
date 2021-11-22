#!/bin/bash

for agent_type in "standard"
#for agent_type in "random" "heuristic"
#for agent_type in "random"
do
    for give_next_instr_reward in 50
    do
        #for give_next_instr_penalty in -5 -10 -15 -20 -30
        for give_next_instr_penalty in -25
        do
            ./coachdial.sh -a ${agent_type} \
                -s 100 \
                -r ${give_next_instr_reward} \
                -p ${give_next_instr_penalty} \
                -n 1
        done
    done
done
