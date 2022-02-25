for agent_type in "standard"
#for agent_type in "random" "heuristic"
#for agent_type in "random"
do
    for maxsteps in 10 #50
    do
        # for give_next_instr_penalty in -5 -10 -15 -20 -30
        for e in 0 10 15 30 50 100
        do
            ./viz.sh -a ${agent_type} \
                -m ${maxsteps} \
                -n 1 \
                -d 0.95 \
                -e ${e} \
                -g 10 \
                -w -1 \
                -r 5 \
                -p -5  
        done

    done
done