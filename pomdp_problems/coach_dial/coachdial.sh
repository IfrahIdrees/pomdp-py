#!/bin/bash
#$ -cwd
#$ -l inf
##$ -l vf=8G
##$ -l gpus=1
##$ -now y
#$ -m aes

# Activate venv
# source ../../../env/bin/activate

# Move to /playground/scripts
# cd ../../../pomdp-py/pomdp-problems/coach_dial

# Get parameters
while getopts a:s:r:p:n: flag
do
    case "${flag}" in
        a) agent_type=${OPTARG};;
        s) nsteps=${OPTARG};;
        r) give_next_instr_reward=${OPTARG};;
        p) give_next_instr_penalty=${OPTARG};;
        n) num_runs=${OPTARG};;
    esac
done

# Run experiment
python coach_dial_problem.py --print_log \
    --output_results \
    --agent_type ${agent_type} \
    --nsteps ${nsteps} \
    --give_next_instr_reward ${give_next_instr_reward} \
    --give_next_instr_penalty ${give_next_instr_penalty} \
    --num_runs ${num_runs} \
    --num_sims 500 \
    --belief "groundtruth" \

