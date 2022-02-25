import re
import ast
import os
import sys
import numpy as np
import json
import pandas as pd
# from script_helper import *
import matplotlib.pyplot as plt
from os.path import exists
from scipy.stats import rankdata
import argparse
from main import parseArguments

##parameters:
"--agent_type", "standard",
"--maxsteps", "10",
"--num_sims", "5",
"--d", "0.95",
"--e", "0",
"--gr", "10",
"--wp", "-1",
"--qr", "5",
"--qp", "-5",

# def parseArguments():
#     parser = argparse.ArgumentParser()

#     # Necessary variables
#     #parser.add_argument("is_random_agent", action="store_true")
#     #parser.add_argument("is_heuristic_agent", action="store_true")
#     # parser.add_argument("--belief", type=str, default="uniform")
#     parser.add_argument("--agent_type", type=str, default="standard",
#                         help="standard, random, heuristic")
#     parser.add_argument("--maxsteps", type=int, default=10, help="number of max steps")
#     parser.add_argument("--num_sims", type=int, default=10,
#                         help="num_sims for POMCP")
#     parser.add_argument("--d", type=float, default=0.95,
#                         help="discount factor")               
#     parser.add_argument("--e", type=int, default=0,
#                         help="discount factor") 
#     parser.add_argument("--gr", type=int, default=10, help="goal reward")
#     parser.add_argument("--wp", type=int, default=-1, help="wait penalty")
#     parser.add_argument("--qr", type=int, default=5, help="question reward")
#     parser.add_argument("--qp", type=int, default=-5, help="question penalty")
#     # parser.add_argument("--give_next_instr_penalty", type=int, default=-10)
#     # parser.add_argument("--num_runs", type=int, default=1)
#     # parser.add_argument("--print_log", action="store_true")
#     # parser.add_argument("--output_results", action="store_true")
#     # parser.add_argument("--simulation_table_path", type=str, default="simulation_table.csv")
#     args = parser.parse_args()

#     # I/O parameters
#     output_name, output_dir, log_dir = get_output_path(args)
#     parser.add_argument("--output_name", type=str, default=output_name)
#     parser.add_argument("--output_dir", type=str, default=output_dir)
#     parser.add_argument("--log_dir", type=str, default=log_dir)
#     # parser.add_argument("-sr", type=float, default = 0.99, help="sensor reliability")
#     args = parser.parse_args()

#     return parser, args

# def get_output_path(args):
#     # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     # ROOT_DIR = os.path.dirname(BASE_DIR)
#     output_name = "md{}_S{}_df{}_e{}_gr{}_wp{}_qr{}_qp{}".format(
#             # args.num_runs,
#             # args.belief,
#             args.maxsteps, 
#             args.num_sims,
#             args.d,
#             args.e,
#             args.gr,
#             args.wp,
#             args.qr,
#             args.qp
#             # args.give_next_instr_reward, 
#             # args.give_next_instr_penalty,
#     )
#     output_dir = "../outputs/{}/{}".format(args.agent_type, output_name)
#     log_dir = "../logs/{}/{}".format(args.agent_type, output_name)
#     os.makedirs("../outputs", exist_ok=True)
#     os.makedirs("../outputs/{}".format(args.agent_type), exist_ok=True)

#     os.makedirs("../logs/", exist_ok=True)
#     os.makedirs("../logs/{}".format(args.agent_type), exist_ok=True)
#     os.makedirs("../logs/{}/{}/mcts".format(args.agent_type,output_name), exist_ok=True)
#     os.makedirs("../logs/{}/{}/reward".format(args.agent_type,output_name), exist_ok=True)
#     # os.makedirs("../logs/{}/{}real".format(args.agent_type), exist_ok=True)

#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(log_dir, exist_ok=True)
#     return output_name, output_dir, log_dir

def get_output_path(args, num_sim):
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # ROOT_DIR = os.path.dirname(BASE_DIR)
    output_name = "md{}_S{}_df{}_e{}_gr{}_wp{}_qr{}_qp{}".format(
            # args.num_runs,
            # args.belief,
            args.maxsteps, 
            num_sim,
            args.d,
            args.e,
            args.gr,
            args.wp,
            args.qr,
            args.qp
            # args.give_next_instr_reward, 
            # args.give_next_instr_penalty,
    )
    output_dir = "../outputs/{}/{}".format(args.agent_type, output_name)
    # log_dir = "../logs/{}/{}".format(args.agent_type, output_name)
    # os.makedirs("../outputs", exist_ok=True)
    # os.makedirs("../outputs/{}".format(args.agent_type), exist_ok=True)

    # os.makedirs("../logs/", exist_ok=True)
    # os.makedirs("../logs/{}".format(args.agent_type), exist_ok=True)
    # os.makedirs("../logs/{}/{}/mcts".format(args.agent_type,output_name), exist_ok=True)
    # os.makedirs("../logs/{}/{}/reward".format(args.agent_type,output_name), exist_ok=True)
    # os.makedirs("../logs/{}/{}real".format(args.agent_type), exist_ok=True)

    # os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(log_dir, exist_ok=True)
    return output_name, output_dir

if __name__ == "__main__":
    parser, args = parseArguments()

    ##combine the csv
    num_sims = [2,10,20] 
    
    for file_num in range(7,8):
        sensor_reliability = [0.6]
        for sr in sensor_reliability:
            cumulative_reward_df = pd.DataFrame(columns = ['Num_Sims',"cumu_reward", "cumu_discounted_reward"])
            for sim in num_sims:
                input_name, input_dir = get_output_path(args, sim)
                df = pd.read_csv("{}/Case{}_{}.txt_overall_stats.csv".format(input_dir,file_num, sr))
                cumulative_reward_df  = pd.concat([cumulative_reward_df, df])
            # base_dir = os.path.dirname(os.path(args.output_dir))
            root_dir = os.path.dirname(args.output_dir)+"/rewardvssim_overall_stats"
            os.makedirs(root_dir, exist_ok=True)
            cumulative_reward_df.to_csv("{}/Case{}_{}_md{}_df{}_e{}_gr{}_wp{}_qr{}_qp{}.csv".format(
                root_dir, 
                file_num, 
                sr,
                args.maxsteps, 
                args.d,
                args.e,
                args.gr,
                args.wp,
                args.qr,
                args.qp), index = False)