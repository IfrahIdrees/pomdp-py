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
# from main import parseArguments

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

def get_output_path(args):
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # ROOT_DIR = os.path.dirname(BASE_DIR)
    output_name = "md{}_S{}_df{}_e{}_gr{}_wp{}_qr{}_qp{}".format(
            # args.num_runs,
            # args.belief,
            args.maxsteps, 
            args.num_sims,
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

def read_csv(args):
    input_name, input_dir = get_output_path(args)
    file_name =  "{}/Case{}_{}.txt_overall_stats.csv".format(input_dir,Case_Num, sr)
    if exists(file_name):
        df = pd.read_csv(file_name)  
    else:
        df = pd. DataFrame() 
    # df["Num_Sims"] = sr 
    return df

def parseArguments():
    parser = argparse.ArgumentParser()

    # Necessary variables
    #parser.add_argument("is_random_agent", action="store_true")
    #parser.add_argument("is_heuristic_agent", action="store_true")
    # parser.add_argument("--belief", type=str, default="uniform")
    parser.add_argument("--agent_type",  nargs="+", default=["standard","htn_baseline","fixed_always_ask", "random"],
                        help="standard, random, heuristic")
    parser.add_argument("--maxsteps", type=int, default=10, help="number of max steps")
    parser.add_argument("--num_sims", type=int, default=10,
                        help="num_sims for POMCP")
    parser.add_argument("--d", type=float, default=0.95,
                        help="discount factor")               
    parser.add_argument("--e", type=int, default=0,
                        help="discount factor") 
    parser.add_argument("--gr", type=int, default=10, help="goal reward")
    parser.add_argument("--wp", type=int, default=-1, help="wait penalty")
    parser.add_argument("--qr", type=int, default=5, help="question reward")
    parser.add_argument("--qp", type=int, default=-5, help="question penalty")
    # parser.add_argument("--give_next_instr_penalty", type=int, default=-10)
    # parser.add_argument("--num_runs", type=int, default=1)
    # parser.add_argument("--print_log", action="store_true")
    # parser.add_argument("--output_results", action="store_true")
    # parser.add_argument("--simulation_table_path", type=str, default="simulation_table.csv")
    args = parser.parse_args()
    # args = parser.parse_args()
    return parser, args


if __name__ == "__main__":
    parser, args = parseArguments()
    # args.agent_type = args.agent_type[0].split(" ")
    RANDOM =  "random" in args.agent_type
    BASELINE = "htn_baseline" in args.agent_type
    FIXED = "fixed_always_ask" in args.agent_type
    ##combine the csv
    # num_sims = [2,10,20] 
    # num_sims = [10] 
    # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/pomdp-py/pomdp_problems/HTN_CoachDial"
    # LOG_DIR = BASE_DIR + "/logs"
    # OUTPUT_DIR = BASE_DIR + "/output"
    results_storage_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/outputs/reward_viz"
    os.makedirs(results_storage_path, exist_ok=True)
    for Case_Num in range(1,11):
        sensor_reliability = [0.99, 0.9, 0.95, 0.8, 0.7, 0.6]
        if Case_Num == 4:
            continue
        columns = ['CaseNum','SensorReliability',"cumu_reward", "cumu_discounted_reward"]
        standard_cumulative_reward_df = pd.DataFrame(columns = columns )
        if BASELINE:
            baseline_cumulative_reward_df = pd.DataFrame(columns = columns )
        if RANDOM:
            random_cumulative_reward_df = pd.DataFrame(columns = columns )
        if FIXED:
            fixed_cumulative_reward_df = pd.DataFrame(columns =columns )
        overall_cumulative_reward_df = pd.DataFrame(columns = columns )

        for sr in sensor_reliability:
            # sim = num_sims[0]
            # for sim in num_sims:
            print("Case_Num", Case_Num, "sensor_reliability", sr)
            if BASELINE:
                args.agent_type = "htn_baseline"
                input_name, input_dir = get_output_path(args)
                df = read_csv(args)
                df["SensorReliability"] = sr
                # df = 
                df = df.mean().to_frame().transpose()
                # df =df.transpose()
                # df.insert(0, "CaseNum", [21, 23, 24, 21], True)
                df["CaseNum"] =  Case_Num
                df["Method"] =  "HTN"
                baseline_cumulative_reward_df  = pd.concat([baseline_cumulative_reward_df, df])
                overall_cumulative_reward_df = pd.concat([overall_cumulative_reward_df, df])

                
            if FIXED:
                args.agent_type = "fixed_always_ask"
                input_name, input_dir = get_output_path(args)
                df = read_csv(args)
                df["SensorReliability"] = sr
                # df = df.drop("Num_Sims")
                df = df.mean().to_frame().transpose()
                df["CaseNum"] =  Case_Num
                df["Method"] =  "Fixed"
                
                fixed_cumulative_reward_df  = pd.concat([fixed_cumulative_reward_df, df])
                overall_cumulative_reward_df = pd.concat([overall_cumulative_reward_df, df])

                # fixed_cumulative_reward_df.drop("Num_Sims")
            if RANDOM:
                args.agent_type = "fixed_always_ask"
                input_name, input_dir = get_output_path(args)
                df = read_csv(args)
                df["SensorReliability"] = sr
                # df = df.drop("Num_Sims")
                df = df.mean().to_frame().transpose()
                df["CaseNum"] =  Case_Num
                df["Method"] =  "RANDOM"


                random_cumulative_reward_df  = pd.concat([random_cumulative_reward_df, df])
                overall_cumulative_reward_df = pd.concat([overall_cumulative_reward_df, df])

                
            args.agent_type = "standard"
            input_name, input_dir = get_output_path(args)
            # if 
            df = read_csv(args)
            df["SensorReliability"] = sr
            df = df.mean().to_frame().transpose()
            df["CaseNum"] =  Case_Num
            df["Method"] =  "Standard"

            # df.drop("Num_Sims")
            standard_cumulative_reward_df  = pd.concat([standard_cumulative_reward_df, df])
            overall_cumulative_reward_df = pd.concat([overall_cumulative_reward_df, df])

            
            # base_dir = os.path.dirname(os.path(args.output_dir))
        if exists(results_storage_path+'/reward.csv'):
            df = pd.read_csv(results_storage_path+'/reward.csv')
            overall_cumulative_reward_df = pd.concat([df, overall_cumulative_reward_df,])
        overall_cumulative_reward_df.to_csv(results_storage_path+'/reward.csv', index = False)  
        plt.clf()

        ##plotting not working!!
        if FIXED:
            fixed_cumulative_reward_df = fixed_cumulative_reward_df.drop("Num_Sims", axis=1)
            # fixed_cumulative_reward_df.plot(title = "Reward - Case:"+str(Case_Num),label = "Fixed HTNDialPOMDP")
            fixed_cumulative_reward_df.plot(x= 'SensorReliability', y='cumu_discounted_reward', title = "Reward - Case:"+str(Case_Num),label = "Fixed HTNDialPOMDP")

        if RANDOM:
            random_cumulative_reward_df = random_cumulative_reward_df.drop("Num_Sims", axis=1)
            random_cumulative_reward_df.plot(title = "Reward - Case:"+str(Case_Num),label = "Random HTNDialPOMDP")
            
        if BASELINE:
            baseline_cumulative_reward_df = baseline_cumulative_reward_df.drop("Num_Sims", axis=1)
            baseline_cumulative_reward_df.plot(x= 'SensorReliability', y='cumu_discounted_reward', title = "Reward - Case:"+str(Case_Num),label = "Random HTNDialPOMDP")
        
        # root_dir = os.path.dirname(args.output_dir)+"/rewardvsn_overall_stats"
        # os.makedirs(root_dir, exist_ok=True)
        # cumulative_reward_df.to_csv("{}/Case{}_{}_md{}_df{}_e{}_gr{}_wp{}_qr{}_qp{}.csv".format(
        #     root_dir, 
        #     file_num, 
        #     sr,
        #     args.maxsteps, 
        #     args.d,
        #     args.e,
        #     args.gr,
        #     args.wp,
        #     args.qr,
        #     args.qp), index = False)
        standard_cumulative_reward_df = standard_cumulative_reward_df.drop("Num_Sims", axis=1)
        ax = standard_cumulative_reward_df.plot(x= 'SensorReliability', y='cumu_discounted_reward',title = "Reward - Case:"+str(Case_Num),label= "standard-HTNDialPOMDP")
        ax.set_ylabel('Reward')
        ax.set_xlabel('Sensor Reliabilities')
        plt.legend()
        plt.savefig(results_storage_path + "/Reward - "+"Case:"+str(Case_Num)+".png")
        plt.clf()
        plt.close()