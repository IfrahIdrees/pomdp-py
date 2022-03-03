"""------------------------------------------------------------------------------------------
Hierarchical Task Recognition and Planning in Smart Homes with Partially Observability
Author: Dan Wang danwangkoala@gmail.com (May 2016 - June 2017)
Supervised by Prof. Jesse Hoey (https://cs.uwaterloo.ca/~jhoey/)
Association: Computer Science, University of Waterloo.
Research purposes only. Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by AGEWELL Networks of Centers of Excellence (NCE).
----------------------------------------------------------------------------------------------"""
# %%
import os
import sys
sys.dont_write_bytecode = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR+"/database")

from tracking_engine import *
from pymongo import MongoClient
from os.path import exists
import config
import logging

import random
random.seed(10)
import numpy as np
np.random.seed(10)

# sys.path.append(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))

# client = MongoClient()
# db = client.smart_home3

'''orignal pomdp-htn database is smart_home3'''
if config.RANDOM_BASELINE:
    client = MongoClient()
    db = client.smart_homeRANDOM
else:
    client = MongoClient()
    db = client.smart_home5
    db_client = db


'''logging information'''

def parseArguments():
    parser = argparse.ArgumentParser()

    # Necessary variables
    #parser.add_argument("is_random_agent", action="store_true")
    #parser.add_argument("is_heuristic_agent", action="store_true")
    # parser.add_argument("--belief", type=str, default="uniform")
    parser.add_argument("--agent_type", type=str, default="standard",
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

    # I/O parameters
    output_name, output_dir, log_dir = get_output_path(args)
    parser.add_argument("--output_name", type=str, default=output_name)
    parser.add_argument("--output_dir", type=str, default=output_dir)
    parser.add_argument("--log_dir", type=str, default=log_dir)
    # parser.add_argument("-sr", type=float, default = 0.99, help="sensor reliability")
    args = parser.parse_args()

    return parser, args


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
    log_dir = "../logs/{}/{}".format(args.agent_type, output_name)
    os.makedirs("../outputs", exist_ok=True)
    os.makedirs("../outputs/{}".format(args.agent_type), exist_ok=True)

    os.makedirs("../logs/", exist_ok=True)
    os.makedirs("../logs/{}".format(args.agent_type), exist_ok=True)
    os.makedirs("../logs/{}/{}/mcts".format(args.agent_type,output_name), exist_ok=True)
    os.makedirs("../logs/{}/{}/reward".format(args.agent_type,output_name), exist_ok=True)
    # os.makedirs("../logs/{}/{}real".format(args.agent_type), exist_ok=True)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return output_name, output_dir, log_dir

# def get_logger(args):
#     os.makedirs("../logs/", exist_ok=True)
#     # os.makedirs("../logs/demo", exist_ok=True)
#     os.makedirs("../logs/{}".format(args.agent_type), exist_ok=True)
#     # os.makedirs("../../../logs/{}/{}".format(args.agent_type, args.output_name), exist_ok=True)
#     logging.basicConfig(level = logging.DEBUG, \
#             format = '%(asctime)s %(levelname)s: %(message)s', \
#             datefmt = '%m/%d %H:%M:%S %p', \
#             filename = '../logs/{}/{}/{}.log'.format(
#                 args.agent_type, args.output_name, args.output_name
#             ), \
#             filemode = 'w'
#     )
#     return logging.getLogger()

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

if __name__ == '__main__':
    
    ############                global variables                ######################
    #######################################################
    #if there is no notification, the engine still should run the whole update process if the generated random is bigger than no_notif_trigger_prob
    no_notif_trigger_prob = 0.01
 
    #sleep interval  
    interval = 1
    
    #conditional probability of p(s|s_t-1, a_t)
    # cond_satisfy = 1.0
    # cond_notsatisfy = 0.0
    cond_satisfy = 0.999
    cond_notsatisfy = 0.001
    
    #threshhold that an explanation is no longer maintain
    delete_trigger = 0.001
    
    ##if there is a notification, the probability that nothing happend
    nothing_happen = 0.01
    
    ##the otherHappen triggering threshhold
    #orignal other_happen = 0.75
    # other_happen = 0.30
    other_happen = 0.75
    
    ##sensor set up files

    sensor_reliability = [0.99,0.95, 0.9, 0.8, 0.7, 0.6]
    # sensor_reliability = [0.9, 0.8, 0.7, 0.6]
    # sensor_reliability = [0.99, 0.9, 0.6]
    # sensor_reliability = [0.7, 0.6]
    # sensor_reliability = [0.6]
    # sensor_reliability = [0.95, 0.99, 0.6]
    sensor_reliability = [0.99, 0.95]
    # sensor_reliability = [0.99,0.6]
    # sensor_reliability = [0.6]
    # sensor_reliability = [1]
    # sensor_reliability = [0.8]
    # sensor_reliability = [0.5, 0.8]
    # sensor_reliability = [0.8]

    #sensor_reliability = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    #6,10
    #nohup running 6,7
    parser, args = parseArguments()
    trials = 21
    for file_num in range(1,2): #7
        # if file_num == 9:
        #     sensor_reliability = [0.99]
        for x in sensor_reliability:
            ##output file name
            # if config.RANDOM_BASELINE:
            #     output_file_name = "Random_Case" + str(file_num) + "_" + str(x) + ".txt"
            # else:
            #     output_file_name = "Case" + str(file_num) + "_" + str(x) + ".txt"
            output_file_name = "Case" + str(file_num) + "_" + str(x) + ".txt"
            mcts_output_filename = "mctsCase"+ str(file_num) + "_" + str(x) + ".txt"
            ##input file name
            input_file_name = "../../../../TestCases/Case" + str(file_num)
            

            cum_rew_file_name =  "{}/{}_overall_stats.csv".format(args.output_dir, output_file_name)
            if os.path.exists(cum_rew_file_name):
                cumulative_reward_df = pd.read_csv(cum_rew_file_name)
            else:
                cumulative_reward_df = pd.DataFrame(columns = ['Num_Sims',"cumu_reward", "cumu_discounted_reward"])
            
            if not exists(input_file_name):
                continue
            
            ##each test case run 20 times range(1,21)
            print("changing iterations")
            total_reward, total_discounted_reward = 0, 0
            for repeat in range(1,trials):
                print("sensor_reliability:",sensor_reliability, "repeating trial number", repeat, x)
                db.method.drop()
                db.state.drop()
                db.operator.drop()
                db.sensor.drop()
                db.Rstate.drop()
                db.backup_state.drop()
                db.backup_sensor.drop()
                sensor_command = ""

                ##add config
                if config.RANDOM_BASELINE:
                    os.system("mongoimport --db smart_homeRANDOM --collection method --drop --file ../KnowledgeBase/method.json")
                    os.system("mongoimport --db smart_homeRANDOM --collection state --drop --file ../KnowledgeBase/state.json")
                    os.system("mongoimport --db smart_homeRANDOM --collection operator --drop --file ../KnowledgeBase/operator.json")
                    os.system("mongoimport --db smart_homeRANDOM --collection Rstate --drop --file ../KnowledgeBase/realState.json")
                    # db.backup_state.insertOne({});
                else:
                    ##Some times those command do not work, add "--jsonArray" to the end of each command line
                    os.system("mongoimport --db smart_home5 --collection method --drop --file ../../../../KnowledgeBase/method.json")
                    os.system("mongoimport --db smart_home5 --collection state --drop --file ../../../../KnowledgeBase/state.json")
                    os.system("mongoimport --db smart_home5 --collection operator --drop --file ../../../../KnowledgeBase/operator.json")
                    os.system("mongoimport --db smart_home5 --collection Rstate --drop --file ../../../../KnowledgeBase/realState.json")
                    # db.backup_state.insertOne({});
                    
                # ##Some times those command do not work, add "--jsonArray" to the end of each command line
                # os.system("mongoimport --db smart_home3 --collection method --drop --file ../../../../KnowledgeBase/method.json")
                # os.system("mongoimport --db smart_home3 --collection state --drop --file ../../../../KnowledgeBase/state.json")
                # os.system("mongoimport --db smart_home3 --collection operator --drop --file ../../../../KnowledgeBase/operator.json")
                # os.system("mongoimport --db smart_home3 --collection Rstate --drop --file ../../../../KnowledgeBase/realState.json")
                
                ##command for sensor reliability set up
                if config.RANDOM_BASELINE:
                    if x == None:
                        sensor_command = "mongoimport --db smart_homeRANDOM --collection sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor.json"
                        # mcts_sensor_command = "mongoimport --db smart_homeRANDOM --collection mcts_sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor.json"
                    else:
                        sensor_command = "mongoimport --db smart_homeRANDOM --collection sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor" + "_" + str(x) + ".json"
                        # mcts_sensor_command = "mongoimport --db smart_homeRANDOM --collection mcts_sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor" + "_" + str(x) + ".json"
                
                else:
                    if x == None:
                        sensor_command = "mongoimport --db smart_home5 --collection sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor.json"
                        # mcts_sensor_command = "mongoimport --db smart_home3 --collection mcts_sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor.json"
                    else:   
                        sensor_command = "mongoimport --db smart_home5 --collection sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor" + "_" + str(x) + ".json"
                        # mcts_sensor_command = "mongoimport --db smart_home3 --collection mcts_sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor.json"

                # if x == None:
                #     sensor_command = "mongoimport --db smart_home3 --collection sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor.json"
                # else:
                #     sensor_command = "mongoimport --db smart_home3 --collection sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor" + "_" + str(x) + ".json"
                os.system(sensor_command)
                # os.system(mcts_sensor_command)
                
                ##command for sensor missing set up
                '''
                sensor_command = "mongoimport --db smart_home3 --collection sensor --drop --file ../KnowledgeBase/missing_sensor/sensor" + "_" + str(x) + ".json"
                os.system(sensor_command)
                '''
                
                
                # logger = get_logger(args)
                # logger.disabled = True
                # logging_args(args)
                # parser = argparse.ArgumentParser()
                # parser.add_argument("--sensor", type=float, default = x, help="sensor reliability")
                # args = parser.parse_args()
                tracking_engine = Tracking_Engine(no_trigger = no_notif_trigger_prob, sleep_interval = interval, cond_satisfy=cond_satisfy, cond_notsatisfy = cond_notsatisfy, delete_trigger = delete_trigger, otherHappen = other_happen, file_name = input_file_name, output_file_name = output_file_name, mcts_output_filename = mcts_output_filename, args=args, db_client = db_client)
                total_reward_per_iter, total_discounted_reward_per_iter = tracking_engine.start()
                cumulative_reward_df.loc[len(cumulative_reward_df.index)] = ([args.num_sims, total_reward_per_iter, total_discounted_reward_per_iter])

                total_reward += total_reward_per_iter
                total_discounted_reward += total_discounted_reward_per_iter
                print("here")

                # df = pd.DataFrame(
                # {
                #     "nsteps": nsteps_list,
                #     "optimal_count": optimal_count_list,
                #     "optimal_coverage": optimal_count_list/nsteps_list,
                #     "cumu_reward": last_cumu_reward_list,
                #     "cumu_discounted_reward": last_cumu_discounted_reward_list,
                # })

                # accuracy_df.loc[len(accuracy_df.index)] = ([case_num,0,0,0,0,0,0])
            denominator = trials -1
            total_reward=total_reward_per_iter/ denominator
            total_discounted_reward = total_discounted_reward_per_iter / denominator
                
            # cumulative_reward_df.loc[len(cumulative_reward_df.index)] = ([args.num_sims, total_reward, total_discounted_reward])
            cumulative_reward_df.to_csv(cum_rew_file_name, index = False)
            # cumulative_reward_df.iloc[-1, accuracy_df.columns.get_loc(str(reliability ))] = round(accurracy, 4)
            print("I am good until now")
            
else:
    print('I am being imported')    

# %%
