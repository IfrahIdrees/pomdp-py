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


import os
import sys
sys.dont_write_bytecode = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
print(f"THe base directory: {BASE_DIR}")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, ROOT_DIR+"/database")
sys.path.insert(0, os.path.dirname(os.path.dirname(ROOT_DIR)))
# /home/ifrah/DDrive/Research_Projects/CoachDial/HTN-Language-ObservationalModel-pomdpdevelopbranch/pomdp-py

# to_remove = "/home/ifrah/DDrive/Research_Projects/CoachDial/HTN-Language-ObservationalModel-pomdpdevelopbranch"
# new_path = []
# for path in sys.path:
#     if to_remove not in path:
#         new_path.append(path)
#     print("The module search path", path)
# print(new_path)
# sys.path = list(new_path)


from pymongo import MongoClient
from os.path import exists
import config
import logging
import argparse
import csv

# sys.path.append(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))

# client = MongoClient()
# db = client.smart_home3
parser, args = parseArguments()
if args.agent_type == "htn_baseline":
    config.baseline = True

from tracking_engine import *

'''orignal pomdp-htn database is smart_home3'''
if config.baseline:
    client = MongoClient()
    db = client.smart_homebaseline
    db_client = db
else:
    client = MongoClient()
    # db = client.smart_home5 ##used for 10,11
    db = client.smart_homeISRRreview
    db_client = db


'''logging information'''


def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

import random
import numpy as np
random_seed = 5999
random.seed(random_seed) #10, 5999
np.random.seed(random_seed) #10,5999

if __name__ == '__main__':
    
    ############                global variables                ######################
    #######################################################
    #if there is no notification, the engine still should run the whole update process if the generated random is bigger than no_notif_trigger_prob
    no_notif_trigger_prob = 0
 
    #sleep interval  
    interval = 1
    
    #conditional probability of p(s|s_t-1, a_t)
    # cond_satisfy = 1.0
    # cond_notsatisfy = 0.0
    cond_satisfy = 0.999
    cond_notsatisfy = 0.001
    
    # ######## before Friday ##################
    # #threshhold that an explanation is no longer maintain
    # delete_trigger = 0.001
    # # config.mcts_delete_trigger = 0.001
    # config._real_delete_trigger = 0.001 #0.000095 #(0.000095 works with time)
    
    # ##if there is a notification, the probability that nothing happend
    # nothing_happen = 0.01
    
    # ##the otherHappen triggering threshhold
    # #orignal other_happen = 0.75
    # # other_happen = 0.30
    # other_happen = 0.87 #0.85 #0.87
    # # other_happen = 0.85
    
    # ##sensor set up files

    # #################################################

    delete_trigger = 0.001
    # config.mcts_delete_trigger = 0.001
    config._real_delete_trigger = 0.00001 #0.000095 #(0.000095 works with time)
    
    ##if there is a notification, the probability that nothing happend
    nothing_happen = 0.01
    
    ##the otherHappen triggering threshhold
    #orignal other_happen = 0.75
    # other_happen = 0.30
    other_happen = 0.86 #0.87 #(use 0.87 for all the cases between except 7 and 9, for 7 and 9 use 0.86) #0.85 #0.87

    sensor_reliability = [0.99,0.95, 0.9, 0.8, 0.7, 0.6]
    # sensor_reliability = [0.7, 0.6]
    # sensor_reliability = [0.8]
    # sensor_reliability = [0.9, 0.8, 0.7, 0.6]
    # sensor_reliability = [0.95]
    # sensor_reliability = [0.99,0.95]
    # sensor_reliability = [0.8, 0.7, 0.6]
    # sensor_reliability = [0.99, 0.8, 0.6]
    # sensor_reliability = [0.9]
    # sensor_reliability = [0.7 , 0.6]
    # sensor_reliability = [0.6]
    # sensor_reliability = [0.95, 0.99, 0.6]
    # sensor_reliability = [0.8]
    # , 0.95]
    # sensor_reliability = [0.6]
    # sensor_reliability = [1]
    # sensor_reliability = [0.8]
    # sensor_reliability = [0.5, 0.8]
    # sensor_reliability = [0.8]

    #sensor_reliability = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    #6,10
    #nohup running 6,7
    
    # trials = 21
    trials = 11
    config.seed = 5999
    config.trials = trials
    random.seed(config.seed)
    # config.randomNs = [random.random() for i in range((config.trials-1)*100*args.num_sims*20)]
    config.randomNs = [random.random() for i in range((21)*1000*args.num_sims*20)]
    # config.randomNs = [0.7514062100906035, 1.0, 0.9205949710733476, 0.9367211756841427, 0.758452287599932, 0.1473436494008996, 0.597971828480958, 0.3083822735356848, 0.21739233925559065, 0.5533682982520421, 0.6290635522399637, 0.05604972577945, 0.07294319938326999, 0.3332888354512096]
    # for file_num in range(13,8,-1): #7
    file_nums =[6,7,9,11,1,2,5,3,10,12,8]
    file_nums =[7,9,11,1,2,3,5,6,10,12,8]
    file_nums =[7,9,11,1,2] #,3,5,6,10,12,8]
    # file_nums =[9,11,1,2]
    # file_nums =[1,2]
    # file_nums =[1]
    # file_nums =[9, 11]
    # 5,3,10,12,8]
    # file_nums =[2]
    # file_nums =[9,6,11,1,2,5,3,10,12,8]
    # file_nums =[6,11,1,2,5,3,10,12,8],
    # file_nums =[3,5,6,10,12,8]
    # repeating trial number 3 0.8 has the weird issue
    file_nums =[1,2,7,9,11]
    # file_nums =[1,2,3,5,6,8,12]
    file_nums =[6,8,12]
    # file_nums =[1]
    #5 0.9 0.8
    # file_nums =[1,2,7,9,11]
    print("i am going to start the main loop")

    if not exists("config_info.csv"):
        with open('config_info.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
            spamwriter.writerow(["max_depth","num_sims","d","e","file_num", "reliability","trial_num", "config.randomIndex","config.realRandomIndex"])          

    
    for file_num in file_nums:
    #5 0.
    # for file_num in range(1,13):
        if file_num == 4:
            continue
        # if file_num == 9:
        #     sensor_reliability = [0.99]
        # if file_num == 1:
        #     sensor_reliability = [0.99, 0.8, 0.6]
        # if file_num == 2:
        #     sensor_reliability = [0.99, 0.8, 0.6]
        for x in sensor_reliability:
            ##output file name
            # if config.RANDOM_BASELINE:
            #     output_file_name = "Random_Case" + str(file_num) + "_" + str(x) + ".txt"
            # else:
            #     output_file_name = "Case" + str(file_num) + "_" + str(x) + ".txt"
            # random_seed+=10
            # random.seed(random_seed) #10, 5999
            # np.random.seed(random_seed) #10,5999
            
            output_file_name = "Case" + str(file_num) + "_" + str(x) + ".txt"
            mcts_output_filename = "mctsCase"+ str(file_num) + "_" + str(x) + ".txt"
            ##input file name
            input_file_name = "../../../../TestCases/Case" + str(file_num)
            

            cum_rew_file_name =  "{}/{}_overall_stats.csv".format(args.output_dir, output_file_name)
            if os.path.exists(cum_rew_file_name):
                cumulative_reward_df = pd.read_csv(cum_rew_file_name)
            else:
                cumulative_reward_df = pd.DataFrame(columns = ['Num_Sims',"cumu_reward", "cumu_discounted_reward", "num_question_asked", "normalized_num_question_asked", "normalized_time_taken"])
            
            if not exists(input_file_name):
                continue
            
            ##each test case run 20 times range(1,21)
            print("changing iterations")
            total_reward, total_discounted_reward,num_question_asked = 0, 0, 0
            for repeat in range(1,trials):

                with open("random_no.txt", 'a') as f:
                    f.write('\n=========='+str(file_num)+"  "+str(x)+"   "+ str(repeat)+'============\n')


                if repeat == 1:
                    # config.seed = 5999
                    config.randomIndex = 0
                    # config.randomIndex = 48
                    config.realRandomIndex = 0
                    config.randomIndex = 48
                    config.realRandomIndex = 48
                    # config.randomIndex = 300
                    # config.realRandomIndex = 96
                    # config.randomIndex = 1776
                    # config.realRandomIndex = 104
                    # config.randomIndex = 48
                    # config.realRandomIndex = 76
                    # config.randomIndex = 48
                    
                    # config.realRandomIndex = 125 ####(imp for current debug) 
                    # config.randomIndex = 972 ####(imp for current debug) 
                    # config.realRandomIndex = 132
                    # config.randomIndex = 1672
                    # config.realRandomIndex 133
                    # config.randomIndex 1972
                    
                    

                if repeat == 2:
                    print("here")
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
                if config.baseline:
                    os.system("mongoimport --db smart_homebaseline --collection method --drop --file ../../../../KnowledgeBase/method.json")
                    os.system("mongoimport --db smart_homebaseline --collection state --drop --file ../../../../KnowledgeBase/state.json")
                    os.system("mongoimport --db smart_homebaseline --collection operator --drop --file ../../../../KnowledgeBase/operator.json")
                    os.system("mongoimport --db smart_homebaseline --collection Rstate --drop --file ../../../../KnowledgeBase/realState.json")
                    # db.backup_state.insertOne({});
                else:
                    ##Some times those command do not work, add "--jsonArray" to the end of each command line
                    os.system("mongoimport --db smart_homeISRRreview --collection method --drop --file ../../../../KnowledgeBase/method.json")
                    os.system("mongoimport --db smart_homeISRRreview --collection state --drop --file ../../../../KnowledgeBase/state.json")
                    os.system("mongoimport --db smart_homeISRRreview --collection operator --drop --file ../../../../KnowledgeBase/operator.json")
                    os.system("mongoimport --db smart_homeISRRreview --collection Rstate --drop --file ../../../../KnowledgeBase/realState.json")
                    # db.backup_state.insertOne({});
                    
                # ##Some times those command do not work, add "--jsonArray" to the end of each command line
                # os.system("mongoimport --db smart_home3 --collection method --drop --file ../../../../KnowledgeBase/method.json")
                # os.system("mongoimport --db smart_home3 --collection state --drop --file ../../../../KnowledgeBase/state.json")
                # os.system("mongoimport --db smart_home3 --collection operator --drop --file ../../../../KnowledgeBase/operator.json")
                # os.system("mongoimport --db smart_home3 --collection Rstate --drop --file ../../../../KnowledgeBase/realState.json")
                
                ##command for sensor reliability set up
                if config.baseline:
                    if x == None:
                        sensor_command = "mongoimport --db smart_homebaseline --collection sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor.json"
                        # mcts_sensor_command = "mongoimport --db smart_homebaseline --collection mcts_sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor.json"
                    else:
                        sensor_command = "mongoimport --db smart_homebaseline --collection sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor" + "_" + str(x) + ".json"
                        # mcts_sensor_command = "mongoimport --db smart_homebaseline --collection mcts_sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor" + "_" + str(x) + ".json"
                
                else:
                    if x == None:
                        sensor_command = "mongoimport --db smart_homeISRRreview --collection sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor.json"
                        # mcts_sensor_command = "mongoimport --db smart_home3 --collection mcts_sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor.json"
                    else:   
                        sensor_command = "mongoimport --db smart_homeISRRreview --collection sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor" + "_" + str(x) + ".json"
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
                total_reward_per_iter, total_discounted_reward_per_iter, num_question_asked_per_iter, test_case_length,total_time_per_iter = tracking_engine.start()
                normalized_question_asked = num_question_asked_per_iter/test_case_length
                normalized_time = total_time_per_iter/test_case_length
                
                cumulative_reward_df.loc[len(cumulative_reward_df.index)] = ([args.num_sims, total_reward_per_iter, total_discounted_reward_per_iter,num_question_asked_per_iter,normalized_question_asked, normalized_time])

                total_reward += total_reward_per_iter
                total_discounted_reward += total_discounted_reward_per_iter
                num_question_asked+= num_question_asked_per_iter
                print("out of step here")

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
            num_question_asked = total_discounted_reward_per_iter / denominator
                
            # cumulative_reward_df.loc[len(cumulative_reward_df.index)] = ([args.num_sims, total_reward, total_discounted_reward])
            cumulative_reward_df.to_csv(cum_rew_file_name, index = False)
            # cumulative_reward_df.iloc[-1, accuracy_df.columns.get_loc(str(reliability ))] = round(accurracy, 4)
            print("I am good until now")

        # with open("config_info.txt", 'a') as f:
        #     f.write('\n=========='+str(file_num)+"  "+str(x)+"   "+ str(repeat)+'============\n')
        #     f.write('config.randomIndex: '+str(config.randomIndex)+ 'config.realRandomIndex: '+str(config.realRandomIndex)) 
        # import csv
            with open('config_info.csv', 'a', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                # spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
                spamwriter.writerow([args.maxsteps, args.num_sims, args.d,args.e,file_num, str(x),str(repeat),config.randomIndex,config.realRandomIndex])           
else:
    print('I am being imported')    

# %%
