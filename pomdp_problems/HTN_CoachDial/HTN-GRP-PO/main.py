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
from tracking_engine import*
from pymongo import MongoClient
from os.path import exists
import config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
# sys.path.append(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))

# client = MongoClient()
# db = client.smart_home3
if config.RANDOM_BASELINE:
    client = MongoClient()
    db = client.smart_homeRANDOM
else:
    client = MongoClient()
    db = client.smart_home3

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
    # sensor_reliability = [0.7, 0.6]
    sensor_reliability = [0.99]
    # sensor_reliability = [0.8]
    # sensor_reliability = [0.5, 0.8]
    # sensor_reliability = [0.8]

    #sensor_reliability = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    #6,10
    #nohup running 6,7
    for file_num in range(7,8):
        # if file_num == 9:
        #     sensor_reliability = [0.99]
        for x in sensor_reliability:
            ##output file name
            if config.RANDOM_BASELINE:
                output_file_name = "Random_Case" + str(file_num) + "_" + str(x) + ".txt"
            else:
                output_file_name = "Case" + str(file_num) + "_" + str(x) + ".txt"
            mcts_output_filename = "MCTS"+ str(file_num) + "_" + str(x) + ".txt"
            ##input file name
            input_file_name = "../../../../TestCases/Case" + str(file_num)

            if not exists(input_file_name):
                continue
            
            ##each test case run 20 times range(1,21)
            print("changeing iterations")
            for repeat in range(1,2):
                print("repeating step number", repeat, x)
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
                    os.system("mongoimport --db smart_home3 --collection method --drop --file ../../../../KnowledgeBase/method.json")
                    os.system("mongoimport --db smart_home3 --collection state --drop --file ../../../../KnowledgeBase/state.json")
                    os.system("mongoimport --db smart_home3 --collection operator --drop --file ../../../../KnowledgeBase/operator.json")
                    os.system("mongoimport --db smart_home3 --collection Rstate --drop --file ../../../../KnowledgeBase/realState.json")
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
                        sensor_command = "mongoimport --db smart_home3 --collection sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor.json"
                        # mcts_sensor_command = "mongoimport --db smart_home3 --collection mcts_sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor.json"
                    else:   
                        sensor_command = "mongoimport --db smart_home3 --collection sensor --drop --file ../../../../KnowledgeBase/sensor_reliability/sensor" + "_" + str(x) + ".json"
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
                
                with open(output_file_name, 'a') as f:
                    f.write('\n========================\n')
  
                tracking_engine = Tracking_Engine(no_trigger = no_notif_trigger_prob, sleep_interval = interval, cond_satisfy=cond_satisfy, cond_notsatisfy = cond_notsatisfy, delete_trigger = delete_trigger, otherHappen = other_happen, file_name = input_file_name, output_file_name = output_file_name, mcts_output_filename = mcts_output_filename)
                tracking_engine.start()
                print("here")
            
            print("I am good until now")
            
else:
    print('I am being imported')    

# %%
