"""------------------------------------------------------------------------------------------
Hierarchical Task Recognition and Planning in Smart Homes with Partially Observability
Author: Dan Wang danwangkoala@gmail.com (May 2016 - June 2017)
Supervised by Prof. Jesse Hoey (https://cs.uwaterloo.ca/~jhoey/)
Association: Computer Science, University of Waterloo.
Research purposes only. Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by AGEWELL Networks of Centers of Excellence (NCE).
----------------------------------------------------------------------------------------------"""
################################################################################################
####                        The control of an algorithm iteration                           ####
################################################################################################

import random
import time
from notification import *
from ExplaSet import *
from State import *
from Simulator import *
import random
import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
from HTN_CoachDial import *

# random.seed(10)

class Tracking_Engine(object):
    def __init__(self, no_trigger = 0, sleep_interval = 1, cond_satisfy=1.0, cond_notsatisfy = 0.0, delete_trigger = 0.001, non_happen = 0.00001, otherHappen = 0.75, file_name = "Case1", output_file_name = "Case1.txt"):
        self._no_trigger = no_trigger
        self._sleep_interval = sleep_interval
        self._cond_satisfy = cond_satisfy
        self._cond_notsatisfy = cond_notsatisfy
        self._delete_trigger = delete_trigger
        self._non_happen = non_happen
        self._other_happen = otherHappen
        self._file_name = file_name
        self._output_file_name = output_file_name

            
    def start(self):
        print
        print("the engine has been started...")
        print
        
        notif = notification(self._file_name)   ##check the current notification
        exp = explaSet(cond_satisfy = self._cond_satisfy, cond_notsatisfy = self._cond_notsatisfy, delete_trigger = self._delete_trigger, non_happen = self._non_happen, output_file_name = self._output_file_name)
        exp.explaInitialize()  
        
        #always iterate
        # language_notifcation_list = ["No, you are wrong", "No, you are wrong", "No, you are wrong", "No, you are wrong"]
        language_notifcation_list = ["Yes, you are right" for i in range(16)]
        # language_notifcation_list.append("No, you are wrong")
        # language_notifcation_list.append("No, you are wrong")
        # # language_notifcation_list = ["Yes, you are right" for i in range(10)]
        # language_notifcation_list.extend(["No, you are wrong" for i in range(10)])

        # , "No, you are wrong", "No, you are wrong", "No, you are wrong"]
        index=0
        last_sensor_notification = ""
        while(notif._notif.qsize()>0):
            
            step = notif.get_one_notif()
            notif.delete_one_notif()
            
            #if no notification, and the random prob is less than no_notif_trigger_prob, sleep the engine
            if step == "none" and random.random()<self._no_trigger:
                time.sleep(self._sleep_interval)
                
            #go through the engine logic
            else:

                if index != 0 and  (index%2) == 1:
                    # ask for language
                    # otherhappen = 0.99
                    # language_notifcation = language_notifcation_list[index %len(language_notifcation_list)].lower().replace(",", "")
                    with open(self._output_file_name, 'a') as f:
                        #version changed in March 14, generate a table
                        f.write("Notif: "+last_sensor_notification + "\t" + str(1) + "\t")
                    exp.setLanguageNotification(step)
                    # exp._prior = exp._action_posterior_prob

                    # exp.update_prior()
                    print(exp._action_posterior_prob)
                    #update the prior to be posterior of the previous iteration

                    highest_action_PS = ["", float('-inf')]
                    for k, v in exp._action_posterior_prob.items():
                        if v > highest_action_PS[1]:
                            highest_action_PS = [k,v]
                    exp.highest_action_PS = highest_action_PS
                    # # plang_st = 0
                    # for k in self._action_posterior_prob: 
                    #     posteriorK = self.cal_posterior(k)
                    print("simulate is", last_sensor_notification )
                    print("DId you just complete", exp.highest_action_PS)

                    if last_sensor_notification == exp.highest_action_PS[0]:
                        step = "Yes"
                    else:
                        step = "No"
                    print("step is", step)
                    # step = raw_input("DId you just complete"+str(exp.highest_action_PS))
                    # time.sleep(3)
                    print("other happen is", exp.otherHappen)
                    # if (step == "No"):
                    #     exp.adjust_posterior()
                    # if (exp.otherHappen > self._other_happen):
                    #     exp.handle_exception()
                    # if (step == "No" or exp.otherHappen > self._other_happen) and exp._explaset:
                    #     exp.adjust_posterior()
                    # elif (step == "No" or exp.otherHappen > self._other_happen):
                    #     exp = copy.copy(last_exp)
                    # if step == "No" or exp.otherHappen > self._other_happen:
                        # exp.adjust_posterior()
                        # exp.handle_exception()

                    exp.update_language_feedback(step, exp.highest_action_PS)
                    # state = State()
                    # state.update_state_belief(exp)
                    # elif exp.otherHappen > self._other_happen:
                        # exp.handle_exception()

                    init_true_state = random.choice([TigerState("tiger-left"),
                                     TigerState("tiger-right")])
                    init_belief = pomdp_py.Histogram({TigerState("tiger-left"): 0.5,
                                                    TigerState("tiger-right"): 0.5})
                    tiger_problem = TigerProblem(0.15,  # observation noise
                                                init_true_state, init_belief)

                    print("\n** Testing POUCT **")
                    pouct = pomdp_py.POUCT(max_depth=3, discount_factor=0.95,
                                        num_sims=4096, exploration_const=50,
                                        rollout_policy=tiger_problem.agent.policy_model,
                                        show_progress=True)
                    test_planner(tiger_problem, pouct, nsteps=1, debug_tree=False)
                    TreeDebugger(tiger_problem.agent.tree).pp

                else: 
                    last_sensor_notification = step
                    if step != "none":
                        ##sensor notification updates value of the state in 
                        sensor_notification = copy.deepcopy(realStateANDSensorUpdate(step, self._output_file_name))

                        ##@II
                        # print("please enter reply:")
                        # language_notifcation = str(raw_input("please enter reply:"))
                        language_notifcation = language_notifcation_list[index %len(language_notifcation_list)].lower().replace(",", "")
                        # index+=1

                        exp.setSensorNotification(sensor_notification)
                        exp.setLanguageNotification(language_notifcation)
                        print("language_notification", language_notifcation)
                        # time.sleep(5)
                    # posterior pending, next step recognition.
                    ##@II
                    last_exp = copy.deepcopy(exp)
                    otherhappen = exp.action_posterior()
            
                    # otherhappen = 0.99
                    
                    # wrong step detect
                    if otherhappen > self._other_happen:
                        # wrong step handling
                        exp.handle_exception()
                        
                    
                    # # correct step procedure
                    else:
                        length = len(exp._explaset)
                        
                        # input step start a new goal (bottom up procedure to create ongoing status)
                        # include recognition and planning
                        exp.explaSet_expand_part1(length)

                        # belief state update, based on knowledge graph and prior of PS
                        state = State()
                        state.update_state_belief(exp)
                        
                        # input step continues an ongoing goal
                        # include recognition and planning 
                        exp.explaSet_expand_part2(length)
                        

                         
                exp.pendingset_generate()
                
                # compute goal recognition result PROB and planning result PS
                taskhint = exp.task_prob_calculate()
                print("taskhint is", taskhint.__dict__)
                
                #output PROB and PS in a file
                ## @II here decide prob of recognizing each task
                # exp.print_explaSet1()
                exp.print_explaSet()
                index+=1
                print("go into the next loop")
                print 
                print
                
                
        
        
        
       
            
       
            
        
    
    
