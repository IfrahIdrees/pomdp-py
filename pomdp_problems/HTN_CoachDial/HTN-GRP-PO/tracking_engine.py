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

from collections import defaultdict
from math import gamma
# from os import EX_TEMPFAIL
import random
from signal import set_wakeup_fd
import time

from pomdp_problems.multi_object_search.models.components import sensor
from notification import *
from ExplaSet import *
from State import *
from Simulator import *
import random
import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
from HTNCoachDial import *
import config
random.seed(10)
from human_simulator import *

class Tracking_Engine(object):
    def __init__(self, no_trigger = 0, sleep_interval = 1, cond_satisfy=1.0, cond_notsatisfy = 0.0, delete_trigger = 0.001, non_happen = 0.00001, otherHappen = 0.75, file_name = "Case1", output_file_name = "Case1.txt", mcts_output_filename="mcts_case1.txt", args = None, db_client = None):
        self._no_trigger = no_trigger
        self._sleep_interval = sleep_interval
        self._cond_satisfy = cond_satisfy
        self._cond_notsatisfy = cond_notsatisfy
        self._delete_trigger = delete_trigger
        self._non_happen = non_happen
        self._other_happen = otherHappen
        self._file_name = file_name
        self._output_file_name = args.log_dir+"/"+output_file_name
        self._db_client = db_client
        # self.mcts_output_filename

        '''setting varibale in config file '''
        config._no_trigger = no_trigger
        config._sleep_interval = sleep_interval
        config._cond_satisfy = cond_satisfy
        config._cond_notsatisfy = cond_notsatisfy
        config._delete_trigger = delete_trigger
        config._non_happen = non_happen
        config._other_happen = otherHappen
        config._file_name = file_name
        config._output_file_name = self._output_file_name
        config._mcts_output_filename = args.log_dir+"/mcts/"+mcts_output_filename


        self._p_l = 0.95

        ##Pomdp object instantiation
        # self.init_belief = type('test', (), {})()
        self.init_worldstate_belief = list(db._state.find())
        self.explaset = explaSet(cond_satisfy = self._cond_satisfy, cond_notsatisfy = self._cond_notsatisfy, delete_trigger = self._delete_trigger, non_happen = self._non_happen, output_file_name = config._output_file_name, mcts_output_filename = config._mcts_output_filename)
        self.explaset.explaInitialize() 
        self.init_worldstate_belief, self.init_worldstate_state, self.observation_prob = convert_object_belief_to_histogram(self.init_worldstate_belief)
        
        '''
        ### remove sensor title
        self.sensor_title = "sensor-notif"
        self.sensor_title = self.sensor_title.split("-")

        init_sensor_state_val = ObjectAttrAndLangState(self.sensor_title[0], {self.sensor_title[1]: None})
        self.init_sensor_state = {}
        self.init_sensor_state["-".join(self.sensor_title)] = init_sensor_state_val
        init_sensor_hist = {}
        self.init_sensor_belief = defaultdict(pomdp_py.Histogram)
        init_sensor_hist[init_sensor_state_val] = 1
        self.init_sensor_belief["-".join(self.sensor_title)] = pomdp_py.Histogram(init_sensor_hist)
        '''
        
        self._step_dict = ['use_soap', 'rinse_hand', 'turn_on_faucet_1', 'turn_off_faucet_1', 'dry_hand', 'switch_on_kettle_1', 'switch_off_kettle_1', 'add_water_kettle_1', 'get_cup_1', 'open_tea_box_1', 'add_tea_cup_1', 'close_tea_box_1', 'add_water_cup_1', 'open_coffee_box_1', 'add_coffee_cup_1', 'close_coffee_box_1', 'drink']
        self.init_explaset_hist = {}
        self.init_explaset_belief = defaultdict(pomdp_py.Histogram)
        
        self.explaset_title = "explaset-action" ##stores the actual sensor notification
        self.explaset_title = self.explaset_title.split("-")
        init_explaset_state_val = ObjectAttrAndLangState(self.explaset_title[0], {self.explaset_title[1]:[]})
        self.init_explaset_state ={}
        self.init_explaset_state["-".join(self.explaset_title)] = init_explaset_state_val
        self.init_explaset_hist[init_explaset_state_val] = 1
        
        self.question_title = "language-indexQuestionAsked"
        self.question_title = self.question_title.split("-")
        init_question_asked_state_val = ObjectAttrAndLangState(self.question_title[0],{self.question_title[1]:None} ) 
        self.init_question_asked_state ={}
        self.init_question_asked_state["-".join(self.question_title)] =  init_question_asked_state_val
        self.init_question_asked_hist = {}
        self.init_question_asked_hist[init_question_asked_state_val] = 1
        self.init_question_asked_belief =defaultdict(pomdp_py.Histogram)

        for step in self._step_dict:
            # obj_id = "action_"+ step
            '''Not add rest of actions, explaset-action is now just a list of actions'''
            # self.init_explaset_hist[ObjectAttrAndLangState(self.explaset_title[0], {self.explaset_title[1]:step})] = 0 
            self.init_question_asked_hist[ObjectAttrAndLangState(self.question_title[0],{self.question_title[1]:step} )] = 0  
        self.init_explaset_belief["-".join(self.explaset_title)] = pomdp_py.Histogram(self.init_explaset_hist)
        self.init_question_asked_belief["-".join(self.question_title)] = pomdp_py.Histogram(self.init_question_asked_hist)
        

        self.feedback_title = "language-feedback"
        self.feedback_title = self.feedback_title.split("-")
        self.init_feedback_state ={}
        init_feedback_state_val = ObjectAttrAndLangState(self.feedback_title[0],{self.feedback_title[1]:None} ) 
        self.init_feedback_state["-".join(self.feedback_title)] =  init_feedback_state_val
        
        # self.init_feedback_state = ObjectAttrAndLangState("language", {"feedback":None})
        self.init_feedback_hist = {}
        self.init_feedback_belief = defaultdict(pomdp_py.Histogram)
        self.init_feedback_hist[init_feedback_state_val] = 1
        self.init_feedback_hist[ObjectAttrAndLangState("language", {"feedback":"yes"})] = 0 
        self.init_feedback_hist[ObjectAttrAndLangState("language", {"feedback":"no"})] = 0 
        self.init_feedback_belief["-".join(self.feedback_title)] = pomdp_py.Histogram(self.init_feedback_hist)
        
        
        # max(self.explaset._action_posterior_prob, key=self.explaset._action_posterior_prob.get)
        #  max(obj_state_dict, key=obj_state_dict.get)
        #{objid: object_state}
        htn_explaset = None
        step_index = -1
        self.init_state = HTNCoachDialState(step_index, htn_explaset, {**self.init_worldstate_state, **self.init_explaset_state, **self.init_feedback_state, **self.init_question_asked_state})
        
        #{object_state:prob}
        self.init_belief = HTNCoachDialBelief( {**self.init_worldstate_belief, **self.init_explaset_belief, **self.init_feedback_belief, **self.init_question_asked_belief})
        self.init_belief.htn_explaset_action_posterior = self.explaset.action_posterior
        # self.observation_prob = {}
        ##declare the hs
        hs = human_simulator(config._output_file_name, config._mcts_output_filename)
        
        self.HTNCoachDial_problem = HTNCoachDial(  # observation noise
                                   self.init_state, self.init_belief, self.observation_prob, None,hs, args)
        # self.HTNCoachDial_problem.agent.set_belief(self.init_belief, prior  = True)
        ##initial belief decides the state. Sample from the belief.
        print("\n** Testing POUCT **")
        max_depth = args.maxsteps
        num_sims= args.num_sims
        discount_factor=args.d
        exploration_const=args.e
        self.pouct = pomdp_py.POUCT(max_depth=max_depth, discount_factor=discount_factor,
                            num_sims=num_sims, exploration_const=exploration_const,
                            rollout_policy=self.HTNCoachDial_problem.agent.policy_model,
                            show_progress=True)
        self.pouct._db = db
        self.pouct._hs = self.HTNCoachDial_problem.hs
        self.pouct._mcts_output_filename = config._mcts_output_filename 
        # reward_output_filename = "D{}_S{}_DF{}_E{}".format(
        #     max_depth,
        #     num_sims,
        #     discount_factor, 
        #     exploration_const
        # )
        self.HTNCoachDial_problem.reward_output_filename = args.log_dir+"/reward/Reward_"+output_file_name
        
        print("Current Parameters are:", config._output_file_name)

        with open(config._output_file_name, 'a') as f:
            f.write('\n========================\n')

        with open(config._mcts_output_filename , 'a') as f:
            f.write('\n========================\n')

        with open(self.HTNCoachDial_problem.reward_output_filename , 'a') as f:
            f.write('\n========================\n')

        #  "maxdepth_"+max_depth+"_df"+"Reward_"+ output_filename
        
                            # num_sims=4096
        ##TODO: SHIFT THE TEST PLANNER
        # test_planner(self.HTNCoachDial_problem, pouct, nsteps=1, debug_tree=False)
        # TreeDebugger(self.HTNCoachDial_problem.agent.tree).pp

    def update_true_state(self, state, new_value_dict):
        obj_name = new_value_dict['object']
        attr_name = new_value_dict['attribute']
        val = new_value_dict['obj_att_value']

        key = "-".join([obj_name, attr_name])
        # state[key].__setitem__(attr_name, val)  
        old_objectstate = state.get_object_state(key)

        if key == "-".join(self.explaset_title):
            state.append_object_attribute(key, attr_name, val)
            # old_objectstate.__setitem__('attr_val', val)
        elif not key in ["-".join(self.explaset_title), "-".join(self.question_title), "-".join(self.feedback_title)]:
            old_objectstate.__setitem__('attr_val', val)
        else:
            old_objectstate.__setitem__(attr_name, val)
        # state.set_object_state(key, new_objectstate)
        return
    
    def start(self):
        print()
        print("the engine has been started...")
        print()
        
        notif = notification(self._file_name)   ##check the current notification
        # exp = explaSet(cond_satisfy = self._cond_satisfy, cond_notsatisfy = self._cond_notsatisfy, delete_trigger = self._delete_trigger, non_happen = self._non_happen, output_file_name = self._output_file_name)
        # exp.explaInitialize()  
        exp =  self.explaset
        
        total_reward = 0
        total_discounted_reward = 0
        index=0
        #always iterate
        prev_step = None
        step_index = -1
        action = None
        gamma = 1


        '''Set the exp set and true state '''

        step_index,step, sensor_notification = self.HTNCoachDial_problem.hs.curr_step(step_index, action, real_step = True)
        # print("step index, sensor_notif is", step_index, sensor_notification)
        # TODO: see if mcts curr_step should be called.
        # self.HTNCoachDial_problem.hs.curr_step(prev_step, action)
        # step = notif.get_one_notif()
        # notif.delete_one_notif()

        
        #if no notification, and the random prob is less than no_notif_trigger_prob, sleep the engine
        
        if step != "none":
            # sensor_notification = copy.deepcopy(realStateANDSensorUpdate(step, self._output_file_name))
            
            exp.setSensorNotification(sensor_notification)
                
        # posterior
        otherHappen, observation_prob = exp.action_posterior(execute=True)
        
        #TODO: self.HTNCoachDial_problem.agent.observation_model.set_lang_objattrs_prob(observation_prob)
        
        # wrong step detect
        if otherHappen > self._other_happen:
            # wrong step handling
            # print("action posterior after bayseian inference is",  exp._action_posterior_prob)
            print("sensor_notif is:", sensor_notification)
            exp.handle_exception()
            
        # correct step procedure
        else:
            length = len(exp._explaset)

            ##for develop branch
            exp.update_without_language_feedback(self._p_l)
            
            # input step start a new goal (bottom up procedure to create ongoing status)
            # include recognition and planning
            exp.explaSet_expand_part1(length)

            # belief state update
            state = State()
            state.update_state_belief(exp)
            ## TODO: user the above function to update belief over the state.
            
            # input step continues an ongoing goal
            # include recognition and planning 
            exp.explaSet_expand_part2(length)
            

                    
        exp.pendingset_generate()
        
        # compute goal recognition result PROB and planning result PS
        exp.task_prob_calculate(self.HTNCoachDial_problem.hs.real_output_filename)
        
        #output PROB and PS in a file
        exp.print_explaSet()

        ## change true state of the environment 
        #  {**self.init_worldstate_state, **self.init_explaset_state, **self.init_sensor_state, **self.init_feedback_state, **self.init_question_asked_state})

        # self.HTNCoachDial_problem.env.set_htn_explaset(self.HTNCoachDial_problem.env.state, exp)
        # self.HTNCoachDial_problem.agent.cur_belief.set_step_index(self.HTNCoachDial_problem.hs.mcts_step_index)
        self.HTNCoachDial_problem.agent.cur_belief.set_step_index(step_index)
        self.HTNCoachDial_problem.agent.cur_belief.set_htn_explaset(exp)
        self.HTNCoachDial_problem.env.state.set_htn_explaset(exp)
        self.HTNCoachDial_problem.env.state.set_step_index(step_index)
        # self.HTNCoachDial_problem.env.state.get_object_state()
        print("step_index",step_index,"step",step, "sensor_notif:", sensor_notification)
        if sensor_notification: #sensor noti
            self.update_true_state(self.HTNCoachDial_problem.env.state, sensor_notification[0])## update world state,
        # attribute =  self.explaset_title[1]
        # self.explaset_title = "-".join(self.explaset_title)
        explaset_state_dict = {"object":self.explaset_title[0], "attribute": self.explaset_title[1] , "obj_att_value": step}
        self.update_true_state(self.HTNCoachDial_problem.env.state, explaset_state_dict)## update explaset action which stores the current sensor (TODO: MAYBE remove sensor state)

        while not self.HTNCoachDial_problem.hs.check_terminal_state(step_index):
            # print("loop condition", self.HTNCoachDial_problem.hs.check_terminal_state(step_index))
            
            rand_ = random.random()
            if rand_<self._no_trigger:
                time.sleep(self._sleep_interval)
                
            #go through the engine logic
            else:
        # while(notif._notif.qsize()>0):
            # self.HTNCoachDial_problem.hs.clear_mcts_history()
            
                # TODO: update feedback and question asked from previous loop which is None for the first time.

                
                # self.init_explaset_state

                '''Need to make copy of db collections'''
                # self._state = db.state
                # self._sensor = db.sensor
                # self._mcts_sensor = db.mcts_sensor

                pipeline = [ {"$match": {}}, 
                            {"$out": "backup_state"},
                ]
                db._state.aggregate(pipeline) ##update db_client.backup_state. db._backup_state (points to backup_state)
                # db._backup_state = self.db_client.backup_state

                pipeline = [ {"$match": {}}, 
                            {"$out": "backup_sensor"},
                ]
                db._sensor.aggregate(pipeline)

                print("going to plan")
                total_reward, total_discounted_reward, step_index, gamma = planner_one_loop(self.HTNCoachDial_problem, self.pouct, nsteps=1, debug_tree=False,  total_reward = total_reward, total_discounted_reward = total_discounted_reward, i=step_index, true_state = step, prob_lang =self._p_l, gamma = gamma)
                index+=1

                '''Not restore as set at start of simulation'''
                # '''Restoring the state for next iteration, env variable in HTNcoachproblem should be reset'''

                # pipeline = [ {"$match": {}}, 
                #             {"$out": "state"},
                # ]
                # db._backup_state.aggregate(pipeline)

                # pipeline = [ {"$match": {}}, 
                #             {"$out": "sensor"},
                # ]
                # db._backup_sensor.aggregate(pipeline)


                print("go into next loop", index)
                print()
                print()
        
        return total_reward, total_discounted_reward 

        # HTN
        # with open(self.HTNCoachDial_problem.reward_output_filename, 'a') as f:
        #     f.write('\n========================\n')
        '''
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

                    if config.RANDOM_BASELINE == True:
                        exp.highest_action_PS = random.choice(self._step_dict)
                    else:
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
                    
                    planner_one_loop(self.HTNCoachDial_problem, pouct, nsteps=1, debug_tree=False)
                    TreeDebugger(self.HTNCoachDial_problem.agent.tree).pp

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
                    ## code with old class interface
                    # exp.update_language_feedback(step, exp.highest_action_PS)

                    # init_true_state = random.choice([TigerState("tiger-left"),
                    #                  TigerState("tiger-right")])
                    # init_belief = pomdp_py.Histogram({TigerState("tiger-left"): 0.5,
                    #                                 TigerState("tiger-right"): 0.5})
                    # tiger_problem = HTNCoachDial(0.15,  # observation noise
                    #                             init_true_state, init_belief)

                    # print("\n** Testing POUCT **")
                    # pouct = pomdp_py.POUCT(max_depth=3, discount_factor=0.95,
                    #                     num_sims=4096, exploration_const=50,
                    #                     rollout_policy=tiger_problem.agent.policy_model,
                    #                     show_progress=True)
                    # test_planner(tiger_problem, pouct, nsteps=1, debug_tree=False)
                    # TreeDebugger(tiger_problem.agent.tree).pp

                    ###HTNCoachDial call
                    # init_true_state = object()
                    # init_true_state.exp = explaSet(cond_satisfy = self._cond_satisfy, cond_notsatisfy = self._cond_notsatisfy, delete_trigger = self._delete_trigger, non_happen = self._non_happen, output_file_name = self._output_file_name)
                    # init_true_state.exp.explaInitialize() 
                    # init_true_state.state = list(db._Rstate.find()) 
                    
                    # State().get_attr_in_effect(init_true_state.exp)

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
                
                '''
        
        
        
       
            
       
            
        
    
    
