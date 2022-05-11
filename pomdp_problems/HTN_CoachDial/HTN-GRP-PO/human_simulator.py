from notification import *
import os
from collections import defaultdict
import numpy as np
import random
import config
random.seed(10)
from Simulator import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
htn_coachdial_dir = os.path.dirname(BASE_DIR)
pomdp_problems_dir = os.path.dirname(htn_coachdial_dir)
pomdp_py_dir = os.path.dirname(pomdp_problems_dir)
WORKTREE_DIR_BASELINE = os.path.dirname(pomdp_py_dir)
# pomdp_py_dir + "/HTN-Language-ObservationalModel"
TESTCASES_DIR = WORKTREE_DIR_BASELINE + "/TestCases"

class human_simulator(object):
    def __init__(self, output_filename, mcts_output_filename):

        self.index_test_case = int(output_filename.split("/")[-1].split("_")[0][4:])-1
        self._notifs = [] ##List of notifications.
        self._notifs_to_index = defaultdict(list)
        # self._index_to_notifs = defaultdict(list)
        self.forgetfulness = config.forgetfulness
        self.wrong_actions = {}
        self.wrong_actions[0] =["use_soap","open_tea_box_1", "rinse_hand"] #soft 
        self.wrong_actions[1]=["turn_off_faucet_1", "close_tea_box_1"] #hard
        self.all_wrong_actions =  sum(self.wrong_actions.values(),[])
        self.correct_actions = ["turn_on_faucet_1", "open_tea_box_1"] #hard
        # self.index_test_case = None
        # self.bool_wrong_actions = None
        # self.prev_step = None
        self.start_action = {}
        self.read_files(TESTCASES_DIR)
        ''' moving mcts step index from simulator to state '''
        # self.mcts_step_index =  -1
        self.real_step_index = -1
        self.mcts_bool_wrong_actions = None
        self.real_bool_wrong_actions = None
        # self.mcts_bool_wrong_actions = None
        self.real_output_filename = output_filename
        self.mcts_output_filename = mcts_output_filename
        self.sensor_notification_dict = {} #step_name:sensor_notification
        self.wrong_per_test_case = {
            #testcase(0 indexed):[step_numbers]
            6: [2],
            7: [2,3],
            8: [2,3,7],
            9: [1,9,10],
            10: [3,14],
            11: [6,8,10]
        }
        # self.

    def read_files(self,dir_name):
        # notifs = []
        # index = 0

        sorted_filename = sorted(os.listdir(dir_name))
        sorted_filename.remove("ignore")
        # for ind, filename in sorted_filename:
        ind = 0
        filename = sorted_filename[0]
        while filename != "Case2":
            if filename == "Case1":
                ind+=1
                filename = sorted_filename[ind]
                continue
            filename = sorted_filename.pop(ind)
            sorted_filename.append(filename)
            filename = sorted_filename[ind]



        for file_name in sorted_filename:
            if os.path.isfile(os.path.join(dir_name, file_name)):
                notif = notification(dir_name+"/"+file_name)
                # notif._notif = list(self._notif)
                self._notifs.append(notif)
                index = 0
                for ind, n in enumerate(list(notif._notif.queue)):
                    if ind == 0:
                        testcase_number = file_name[4:]
                        self.start_action[int(testcase_number)] = n
                    # self._index_to_notifs[index] = n
                    self._notifs_to_index[n].append(index)
                    index+=1
                ##design to have all files in one dict
                # notif._notif.put("End")
                # self._notifs.extend(list(notif._notif))
                # self._index_to_notifs[index] = notif
                # self._notifs_to_index[notif].append(index)
                # index+=1
    def random_goal_selection(self):
        self.index_test_case = random.randint(0, len(self._notifs)-1)
        print("Selected test case is", self.index_test_case)

    # def curr_step(self, prev_step, action, real_step = False):
    #     num = random.random()
    #     if real_step:
    #         prev_step_index = self.real_step_index
    #         # bool_wrong_actions = self.real_bool_wrong_actions
    #         # self.real_bool_wrong_actions = None
    #     else:
    #         prev_step_index = prev_step
    #         # prev_step_index = self.mcts_step_index
    #         # bool_wrong_actions = self.mcts_bool_wrong_actions
    #         # self.mcts_bool_wrong_actions = None

    #     # title = "sensor_notif"
    #     # title_split = title.split("_")
    #     # self.prev_step = prev_step.attributes[title_split]
    #     # self.prev_step  = prev_step._sensor_notification
    #     # self.prev_step = prev_step
    #     # TODO: Done Need to deal with wrong action execution for mcts and real separately.
        
        
    #         # print("wrong step - true", num < 1 - self.forgetfulness)
    #     prev_step_index+=1
    #         # print(self._notifs[self.index_test_case]._notif.queue, prev_step_index )
    #     curr_step = self._notifs[self.index_test_case].get_one_notif(prev_step_index)

    #     if real_step:
    #         self.real_step_index = prev_step_index 
    #         sensor_notification = realStateANDSensorUpdate(curr_step, self.real_output_filename, real_step = True)
    #         # print("simulation", prev_step_index, curr_step)
    #         a=1
    #     else:
    #         # self.mcts_step_index = prev_step_index
    #         sensor_notification = realStateANDSensorUpdate(curr_step, self.mcts_output_filename, real_step = False)
    #     self.sensor_notification_dict[curr_step] = sensor_notification
    #     return prev_step_index, curr_step, sensor_notification
        

    # this curr step function dealt with wrong steps+96
    def curr_step(self, prev_step, action, real_step = False):
        num = random.random()
        if real_step:
            prev_step_index = self.real_step_index
            bool_wrong_actions = self.real_bool_wrong_actions
            # self.real_bool_wrong_actions = None
        else:
            prev_step_index = prev_step
            # prev_step_index = self.mcts_step_index
            bool_wrong_actions = self.mcts_bool_wrong_actions
            # self.mcts_bool_wrong_actions = None

        # title = "sensor_notif"
        # title_split = title.split("_")
        # self.prev_step = prev_step.attributes[title_split]
        # self.prev_step  = prev_step._sensor_notification
        # self.prev_step = prev_step
        # TODO: Done Need to deal with wrong action execution for mcts and real separately.
        
        if bool_wrong_actions and action == "give-next-instruction":
            curr_step = self.correct_actions[bool_wrong_actions]
            # self.bool_wrong_actions = None

            if real_step:
                self.real_bool_wrong_actions = None
            else:
                self.mcts_bool_wrong_actions = None
        else:
            # print("wrong step - true", num < 1 - self.forgetfulness)
            if num < 1 - self.forgetfulness:
                # print("goal test case index", )
                prev_step_index+=1
                # print(self._notifs[self.index_test_case]._notif.queue, prev_step_index )
                curr_step = self._notifs[self.index_test_case].get_one_notif(prev_step_index)

            else:
                index_soft_or_hard = random.randint(0,1)
                if index_soft_or_hard:
                    bool_wrong_actions = index_soft_or_hard
                curr_step = random.choice(self.wrong_actions[index_soft_or_hard])


        if real_step:
            self.real_step_index = prev_step_index 
            sensor_notification = realStateANDSensorUpdate(curr_step, self.real_output_filename, real_step = True)
            # print("simulation", prev_step_index, curr_step)
            a=1
        else:
            # self.mcts_step_index = prev_step_index
            sensor_notification = realStateANDSensorUpdate(curr_step, self.mcts_output_filename, real_step = False)
        self.sensor_notification_dict[curr_step] = sensor_notification
        return prev_step_index, curr_step, sensor_notification
        # pass


        # ##design to have all files in one dict
        # indices = np.asarray(self._notifs_to_index[prev_step])
        # curr_step_indices = indices+1
        # curr_step = [self._notifs_to_index[ind] for ind in curr_step_indices]

        # num = random.random()
        # if num < 1 - self.forgetfulness:
        #     curr_step = random.choice(curr_step)
        # else:
        #     index_soft_or_hard = random.randint(0,1)
        #     curr_step = random.choice(self.wrong[index_soft_or_hard])
        # return curr_step 

    def probability(self, next_state, state):
        state =  state._sensor_notification
        next_state = next_state._sensor_notification
        if state == None and next_state == self.start_action[self.goal_selection]:
            return 1-self.forgetfulness
        elif abs(self._notifs_to_index[next_state] - self._notifs_to_index[state]) == 1:
            return 1-self.forgetfulness 
        else:
            return self.forgetfulness
            

        # if abs(self._notifs_to_index[next_state] - self._notifs_to_index[state]) == 1:

        # indices = np.asarray(self._notifs_to_index[state])
        # next_step_indices = indices+1
        # next_step = [self._notifs_to_index[ind] for ind in next_step_indices]

        # if next_state in next_step:
        #     return 1 -  self.forgetfulness
        # else:
        #     return self.forgetfulness

    def get_all_states(self):
        return self._notifs_to_index.keys()

    # for notif in notifs:

    def check_notif_queue_length(self):
        return self._notifs[self.index_test_case]._notif.qsize()

    def check_terminal_state(self, curr_state_step_index):
        # print(curr_state_step_index, self._notifs[self.index_test_case]._notif.qsize() -1 )
        return curr_state_step_index > (self._notifs[self.index_test_case]._notif.qsize() - 1)
    
    # def real_check_terminal_state(self):
    #     return self.real_step_index == (self._notifs[self.index_test_case]._notif.qsize() - 1)
    
    def clear_mcts_history(self):
        return self.real_step_index

    def return_step(self, step):
        return self._notifs[self.index_test_case].get_one_notif(step)
        

    def check_wrong_step(self, step_index):
        # print(step_index, self.wrong_per_test_case.keys(), self.index_test_case)
        if self.index_test_case in self.wrong_per_test_case.keys():
            return step_index in self.wrong_per_test_case[self.index_test_case]
        else:
            return False
