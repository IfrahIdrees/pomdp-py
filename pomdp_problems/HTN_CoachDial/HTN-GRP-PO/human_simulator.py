from notification import *
import os
from collections import defaultdict
import numpy as np
import random
import config
random.seed(10)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
htn_coachdial_dir = os.path.dirname(BASE_DIR)
pomdp_problems_dir = os.path.dirname(htn_coachdial_dir)
pomdp_py_dir = os.path.dirname(pomdp_problems_dir)
WORKTREE_DIR_BASELINE = os.path.dirname(pomdp_py_dir)
# pomdp_py_dir + "/HTN-Language-ObservationalModel"
TESTCASES_DIR = WORKTREE_DIR_BASELINE + "/TestCases"

class human_simulator(object):
    def __init__(self):
        self._notifs = [] ##List of notifications.
        self._notifs_to_index = defaultdict(list)
        # self._index_to_notifs = defaultdict(list)
        self.forgetfulness = config.forgetfulness
        self.wrong_actions = {}
        self.wrong_actions[0] =["use_soap","open_tea_box", "rinse_hand"] #soft 
        self.wrong_actions[1]=["turn_off_faucet_1", "close_tea_box"] #hard
        self.correct_actions = ["turn_on_faucet_1", "open_tea_box"] #hard
        self.index_test_case = None
        self.bool_wrong_actions = None
        self.prev_step = None
        self.start_action = {}
        self.read_files(TESTCASES_DIR)
        
        # self.

    def read_files(self,dir_name):
        # notifs = []
        # index = 0
        for file_name in os.listdir(dir_name):
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
    def goal_selection(self):
        self.index_test_case = random.randint(0, len(self._notifs)-1)

    def curr_step(self, prev_step, action):
        # title = "sensor_notif"
        # title_split = title.split("_")
        # self.prev_step = prev_step.attributes[title_split]
        # self.prev_step  = prev_step._sensor_notification
        self.prev_step = prev_step
        num = random.random()
        if self.bool_wrong_actions and action == "give-next-instruction":
            curr_step = self.correct_actions[self.bool_wrong_actions]
            self.bool_wrong_actions = None
        else:
            if num < 1 - self.forgetfulness:
                # print("goal test case index", )
                curr_step = self._notifs[self.index_test_case].get_one_notif()
            else:
                index_soft_or_hard = random.randint(0,1)
                if index_soft_or_hard:
                    self.bool_wrong_actions = index_soft_or_hard
                curr_step = random.choice(self.wrong[index_soft_or_hard])
            return curr_step
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

        

