from notification import *
import os
from collections import defaultdict
import numpy as np
import random
import config
random.seed(10)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pomdp_problems_dir = os.path.dirname(BASE_DIR)
pomdp_py_dir = os.path.dirname(pomdp_problems_dir)
WORKTREE_DIR_BASELINE = os.path.dirname(pomdp_py_dir)
# pomdp_py_dir + "/HTN-Language-ObservationalModel"
TESTCASES_DIR = WORKTREE_DIR_BASELINE + "/TestCases"

class human_simulator(object):
    def __init__(self):
        self._notifs = []
        self._notifs_to_index = defaultdict(int)
        self._index_to_notifs = defaultdict(list)
        self.forgetfulness = config.forgetfulness
        self.wrong_actions = {}
        self.wrong_actions[0] =["use_soap","open_tea_box", "rinse_hand"] #soft 
        self.wrong_actions[1]=["turn_off_faucet_1", "close_tea_box"] #hard


    def read_files(self,dir_name):
        # notifs = []
        index = 0
        for file_name in os.listdir(dir_name):
            notif = notification(filename)
            notif._notif.put("End")
            self._notifs.extend(list(notif._notif))
            self._index_to_notifs[index] = notif
            self._notifs_to_index[notif].append(index)
            index+=1

    def curr_step(self, prev_step):
        indices = np.asarray(self._notifs_to_index[prev_step])
        curr_step_indices = indices+1
        curr_step = [self._notifs_to_index[ind] for ind in curr_step_indices]

        num = random.random()
        if num < 1 - self.forgetfulness:
            curr_step = random.choice(curr_step)
        else:
            index_soft_or_hard = random.randint(0,1)
            curr_step = random.choice(self.wrong[index_soft_or_hard])
        return curr_step

    def probability(self, next_state, state):
        indices = np.asarray(self._notifs_to_index[state])
        next_step_indices = indices+1
        next_step = [self._notifs_to_index[ind] for ind in next_step_indices]

        if next_state in next_step:
            return 1 -  self.forgetfulness
        else:
            return self.forgetfulness

    def get_all_states(self):
        return self._notifs_to_index.keys()

    # for notif in notifs:

        

