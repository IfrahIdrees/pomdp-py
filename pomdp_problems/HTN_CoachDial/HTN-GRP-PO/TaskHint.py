"""------------------------------------------------------------------------------------------
Hierarchical Task Recognition and Planning in Smart Homes with Partially Observability
Author: Dan Wang danwangkoala@gmail.com (May 2016 - June 2017)
Supervised by Prof. Jesse Hoey (https://cs.uwaterloo.ca/~jhoey/)
Association: Computer Science, University of Waterloo.
Research purposes only. Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by AGEWELL Networks of Centers of Excellence (NCE).
----------------------------------------------------------------------------------------------"""

#######################################################################################################
####                    The TaskHint class. Produce hierarchical prompt                            ####
####                    Also refer to "Interface specification part II"                            ####
#######################################################################################################


import sys
sys.dont_write_bytecode = True
from helper import *


class TaskHint(object):
    def __init__(self, output_file_name = "Case4.txt"):
        self._output_file_name = output_file_name
        self.prompt_task = {}
        self.step_dict = set(['use_soap', 'rinse_hand', 'turn_on_faucet_1', 'turn_off_faucet_1', 'dry_hand', 'switch_on_kettle_1', 'switch_off_kettle_1', 'add_water_kettle_1', 'get_cup_1', 'open_tea_box_1', 'add_tea_cup_1', 'close_tea_box_1', 'add_water_cup_1', 'open_coffee_box_1', 'add_coffee_cup_1', 'close_coffee_box_1', 'drink'])
        
        
    
    #reset the prompt_task
    def reset(self):
        self.prompt_task = {}    
    
    #task_id: the name of the task
    #expla_prob: the probability of the corresponding explanation
    #level: the list of level of the task in this explanation, it is a list>>
    def add_task(self, task_tag, expla_prob, level):
        if task_tag in self.prompt_task.keys():
            key_value = self.prompt_task.get(task_tag)
            key_value[0] = key_value[0]+expla_prob
            key_value[1] = key_value[1]+level
            new_dict = {task_tag: key_value}
            self.prompt_task.update(new_dict)
        else:
            key_value = []
            key_value.append(expla_prob)
            key_value.append(level)
            new_dict = {task_tag:key_value}
            self.prompt_task.update(new_dict)    
        
    def average_level(self):
        
        for k, v in self.prompt_task.items():
            ave = list_average(v[1])    #ave is average level
            key_value = []
            key_value.append(v[0])
            key_value.append(ave)
            new_dict = {k:key_value}
            self.prompt_task.update(new_dict)
            
    def get_key(self, item):
        return item[1]
    
    def print_taskhintInTable(self):
        step_level_hint = {}
        for k, v in self.prompt_task.items():
            if k in self.step_dict:
                step_level_hint[k] = round(v[0], 8)
                
        wash_hand = 0.0
        make_tea = 0.0
        make_coffee = 0.0
        
        if 'wash_hand' in self.prompt_task:
            wash_hand = round(self.prompt_task['wash_hand'][0], 8)
        if 'make_tea' in self.prompt_task:
            make_tea = round(self.prompt_task['make_tea'][0], 8)
        if 'make_coffee' in self.prompt_task:
            make_coffee = round(self.prompt_task['make_coffee'][0], 8)
            
        with open(self._output_file_name, 'a') as f:
            f.write(str(wash_hand) + "\t" + str(make_tea) + "\t" + str(make_coffee) + "\t" + str(step_level_hint) + "\t")
        
    def print_taskhint(self):
        hint_in_level_format = {}
        for k, v in self.prompt_task.items():
            if v[1] in hint_in_level_format:
                hint_in_level_format[v[1]].append([k, v[0]])
            else:
                level_task_list = []
                level_task_list.append([k, v[0]])
                hint_in_level_format[v[1]] = level_task_list
            
        for key in hint_in_level_format:
            hint_in_level_format[key] = sorted(hint_in_level_format[key], key = self.get_key, reverse = True)
        
        with open(self._output_file_name, 'a') as f:
            f.write("Hint Output In Level Sequence: \n")
            for key in hint_in_level_format:
                line_new = "------------Level  " + str(key) + "-------------------\n"
                f.write(line_new)
                for task in hint_in_level_format[key]:
                    line_new = '{:>8}  {:<20}  {:>20}  {:>12}'.format("task name: ", task[0], "with probability of: ", round(task[1], 4))
                    f.write(line_new)
                    f.write("\n")
                f.write("\n")
            f.write("\n")
