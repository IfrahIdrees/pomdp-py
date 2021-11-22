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
####                        ExecuteSequence class                                           ####
####                        Store execute sequence, effect summary                          ####
################################################################################################
import sys
sys.dont_write_bytecode = True

import copy
from termcolor import colored
from database import *


db = DB_Object()
# 1. sequence: []
# 2. effect_summary = {"obj_name/att_name": {"value": v, "step_name": sn}},}

class ExecuteSequence(object):
    def __init__(self, sequence = [], effect_summary = {}):
        self._sequence = sequence
        self._effect_summary = effect_summary
        
    def add_action(self, step_name):
        self._sequence.append(step_name)
        op = db.get_operator(step_name)
        for obj in op["effect"]:
            for att in op["effect"][obj]:
                new_key = obj + "/" + att
                if (new_key in self._effect_summary) and (self._effect_summary[new_key]["value"] == op["effect"][obj][att]):
                    continue
                else:
                    new_item = {}
                    new_item["value"] = op["effect"][obj][att]
                    new_item["step_name"] = step_name
                    self._effect_summary[new_key] = new_item
                    
                    
                    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                     
