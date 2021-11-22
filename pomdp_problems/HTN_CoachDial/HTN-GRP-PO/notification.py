"""------------------------------------------------------------------------------------------
Hierarchical Task Recognition and Planning in Smart Homes with Partially Observability
Author: Dan Wang danwangkoala@gmail.com (May 2016 - June 2017)
Supervised by Prof. Jesse Hoey (https://cs.uwaterloo.ca/~jhoey/)
Association: Computer Science, University of Waterloo.
Research purposes only. Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by AGEWELL Networks of Centers of Excellence (NCE).
----------------------------------------------------------------------------------------------"""
#store the input test case into a step list


from queue import *

class notification(object):
    def __init__(self, file_name):
        self._notif = Queue()
        
        step_input = open(file_name, "r")
        steps = step_input.readlines()
        step_input.close()
        
        for step in steps:
            step = ''.join(step.split("\n"))
            self._notif.put(step)
        
    ##without deleting
    def get_one_notif(self):
        if self._notif.empty():
            return None
        else:
            return self._notif.queue[0]  
        
        
    ##delete the next element in insertion order
    def delete_one_notif(self):
        if not self._notif.empty():
            self._notif.get() 

