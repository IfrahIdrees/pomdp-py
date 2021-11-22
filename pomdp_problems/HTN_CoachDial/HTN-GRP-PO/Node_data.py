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
####                    The Node_data class                                                         ####
####                    Also refer to "Interface specification part II"                            ####
#######################################################################################################


import sys
sys.dont_write_bytecode = True

class Node_data():
    def __init__(self, complete = False, ready=False, branch = False, pre="", dec=""): 
            self._completeness = complete #complete or not
            ##this information will tell whether this step can be implemented
            ##according to the tree structure
            self._ready = ready #can be implemented in the next step or not
            self._pre = pre #its predecessors
            self._dec = dec #its decedants
            self._branch = branch #if the branching prob for this node has been considered
    
    #out put all the properties of the data instance
    def print_property(self):
        print("completeness:", self._completeness)
        print("ready       :", self._ready)
        print("predecessors:", self._pre)
        print("decedants   :", self._dec)
        print("branch fac  :", self._branch)
