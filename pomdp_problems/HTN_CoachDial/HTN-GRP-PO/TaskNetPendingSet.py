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
####          The TaskNetPendingSet class. Map to the "decomposed goal network" in thesis          ####
####          Also refer to "Interface specification part II"                                      ####
#######################################################################################################


import sys
sys.dont_write_bytecode = True

from treelib import Tree
from treelib import Node
from TaskNet import *


class TaskNetPendingSet(object):
    def __init__(self, tree = Tree(), branch_factor = 1, pending_actions = []):
        self._tree = tree
        self._branch_factor = branch_factor
        self._pending_actions = pending_actions
   
   
    #the action exist in the pending_actions of the TaskNetPendingSet,
    #and now this action has happened. generate a new TaskNet based on 
    #this decomposition.
    def generate_new_taskNet(self, action):
        theTree = self._tree
        action_node = theTree.get_node(action)
        action_node.data._completeness = True
        print(theTree.get_node(theTree.root).tag)
        print(self._branch_factor)
        newTaskNet = TaskNet(goalName = theTree.get_node(theTree.root).tag, tree = theTree, expandProb = self._branch_factor)
        newTaskNet.update()
        return newTaskNet
