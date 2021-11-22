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
####                    The TaskNet class. Map to the "goal network" in thesis                     ####
####                    Also refer to "Interface specification part II"                            ####
#######################################################################################################



import sys
sys.dont_write_bytecode = True

import copy
from treelib import Tree
from treelib import Node
from collections import deque
from TaskNetPendingSet import *
from database import *
from helper import *
from Node_data import *
from ExecuteSequence import *


db = DB_Object()
class TaskNet(object):
    def __init__(self, goalName="", tree=Tree(), expandProb = 1, pendingset=TaskNetPendingSet(), complete = False, execute_sequence = None):
        self._goalName=goalName ##this is the goal name of the tree, it is a string;
        self._tree = tree
        self._expandProb=expandProb
        self._pendingset = pendingset #[tree, action, new_added branch prob compared with the self._tree]
        self._complete = complete
        self._execute_sequence = execute_sequence # 

    ##Functions:
    
    ##(1) update node completeness property
    ##(2) update node ready property
    ##(3) generate the new pending set
    ##(4) for each action in the pending set, calculate the branching prob based on the tree structure 
    
    def update(self):
        rootnode = self._tree.get_node(self._tree.root)
        
        #update completenes
        self.complete_update(rootnode, self._tree)
        
        if rootnode.data._completeness == True:
            self._complete=True
            return
        
        #update ready-ness
        self.readiness_update(self._tree.root, self._tree)
        
        #update pendingset
        self._pendingset = self.pendingset_update(self._tree.root, self._tree)
        
        
        
    #update pending set
    #pendingset is a list, each element has the format of 
    #[tree, action, new_added branch prob compared with the initial tree]
    def pendingset_update(self, node, tree):
        '''
        tree.show(line_type = "ascii")
        all_node = tree.all_nodes()
        print "Inside pendingSet update--------------------------------"
        for x in all_node:
            print x.tag, "   ", x.data._ready
        '''
        expand_tree = []
        tree_queue = deque([])
        tree_queue.append([copy.deepcopy(tree), 1])
        while tree_queue:
            thisTree = tree_queue.popleft()
            leaves = thisTree[0].leaves()
            
            finish = True 
            #to check if the tree finished its decomposition process
            for leaf in leaves:
                
                if leaf.data._ready==True:
                    method = db.find_method(leaf.tag)
                    if method==None: 
                        continue
                    else:
                        finish = False
                        branches = self.method_decompose(method)
                        for x in branches:
                            newTree = copy.deepcopy(thisTree[0])
                            self.add_child(newTree, leaf.tag, x[1])
                            tree_queue.append([newTree, thisTree[1]*x[0]])
                        break
                        
            if finish==True:
                self.readiness_update(self._tree.root, self._tree)
                tree_pending=[x.tag for x in leaves if x.data._ready==True and x.data._completeness==False]
                thisTree.append(tree_pending)
                mytasknetpending = TaskNetPendingSet(tree = thisTree[0], branch_factor = thisTree[1], pending_actions = thisTree[2])
                expand_tree.append(mytasknetpending)
        return expand_tree
    

    def method_decompose(self, method):
        ##Step 1: calculate the precondition satisfy prob for each branch
        prob = []
        for branch in method["precondition"]:
            prob_temp=1
            for ob_name in branch:
                for attri in branch[ob_name]:
                    prob_temp = prob_temp * db.get_attribute_prob_1(branch[ob_name][attri], ob_name, attri)    
            prob.append(prob_temp)
            
        ##Step 2: normatlize on the prob    
        my_normalize_1(prob)
        satisfy_branch = []
        for i, x in enumerate(method["subtasks"]):
            satisfy_branch.append([prob[i], x])
            
        return satisfy_branch      
    

    def add_child(self, tree, node_id, branch):
        for x in branch:
            mydata=None
            if len(branch[x]["pre"]) == 0:
                mydata=Node_data(ready=True, branch = False, pre=branch[x]["pre"], dec = branch[x]["dec"])
            else:
                mydata=Node_data(ready=False, branch = False, pre=branch[x]["pre"], dec = branch[x]["dec"])
            tree.create_node(x, x, parent = node_id, data = mydata)
 
    #update completeness------a new version
    def complete_update(self, node, tree):
        if node.is_leaf(): return
        child = node._fpointer #get the id of childrens
        
        for x in child:
            self.complete_update(tree.get_node(x), tree)
        
        completeness = True
        for x in child:
            thenode = tree.get_node(x)
            if thenode.data._completeness == False:
                completeness = False
                break
        
        node.data._completeness = completeness
        return
        
    
    #update readiness
    #idea: from top to bottom, BFS
    def readiness_update(self, root_id, tree):
        '''
        tree.show(line_type = "ascii")
        all_node = tree.all_nodes()
        print "before update the ready ones is--------------------------------"
        for x in all_node:
            print x.tag, "   ", x.data._ready
        '''
        
        node_queue = deque([])
        node_queue.append(root_id)
         
        while node_queue:
            cur_node = tree.get_node(node_queue.popleft())
            #case 1: this node has no parent, readiness = true
            if cur_node.is_root():
                cur_node.data._ready=True
            #case 2: this node has parent, and len(pre)=0
                    #equal to parents's readiness
            elif len(cur_node.data._pre)==0:
                pid = cur_node._bpointer
                cur_node.data._ready = tree.get_node(pid).data._ready
            #case 3: this node has parent, and len(pre)>0
            else:
                cur_node.data._ready=True
                pre_list = cur_node.data._pre
                
                for x in pre_list:
                    
                    precondition_node = tree.get_node(x)
                    if precondition_node.data._ready == False:
                        precondition_node.data._completeness = False
                    
                    if precondition_node.data._completeness == False:
                        
                        cur_node.data._ready = False
                        break
            if cur_node.is_leaf() == False:
                child = cur_node._fpointer
                node_queue.extend(child)        


##############################################################################################################
####                                handle exception                                                    ######
##############################################################################################################
    

    def repair_taskNet_lastnotif(self, sensor_notification):
        # affect_steps = False
        # the_key = sensor_notification[-1]["object"] + "/" + sensor_notification[-1]["attribute"]
        index = -1
        affect_steps = [False] * len(self._execute_sequence._sequence) #False: not affected, True: affected
        affect_num = 0
        # For the last affected steps in affect_steps, modify the completeness as False
        if len(self._execute_sequence._sequence) == 0 or not sensor_notification:
            return [False, None, None]
        else:
            
            notif = sensor_notification[-1]
            the_key = notif["object"] + "/" + notif["attribute"]
            
            if (the_key in self._execute_sequence._effect_summary) and \
                self._execute_sequence._effect_summary[the_key]["value"] != notif["obj_att_value"]:
                the_index = self._execute_sequence._sequence.index(self._execute_sequence._effect_summary[the_key]["step_name"])
                affect_steps[the_index] = True
                affect_num = affect_num + 1
        
            # For all the affected steps in affect_steps, modify the completeness as False
            for index in range(0, len(affect_steps)):
                if affect_steps[index] == True:
                    step_node = self._tree.get_node(self._execute_sequence._sequence[index])
                    if step_node != None and step_node.data._completeness == True:
                        step_node.data._completeness = False
                        self.repair_taskNet_completeness_readiness(step_node)
        
            if affect_num > 0:  #violate the hard constraints
                self._pendingset = copy.deepcopy(self.pendingset_update(self._tree.root, self._tree))
          
                # Update the _execute_sequence according to the updated tree structure
                new_execute_sequence = ExecuteSequence(sequence = [], effect_summary = {})    
                for step in self._execute_sequence._sequence:
                    step_node = self._tree.get_node(step)
                    if (step_node != None) and (step_node.data._completeness == True):
                        new_execute_sequence.add_action(step)
                
                # update the belief state
                new_effect_summary = copy.deepcopy(new_execute_sequence._effect_summary)
                old_effect_summary = copy.deepcopy(self._execute_sequence._effect_summary)
                self._execute_sequence = copy.deepcopy(new_execute_sequence)
                        
                return [True, new_effect_summary, old_effect_summary]
            else:
                return [False, None, None]

            #     self._pendingset = copy.deepcopy(self.pendingset_update(self._tree.root, self._tree))
            
            #     # Update the _execute_sequence according to the updated tree structure
            #     new_execute_sequence = ExecuteSequence(sequence = [], effect_summary = {})  



            # # step_node = self._tree.get_node(self._execute_sequence._sequence[index])
            # # if step_node != None and step_node.data._completeness == True:
            # #     step_node.data._completeness = False
            # #     self.repair_taskNet_completeness_readiness(step_node)

            # #if readiness anc completness is updated what is the next pending set.
            # self._pendingset = copy.deepcopy(self.pendingset_update(self._tree.root, self._tree))
            # # for voilated constraints Update the _execute_sequence according to the updated tree structure
            # new_execute_sequence = ExecuteSequence(sequence = [], effect_summary = {})    
            # for step in self._execute_sequence._sequence:
            #     step_node = self._tree.get_node(step)
            #     if (step_node != None) and (step_node.data._completeness == True):
            #         new_execute_sequence.add_action(step)

            # # updte the belief state
            # new_effect_summary = copy.deepcopy(new_execute_sequence._effect_summary)
            # old_effect_summary = copy.deepcopy(self._execute_sequence._effect_summary)
            # self._execute_sequence = copy.deepcopy(new_execute_sequence)
                        
            # return [True, new_effect_summary, old_effect_summary]

        

    def repair_taskNet(self, sensor_notification):
        affect_steps = [False] * len(self._execute_sequence._sequence) #False: not affected, True: affected
        affect_num = 0
        for notif in sensor_notification: ##@II look at last notif only.
            the_key = notif["object"] + "/" + notif["attribute"]
            
            if (the_key in self._execute_sequence._effect_summary) and \
                self._execute_sequence._effect_summary[the_key]["value"] != notif["obj_att_value"]:
                the_index = self._execute_sequence._sequence.index(self._execute_sequence._effect_summary[the_key]["step_name"])
                affect_steps[the_index] = True
                affect_num = affect_num + 1
        
        # For all the affected steps in affect_steps, modify the completeness as False
        for index in range(0, len(affect_steps)):
            if affect_steps[index] == True:
                step_node = self._tree.get_node(self._execute_sequence._sequence[index])
                if step_node != None and step_node.data._completeness == True:
                    step_node.data._completeness = False
                    self.repair_taskNet_completeness_readiness(step_node)
    
        if affect_num > 0:  #violate the hard constraints
            self._pendingset = copy.deepcopy(self.pendingset_update(self._tree.root, self._tree))
          
            # Update the _execute_sequence according to the updated tree structure
            new_execute_sequence = ExecuteSequence(sequence = [], effect_summary = {})    
            for step in self._execute_sequence._sequence:
                step_node = self._tree.get_node(step)
                if (step_node != None) and (step_node.data._completeness == True):
                    new_execute_sequence.add_action(step)
            
            # update the belief state
            new_effect_summary = copy.deepcopy(new_execute_sequence._effect_summary)
            old_effect_summary = copy.deepcopy(self._execute_sequence._effect_summary)
            self._execute_sequence = copy.deepcopy(new_execute_sequence)
                      
            return [True, new_effect_summary, old_effect_summary]
        else:
            return [False, None, None]
        
    
        
    # The completeness of this node changed from True to False            
    def repair_taskNet_completeness_readiness(self, node):
        affected_node = tree_queue = deque([])
        affected_node.append(node)
        while len(affected_node) > 0:
            current_node = affected_node.popleft()
            # change parent completeness state
            current_node_hold = current_node
            while (current_node_hold.bpointer != None):
                parent_node = self._tree.get_node(current_node_hold.bpointer)
                if parent_node.data._completeness == True:
                    parent_node.data._completeness = False
                    affected_node.append(parent_node)
                    current_node_hold = parent_node
                else:
                    break
                
            # change the decendent step readiness state and completeness state
            for x in current_node.data._dec:
                new_node = self._tree.get_node(x)
                if new_node != None:
                    new_node.data._ready = False
                    new_node.data._completeness = False
                    
                    # if the new_node is not leaf, then since it's readiness is False now
                    # delete it's child node
                    if new_node.is_leaf() == False:
                        child = copy.deepcopy(new_node._fpointer)
                        
                        for x in range(0, len(child)):
                            self._tree.remove_node(child[x])
                    
                    # append the new_node into affected not
                    affected_node.append(new_node)
        
    
    
       
    
    
   
                
                
                
               
   
                
                
                
                
                
                
                
                
                
                
                
                
                
    
        
