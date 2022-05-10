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
####                    The Explanation class                                                      ####
####                    Also refer to "Interface specification part II"                            ####
#######################################################################################################



import sys
sys.dont_write_bytecode = True
from termcolor import colored
import copy
from treelib import Tree
from treelib import Node
from TaskNet import *
from Node_data import *
from database import *
from helper import *
from TaskHint import *
from ExecuteSequence import *

db = DB_Object()

class Explanation(object):
    
    def __init__(self, v=0, forest=[], pendingSet=[], start_task = {}):
        self._prob = v
        self._forest = forest
        self._pendingSet = pendingSet
        self._start_task = start_task #format {task1: 0, task2:1}, 0 stands for not started yet, 1 stands for ongoing
    
        
    def set_prob(self, v):
        self._prob = v
        
    def set_forest(self, forest):
        self._forest = forest
     
    def set_pendingSet(self, pendingset):
        self._pendingSet = list(pendingset)   
    
    def add_tasknet_ele(self, tasknet):
        self._forest.append(tasknet)
    
    def add_pendSet_ele(self, val):
        self._pendingSet.append(val)
                   
    def update_forest(self, forest):
        self._forest = forest
        
    def update_pendSet(self, pendingSet):
        self._pendingSet = pendingSet


    
    
#########################################################################################
#####           generate new explanations based on one action                       #####
#########################################################################################
   
    #use to udpate the current explanation according to the input action_explanation
    #act_expla is the explanation for this observation, expla is the current explanation
    ##"generate_new_expla_part1" is used to generate explanations that add a new tree structure, a bottom-up process
    ##The bottom-up process depend on the previous state.     
    def generate_new_expla_part1(self, act_expla):
        new_explas = []
        find = False
       
        ##Case1 : drop an on-going unfinished task, start a new one. 
        tempstart_task = copy.deepcopy(self._start_task)
        for start_task in tempstart_task:
            if tempstart_task[start_task] == 0: #inside this explanation, "start_task" has not been started
                target_method = db.find_method(start_task)
                if act_expla[0] in target_method["start_action"]:
                    find = True
                    newTaskNets = self.initialize_tree_structure(act_expla[0])
                    for g in newTaskNets:
                        if tempstart_task[g._goalName] == 0:
                            tempstart_task[g._goalName] = 1
                            newstart_task = copy.deepcopy(self._start_task)
                            prob = act_expla[1]*g._expandProb*self._prob
                            if g._complete == True:
                                newstart_task[g._goalName] = 0
                                newexp = Explanation(v=prob, forest = list(self._forest), start_task=newstart_task)
                            else:
                                newforest = list(self._forest)
                                #@thesiswriting
                                '''##here means it will start a new goal. 
                                ##However, if the already started goal do not have any execute sequence, remove
                                for tasknet in newforest:
                                    if len(tasknet._execute_sequence._sequence) == 0:
                                        newstart_task[tasknet._goalName] = 0
                                        newforest.remove(tasknet)'''
                                newforest.append(g)
                                newstart_task[g._goalName] = 1
                                newexp = Explanation(v=prob, forest = newforest, start_task=newstart_task)
                            new_explas.append(newexp)
        return new_explas
        
    #use to udpate the current explanation according to the input action_explanation
    #act_expla is the explanation for this observation, expla is the current explanation
    ##"generate_new_expla_part2" is used to generate explanations by decomposing an existing tree structure. , a top-down process
    ##The top=down process depends on the current state. (after adding the effect of just happened action)
    def generate_new_expla_part2(self, act_expla):
        new_explas = []
        find = False
        ##Case2 : continue on an on-going task
            ##update existing tree structure, if the action exist in the 
            ##pending set of this tree structure
        for taskNet in self._forest:
            for taskNetPending in taskNet._pendingset:
                if act_expla[0] in taskNetPending._pending_actions: 
                    find = True
                    
                    #get a new taskNet start
                    theTree = copy.deepcopy(taskNetPending._tree)
                    action_node = theTree.get_node(act_expla[0])
                    action_node.data._completeness = True
                    executed_sequence = ExecuteSequence(sequence = copy.deepcopy(taskNet._execute_sequence._sequence), effect_summary = copy.deepcopy(taskNet._execute_sequence._effect_summary))
                    executed_sequence.add_action(act_expla[0])
                    newTaskNet = TaskNet(goalName = theTree.get_node(theTree.root).tag, tree = theTree, expandProb = taskNetPending._branch_factor, execute_sequence = executed_sequence)
                    #@thesiswriting 
                    '''newTaskNet = TaskNet(goalName = theTree.get_node(theTree.root).tag, tree = theTree, expandProb = taskNetPending._branch_factor, execute_sequence = copy.deepcopy(executed_sequence))'''
                    newTaskNet.update()
                    #get a new taskNet end
                    
                    newforest = list(self._forest)
                    newforest.remove(taskNet)
                    prob = act_expla[1]*newTaskNet._expandProb*self._prob
                    
                        ##this goal has already been completed
                        ##remove it and add its start action into 
                        ##the explanation start action list
                    if newTaskNet._complete==True:
                        newstart_task = copy.deepcopy(self._start_task)
                        newstart_task[newTaskNet._goalName] = 0
                        newexp = Explanation(v=prob, forest = newforest, start_task=newstart_task)
                        
                        ##this goal has not been completed
                    else:
                        newforest.append(newTaskNet)
                        newstart_task = copy.deepcopy(self._start_task)
                        newexp = Explanation(v=prob, forest = newforest, start_task=newstart_task)
                    
                    new_explas.append(newexp)             

        return new_explas
        
    ##generate a tree structure for the action_explanation
    ##this function is called only when the _forest parameter is null
    ##(with length of 0)
    ##Input: is only the action name (probability is not included)
    def initialize_tree_structure(self, action):
        task_net = []
        temp_forest = deque([])
        tree = Tree()
        opdata = Node_data(complete = True)
        tree.create_node(action, action, data=opdata)
        temp_forest.append([tree, 1])     
        
        while temp_forest:
            length = len(temp_forest)
            for i in range(length):
                thisTree = copy.deepcopy(temp_forest.popleft())
                tag = thisTree[0].get_node(thisTree[0].root).tag
                parents = db.get_parent_list(tag)
                if parents==False: print("error happend here please check")
                if len(parents)>0: 
                    for x in parents: #x must be an method
                        method = db.find_method(x)
                        decompose_choose = self.method_precond_check(method,tag)
                        for decompose in decompose_choose:
                            decompose[0]=thisTree[1]*decompose[0]
                            temptree = copy.deepcopy(thisTree[0])
                            temp_forest.append(self.my_create_new_node(x, decompose, temptree))
        
                elif len(parents)==0: #this tree already reached goal node
                    executed_sequence = ExecuteSequence(sequence = [], effect_summary = {})
                    executed_sequence.add_action(action)
                    my_goal = TaskNet(goalName=tag, tree=thisTree[0], expandProb=thisTree[1], execute_sequence = executed_sequence)
                    #@thesiswriting
                    '''my_goal = TaskNet(goalName=tag, tree=thisTree[0], expandProb=thisTree[1], execute_sequence = copy.deepcopy(executed_sequence))'''
                    my_goal.update()
                    task_net.append(my_goal)
        
        return task_net        



    def method_precond_check(self, method, child):
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
        
        ##Step 3: return all the branches that include the specified child
        satisfy_branch = []
        
        for i, x in enumerate(method["subtasks"]):
            find=False
            for y in x:
                if y==child:
                    find=True
            #find the child in this branch, attach it into the satisfy_branch 
            if find==True:
                satisfy_branch.append([prob[i], x])        
        return satisfy_branch    

       
    def my_create_new_node(self, parent, decompose, childTree):
        newTree = Tree()
        parent_data = Node_data(complete=False, ready = True, branch = True)
        newTree.create_node(parent, parent, data=parent_data)
        
        known_child = childTree.get_node(childTree.root)
        for x in decompose[1]:
            if x==known_child.tag:
                known_child.data._pre = decompose[1][x]["pre"]
                known_child.data._dec = decompose[1][x]["dec"]
                newTree.paste(newTree.root, childTree)
            else:
                mydata = Node_data(pre=decompose[1][x]["pre"], dec=decompose[1][x]["dec"])
                newTree.create_node(x, x, parent=newTree.root, data= mydata)
        return [newTree, decompose[0]]   


#####################################################################################################################
#####               generate the pending set for each explanation based on the current tree structure           #####
#####################################################################################################################

    def create_pendingSet(self):      
        if len(self._pendingSet)==0:
            self.set_pendingSet(self.real_create_pendingSet())
        else:
            self.set_pendingSet(self.normalize_pendingSet_prior())
                 
    def real_create_pendingSet(self):
        pendingSet = set()
        for taskNet in self._forest:
            for taskNetPending in taskNet._pendingset:
                for action in taskNetPending._pending_actions:
                    pendingSet.add(action)
        
        #if currently the pending set has no action, need to initialize from start tasks
        if len(pendingSet)==0:
            for start_task in self._start_task:
                if self._start_task[start_task] == 0:
                    theMethod = db.find_method(start_task)
                    for y in theMethod["start_action"]:
                        pendingSet.add(y)
                        
        pendingSet1 = []
        prior = float(1)/len(pendingSet)
        for x in pendingSet:
            pendingSet1.append([x, self._prob*prior])
        
        return pendingSet1
        
        
    def normalize_pendingSet_prior(self):
        pendingSet =  self._pendingSet
        prior = float(1)/len(pendingSet)
        for x in pendingSet:
            x[1]=self._prob*prior
        
        return pendingSet 


##############################################################################################################
####                                    generate task hint in levels                                      ####
##############################################################################################################
    def generate_task_hint(self, taskhint):
        #########step1--------------------------------------
        ##get all node and their levels from this explanation's forest
        ##all those node should share the same prob.
        task_Name_Level = {}
        if len(self._forest)==0:
            task_Name_Level = {"nothing":[]}
        else:
            for taskNet in self._forest:
                for taskNetPending in taskNet._pendingset:
                    node_list = taskNetPending._tree.all_nodes()
                    
                    #only select nodes whose completeness is False, and readiness is True
                    node_list = [x for x in node_list if x.data._completeness==False and x.data._ready==True]
                    
                    for node in node_list:
                        level_num = taskNetPending._tree.depth(node)
                        if node._tag in task_Name_Level.keys():
                            level_list = task_Name_Level.get(node._tag)
                            level_list.append(level_num)
                            new_dict = {node._tag:level_list}
                            task_Name_Level.update(new_dict)
                        else:
                            level_list = []
                            level_list.append(level_num)
                            new_dict = {node._tag:level_list}
                            task_Name_Level.update(new_dict)
   
        ###########Step2----------------------------------------------
        ##add the task_Name_Level dict to the TaskHint 
        for k,v in task_Name_Level.items():
            taskhint.add_task(task_tag=k, expla_prob=self._prob, level = v)
                   
    
##############################################################################################################
#####                               Exception Handling Process                                           #####
##############################################################################################################

    def repair_expla(self, sensor_notification):
        update_belief_state = False
        belief_state_repair_summary = {}
        old_effect_summary = {}
        
        for taskNet in self._forest:
            repair_result = taskNet.repair_taskNet(sensor_notification)
            if repair_result[0] == True:
                new_effect_summary = copy.deepcopy(repair_result[1])
                old_effect_summary = copy.deepcopy(repair_result[2])
                self.belief_state_repair(belief_state_repair_summary, new_effect_summary, old_effect_summary)
                self.set_pendingSet(self.real_create_pendingSet())
                update_belief_state = True
                
        if update_belief_state is True:
            return [self._prob, belief_state_repair_summary]
        else:
            return [0.0, None]
       
            
    def belief_state_repair(self, belief_state_repair_summary, new_effect_summary, old_effect_summary):
        for key in old_effect_summary:
            if key in new_effect_summary:
                belief_state_repair_summary[key] = new_effect_summary[key]["value"]
            else:
                key_term = key.split("/")
                belief_state_repair_summary[key] = db.get_reverse_attribute_value(key_term[0], key_term[1], old_effect_summary[key]["value"])   
