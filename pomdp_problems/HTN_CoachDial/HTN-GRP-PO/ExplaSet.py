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
####                    The ExplaSet class                                                         ####
####                    Also refer to "Interface specification part II"                            ####
#######################################################################################################    

import sys
sys.dont_write_bytecode = True
import copy
from collections import deque
from database import *
# from pomdp_problems.HTNcoachDial.database import *
from Explanation import *
from helper import *
from TaskHint import *
from SensorCheck import *
from CareGiver import *
import math

#from __future__ import print_function  # Only needed for Python 2

db = DB_Object()

class explaSet(object):
    def __init__(self, cond_satisfy = 1.0, cond_notsatisfy = 0.0, delete_trigger = 0.001, non_happen = 0.0001, output_file_name = "Case4.txt", mcts_output_filename = "mcts_Case4.txt"):
        self._cond_satisfy = cond_satisfy
        self._cond_notsatisfy = cond_notsatisfy
        self._delete_trigger = delete_trigger
        self._explaset = deque([])
        self._action_posterior_prob = {}
        self._non_happen = non_happen
        self._sensor_notification = []
        self._output_file_name = output_file_name
        self._prior = {}
        self._language_notification = []
        self.highest_action_PS = []
        self.otherHappen = None
        self._mcts_output_filename = mcts_output_filename
        self.mcts_step_index = -1
    
    ##################################################################################################    
    ####                                        Part I                                           #####
    ####                Basic function about the explanation set. Include:                       #####
    ####                add, pop, length, get, set parameter value, normalize, print             #####
    ##################################################################################################
    
    def add_exp(self, e):
        if e._prob > self._delete_trigger:
            self._explaset.append(e)
    
    def pop(self):
        #get an explanation and remove it
        return self._explaset.popleft()
    
    def length(self):
        return len(self._explaset)
        
    def get(self, index):
        return self._explaset[len(self._explaset) - index] 
           
    def setSensorNotification(self, sensor_notification):
        self._sensor_notification = copy.deepcopy(sensor_notification)

    def setLanguageNotification(self, language_notification):
        self._language_notification = copy.deepcopy(language_notification)
        
    # Functionality: remove explanations with prob smaller than delete_trigger
    # Normalize the remaining explanations
    def normalize(self):
        leng = len(self._explaset)
        my_sum=0
        for x in self._explaset:
            my_sum=my_sum+x._prob         
        if my_sum == 0.0:
            return       
        for x in self._explaset:
            x._prob = x._prob/my_sum   
    
    
    def print_explaSet(self):
        with open(self._output_file_name, 'a') as f:
            f.write(str(len(self._explaset)) + "\n")
            #f.write('{:>12}'.format(str(len(self._explaset))))
            #f.write('\n')

    def mcts_print_explaSet(self):
        with open(self._mcts_output_filename, 'a') as f:
            f.write(str(len(self._explaset)) + "\n")
    
    # write the explanation into a .txt file
    def print_explaSet1(self):
        
        with open(self._output_file_name, 'a') as f:
            new_line = "Explanation Number:  " + str(len(self._explaset)) + "\n"
            f.write(new_line)
        
            for index in range(len(self._explaset)):
                x = self._explaset[index]
                new_line = "\n" + "--------------Explanation " + str(index+1) + "------------------\n"
                f.write(new_line)
                
                new_line = '{:>30} {:>12}'.format("The probability: ", round(x._prob, 4))
                f.write(new_line)
                f.write("\n")
                
                new_line = '{:>30} {:>12}'.format("The current pending set is: ", x._pendingSet)
                f.write(new_line)
                f.write("\n")
                
                new_line = '{:>30} {:>12}'.format("The tasks ongoing are: ", x._start_task)
                f.write(new_line)
                f.write("\n")

                for y in x._forest:
                    f.write("\n")
                    new_line = '{:>30} {:<12}'.format("Goal Name: ", y._goalName)
                    f.write(new_line)
                    f.write("\n")
                    
                    for actions in y._pendingset:
                        new_line = '{:>30} {:>12}'.format("The pendingSet for this Goal: ", actions._pending_actions)
                        f.write(new_line)
                        f.write("\n")
            f.write("\n")
        '''  
            for index in range(len(self._explaset)):
                x = self._explaset[index]
                new_line = "\n" + "--------------Explanation " + str(index+1) + "------------------\n"
                f.write(new_line)
                
                new_line = '{:>30} {:>12}'.format("The probability: ", round(x._prob, 4))
                f.write(new_line)
                f.write("\n")
                
                new_line = '{:>30} {:>12}'.format("The current pending set is: ", x._pendingSet)
                f.write(new_line)
                f.write("\n")
                
                new_line = '{:>30} {:>12}'.format("The tasks ongoing are: ", x._start_task)
                f.write(new_line)
                f.write("\n")

                for y in x._forest:
                    f.write("\n")
                    new_line = '{:>30} {:<12}'.format("Goal Name: ", y._goalName)
                    f.write(new_line)
                    f.write("\n")
                    
                    for actions in y._pendingset:
                        new_line = '{:>30} {:>12}'.format("The pendingSet for this Goal: ", actions._pending_actions)
                        f.write(new_line)
                        f.write("\n")
            f.write("\n")      
        '''
    ##################################################################################################    
    ####                                        Part II                                          #####
    ####                Explanation Set initialization, especially initialize pendingSet         #####
    ####                This part is only executed when start the agent                          #####
    ##################################################################################################
    
    def explaInitialize(self):
        #initialzie the explanation. At very beginning, the explanation is "nothing happend"
        #It's pendingSet is all the start actions of all the method in the database
        goal = db.find_all_method()
        mypendingSet=[]
        mystart_task = {}
        candicatePendingSet = {}
       
        goalNum = 0
        for x in goal:
            if len(x["start_action"])>0:
                goalNum = goalNum + 1
                mystart_task[x["m_name"]] = 0
        goalPrior = float(1) / goalNum
        for x in goal:
            if len(x["start_action"])>0:
                mystart_task[x["m_name"]] = 0 #this task has not started yet
                
                for y in x["start_action"]:
                    if y not in candicatePendingSet:
                        candicatePendingSet[y] = goalPrior*float(1)/len(x["start_action"])
                    else:
                        candicatePendingSet[y] = candicatePendingSet[y] + goalPrior*float(1)/len(x["start_action"]) 
        
        
        for x in candicatePendingSet:
            mypendingSet.append([x, candicatePendingSet[x]])
        #nothing has happened with prob 1
        exp = Explanation(v=1, pendingSet=mypendingSet, start_task = mystart_task)
        self._explaset.append(exp)
        
    ##################################################################################################    
    ####                                        Part III                                         #####
    ####                Calculate posterior probability of actions in the pendingSet             #####
    ####                Also refer to "Step recognition.md"                                      #####
    ##################################################################################################
    def update_prior(self):
        self._action_posterior_prob = {}
        for expla in self._explaset:           
            for action in expla._pendingSet:
                if action[0] in self._action_posterior_prob:
                    self._action_posterior_prob[action[0]] = self._action_posterior_prob[action[0]] + action[1]
                    
                else:
                    self._action_posterior_prob[action[0]] = action[1]

        #---------------------------------
        '''   
            for start_task in expla._start_task:
                if expla._start_task[start_task] == 0:
                    #target_method is the highest level goal
                    target_method = db.find_method(start_task)
                    ### float so considering 0.
                    ##@II add added float but gave error so removed it.
                    initialize_prob = expla._prob / (len(expla._pendingSet) + len(target_method["start_action"]))
                    for start_action in target_method["start_action"]:
                        if start_action in self._action_posterior_prob:
                            self._action_posterior_prob[start_action] = self._action_posterior_prob[start_action]+initialize_prob
                        else:
                            ## when go to else?
                            self._action_posterior_prob[start_action] = initialize_prob
        '''
        #---------------------------------
        ## previous code sets prior over the nexxt pending task to be done p(at)
        # print("Start of iteration, prior is", self._action_posterior_prob, file=sys.stderr)
        self._prior = copy.deepcopy(self._action_posterior_prob)
            
    def action_posterior(self, execute=False):
        # #calaculate posterior action for every low_level action in pending set
        # # self._action_posterior_prob = {}
        # # otherHappen = 1
        # # for expla in self._explaset:           
        # #     for action in expla._pendingSet:
        # #         if action[0] in self._action_posterior_prob:
        # #             self._action_posterior_prob[action[0]] = self._action_posterior_prob[action[0]] + action[1]
                    
        # #         else:
        # #             self._action_posterior_prob[action[0]] = action[1]

        # # #---------------------------------
        # # #'''   
        # #     for start_task in expla._start_task:
        # #         if expla._start_task[start_task] == 0:
        # #             #target_method is the highest level goal
        # #             target_method = db.find_method(start_task)
        # #             ### float so considering 0.
        # #             ##@II add added float but gave error so removed it.
        # #             initialize_prob = expla._prob / (len(expla._pendingSet) + len(target_method["start_action"]))
        # #             for start_action in target_method["start_action"]:
        # #                 if start_action in self._action_posterior_prob:
        # #                     self._action_posterior_prob[start_action] = self._action_posterior_prob[start_action]+initialize_prob
        # #                 else:
        # #                     ## when go to else?
        # #                     self._action_posterior_prob[start_action] = initialize_prob
        # # #'''
        # # #---------------------------------
        # # ## previous code sets prior over the nexxt pending task to be done p(at)
        # # print("Start of iteration, prior is", self._action_posterior_prob)
        # # self._prior = copy.deepcopy(self._action_posterior_prob)
        otherHappen = 1
        self.update_prior()
        ##@II calculate the highest PS
        highest_action_PS = ["", float('-inf')]
        for k, v in self._action_posterior_prob.items():
            if v > highest_action_PS[1]:
                highest_action_PS = [k,v]

        self.highest_action_PS = highest_action_PS
        plang_st = 0
        observation_prob = {}
        for k in self._action_posterior_prob: 
            posteriorK, observation_prob = self.cal_posterior(k) #p(obs| st-1, at) since sum over st-1 so PosteriorK returns p(obst|at)
            
            ##@II posteriorK is the p(w_obst|PS) 
            ##calculate p(l_obs|PS)
            # plang_st = self.cal_lang_posterior(highest_action_PS, k)
            otherHappen = otherHappen - posteriorK * self._action_posterior_prob[k] #1 - priir*posteior, prob of other action happening  ## p(obs|at)*p(at)= p(at|ob) so other_happen = 1 -p(at|obs)
            self._action_posterior_prob[k] = self._action_posterior_prob[k] * posteriorK # p(at)*p(o|st-1,at) ## posterior p(at|ob) =  p(ob|at)*p(at)
            
            # plang_st = self.cal_lang_posterior(highest_action_PS, k)
            # otherHappen = otherHappen - plang_st * posteriorK * self._action_posterior_prob[k] #1 - priir*posteior, prob of other action happening  ## p(obs|at)*p(at)= p(at|ob) so other_happen = 1 -p(at|obs)
            # self._action_posterior_prob[k] = plang_st * self._action_posterior_prob[k] * posteriorK # p(at)*p(o|st-1,at) ## posterior p(at|ob) =  p(ob|at)*p(at)
            
            # print(k, self._action_posterior_prob[k], otherHappen)

        #nothing
        ##sensor, did you right now do this, yes (formulation ask question about current step)
        if execute:
            with open(self._output_file_name, 'a') as f:
                f.write(str(round(otherHappen, 4)) + "\t")
        else:
            with open(self._mcts_output_filename, 'a') as f:
                f.write(str(round(otherHappen, 4)) + "\t")

        ##@II
        self.otherHappen = otherHappen
        return otherHappen, observation_prob #high prob that language is correct
       
    def cal_lang_posterior(self, highest_action_PS, action):
        if action == highest_action_PS[0]:
            if 'yes' in self._language_notification:
                return 0.99
            else:
                return 0.01
        else:
            if 'yes' in self._language_notification:
                return 0.01
            else:
                return 0.99


    #version begin from March 14, 2017
    def cal_posterior(self, action):
        op = db.get_operator(action)
        
        objAttSet = set()
        for obj in op["precondition"]:
            for att in op["precondition"][obj]:
                objAttSet.add(obj+"-"+att) 
        for obj in op["effect"]:
            for att in op["effect"][obj]:
                objAttSet.add(obj + "-" + att)
                
        title = []
        for item in objAttSet:
            title.append(item.split("-"))
        #attribute is the corresponding attribute distribution in title, ## attribute includes prior prob of the state of sensors and observe includes p(obs| previousstate of sensor)
        attribute = []
        observe_prob = []
        observation_prob = {}
        ## add language in observe_prob

        for item in title:
            # get distribution for yes, no value of the attribute. This is prob for previous state of sensor
            # This is fetch from database state.json
            attri_distribute = db.get_object_attri(item[0], item[1])
            attribute.append(attri_distribute) ##initial state of the attribute of objects.
            observe_distribute = {}
            # now for every prior state of sensor calculate p(obs|prior state of sensor), this is dependant on the sensor realiabilty, sensor give ground truth with sensor_0.99json realibilty, groundth truth is stored in the databse
            for value in attri_distribute:
                observe_distribute[value] = db.get_obs_prob(value, item[0], item[1])
            ## attribute and ##observe just fetching from database. It is fixed initially.
            observe_prob.append(observe_distribute)
            observation_prob_key = "-".join(item)
            observation_prob[observation_prob_key] = observe_distribute
        enum = self.myDFS(attribute)
        new_prob=self.variable_elim(enum, op, title, attribute, observe_prob) #p(o|st-1,a)
   
        return new_prob, observation_prob
         
    ##dfs is used to generate the enumeration of all possible state combinations    
    def myDFS(self, attribute):
        enum = []
        va = []
        self.realMyDFS(enum, va, attribute)
        return enum
    
    def realMyDFS(self, enum, va, attribute):
        if len(va) == len(attribute):
            enum.append(list(va))
            return
        index = len(va)
        for x in attribute[index]:
            va.insert(index, x)
            self.realMyDFS(enum, va, attribute)
            va.pop()
                
    ##implement the bayesian network calculation for one possible state, # total prob is not 1.
    ##op: the operator in knowlege base, prob: the prior of the action, for a action in pending_set return probabilit of happening/(total prob)
    def variable_elim(self, enum, op, title, attribute, observe_prob):

        new_prob_1 = 0 #this action happened
        new_prob_2 = 0 #this action does not happend
        # print("------------------------------------ op ", op["st_name"])
        for ind, before in enumerate(enum):
            for ind2, after in enumerate(enum):
                # print("before", before, "after", after)
                # print(self.check_condition_met(op, before,title, "precondition"), self.check_condition_met(op, after,title, "effect") )
                p = self.bayesian_expand(before, after, op, title, attribute, observe_prob)
                # print(p)
                # print("--------------------------")
                new_prob_1 = new_prob_1 + p[0]
                new_prob_2 = new_prob_2 + p[1]
                # print(new_prob_1, new_prob_2)       
        # try:
        return float(new_prob_1)/(new_prob_1+new_prob_2) #p(obs| st-1, at) #p(turnonfaucet | nothing, turn_on_faucet) p(obser | nothing, switch on kettle)
        # except ZeroDivisionError:
            # return 0
        # return float(new_prob_1)
        # /(new_prob_1+new_prob_2) #p(obs| st-1, at) #p(turnonfaucet | nothing, turn_on_faucet) p(obser | nothing, switch on kettle)
        
        
    def check_condition_met(self, op, belief_state,title, state_position="effect"):
        match = True

        for obj in op[state_position]:
            for att in op[state_position][obj]:
                for i,conditions in enumerate(title):
                    if obj == conditions[0] and att == conditions[1]:
                        if att == "ability":
                            ability_list = belief_state[i].split(",")
                            match = match and compare_ability(ability_list, op[state_position][obj][att])
                            # print("ability", match)
                        else:
                            match = match and op[state_position][obj][att] == belief_state[i]

                    if match == False:
                        return match
        return match

    def get_language_obs_prob(self, match, po_s):
        if match and "yes" in self._language_notification:
            po_s *= 0.99
        elif match and "no" in self._language_notification:
            po_s *=0.01
        elif not match and "yes" in self._language_notification:
            po_s *=0.01
        elif not match and "no" in self._language_notification:
            po_s *=0.99

        return po_s
    #sv: an concrete state value, op: the operator in knowledge base
    #state_c: the notification        
    def bayesian_expand(self, before, after, op, title, attribute, observe_prob): 
        #calculate p(s_t-1), using attribute that has databse of prioir probablities
        ps_before = 1
        for i, s in enumerate(before):
            ps_before = ps_before * attribute[i][s]
            
        #calculate p(o_t|s_t), using observe prob
        po_s = 1
        for i, s in enumerate(after):
            po_s = po_s * observe_prob[i][s]
                
        ##@@II - multiply language observation into the observational model.
        # po_s_before = po_s
        # match = self.check_condition_met(op, after, title, "effect") & self.check_condition_met(op, before, title, "precondition")
        # po_s = self.get_language_obs_prob(match, po_s)
        # if match == True:
        #     print(match)
        # print(match,"yes" in self._language_notification ,po_s_before, po_s, po_s == po_s_before)
        #calculate p(s|s_t-1, a_t)
        ps_actANDs_1 = self.get_ps_actANDs_1(before, after, op, title) #change happens
        ps_actANDs_2 = self.get_ps_actANDs_2(before, after) #change does not happen
        # print("p(st-1)", ps_before, "po_s", "ps|s,a", ps_actANDs_1 )
        # print("p(st-1)", ps_before, "po_s", "ps|s,a'", ps_actANDs_2 )

        ##if the action is not taken p(s|st-1, a) and 
        # p(o == no |s, a') ==0.99
        # p(o ==yes |s,a') ==0.01
        # po_sa = (po_s_before * 0.99 if "no" in self._language_notification else po_s_before * 0.01)
        prob = []
        prob.append(float(ps_before)*po_s*ps_actANDs_1)
        prob.append(float(ps_before)*po_s*ps_actANDs_2)
        # print(prob)
        return prob

    #calculate p(s|s_t-1, a_t) happen    
    def get_ps_actANDs_1(self, before, after, op, title):   
        bef = list(before)
        af = list(after)
        
        #check precondition
        prob = 1
        precond = op["precondition"]
        for ob in precond:
            for attri in precond[ob]:
                index = title.index([ob, attri])
                if attri=="ability":
                    ability_list = bef[index].split(",")
                    if compare_ability(ability_list, precond[ob][attri]) is False:
                        print("return not satisfy because of ability not enough ")
                        return self._cond_notsatisfy
                else:
                    if precond[ob][attri]!=bef[index]:
                        return self._cond_notsatisfy
      
        ##check effect
        effect = op["effect"]
        for ob in effect:
            for attri in effect[ob]:
                index=title.index([ob, attri])
                bef[index]=effect[ob][attri]
        if bef!=af:  
            return self._cond_notsatisfy
            
        return self._cond_satisfy


    #calculate p(s|s_t-1, a_t) not happen
    def get_ps_actANDs_2(self, before, after):
        if before==after: return self._cond_satisfy
        else: return self._cond_notsatisfy        

    ##################################################################################################    
    ####                                        Part IV                                          #####
    ####            Expand the explanation Set                                                   #####
    ####            Each explanation can be extended into multiple explanations.                 #####
    ####            Based on which actions has happened                                          #####
    ##################################################################################################
 
    ##"explaSet_expand_part1" is used to generate explanations that add a new tree structure, a bottom-up process
    ##The bottom-up process depend on the previous state.     
    def explaSet_expand_part1(self, length):
        
        for i in range(length):
            x =   self.get(i+1) ##x is prior for the pending step
            # print("action posterior after bayseian inference is",  self._action_posterior_prob)
            for action in self._action_posterior_prob:
                #case1: nothing happened: update the prob of the explanation,do not need to update tree structure. 
                if action == "nothing":
                    newstart_task = copy.deepcopy(x._start_task)
                    newexpla = Explanation(v=x._prob*self._action_posterior_prob[action], forest = x._forest, pendingSet = x._pendingSet, start_task = newstart_task)
                    self.add_exp(newexpla)
                                 
                else:
                #case2:something happend, need to update the tree structure
                    new_explas = x.generate_new_expla_part1([action, self._action_posterior_prob[action]])
                    # given the prior of PS, for every explanation in the explanation set, and every action in prior/posterior(prior and posterior set does not change, 
                    # we generate explanation for every goal and the probablity associated with it)
                    ##@II this should get affected by the language prob
                    
                    for expla in new_explas:
                        self.add_exp(expla)  ## add if probability > 0.001 as in papers
                        
        
        return
        
    ##"explaSet_expand_part2" is used to generate explanations by decomposing an existing tree structure. , a top-down process
    ##The top=down process depends on the current state. (after adding the effect of just happened action)                
    def explaSet_expand_part2(self, length):
        for i in range(length):
            x =   self.pop()
            for action in self._action_posterior_prob:
                if action == "nothing":
                    continue
                new_explas = x.generate_new_expla_part2([action, self._action_posterior_prob[action]])
                for expla in new_explas:
                    self.add_exp(expla)          
        return
        
    def update_with_language_feedback(self, step, highest_action_PS, p_l):
        # for expla in self._explaset:
        #     expla
        weights = [0 for i in range(len(self._explaset))]
        for expla in self._explaset:
            # goal_prob = expla._prob
            correct_taskNets = 0
            
            for taskNet_ in expla._forest:
                # for taskNet_ in forest:
                ExecuteSequence =  taskNet_._execute_sequence._sequence
                if ExecuteSequence == []:
                    # taskNet_._expandProb *= 0.01
                    # expla._prob*=0.01
                    continue
                if highest_action_PS[0] == ExecuteSequence[-1] and step == 'yes':
                    correct_taskNets+=1
                    # taskNet_._expandProb *= 0.99 
                    # expla._prob*=0.99
                elif not (highest_action_PS[0] == ExecuteSequence[-1]) and step == 'yes':
                    # taskNet_._expandProb *= 0.01
                    # expla._prob*=0.01
                    continue
                elif step == 'no' and  not (highest_action_PS[0] == ExecuteSequence[-1]):
                    correct_taskNets+=1
                    # taskNet_._expandProb *= 0.99
                    # expla._prob*=0.99
                elif step == 'no' and  (highest_action_PS[0] == ExecuteSequence[-1]):
                    # taskNet_._expandProb *= 0.01
                    # expla._prob*=0.01
                    continue

                # taskNet_._expandProb *= p_l
                # expla._prob*=p_l
            delta = 0.001
            if len(expla._forest) == 0:
                weight = 0+delta
            else:
                weight = float(correct_taskNets)/len(expla._forest)+delta
            expla._prob*=weight*p_l #using tasknets as weightd for adjust probs of explanation prob. but after normalizing it is the same.


        return

                #         expla._pendingSet
                # if action[0] in self._action_posterior_prob:
                #     self._action_posterior_prob[action[0]] = self._action_posterior_prob[action[0]] + action[1]
                    
                # else:
                #     self._action_posterior_prob[action[0]] = action[1]
            # for ac


    def update_without_language_feedback(self, p_l):
        # for expla in self._explaset:
        #     expla

        for expla in self._explaset:
            # goal_prob = expla._prob
            # for taskNet_ in expla._forest:
                # for taskNet_ in forest:
                # ExecuteSequence =  taskNet_._execute_sequence._sequence
                # if highest_action_PS[0] in ExecuteSequence and step == 'Yes':
                #     taskNet_._expandProb *= 0.99 
                #     expla._prob*=0.99
                # elif not (highest_action_PS[0] in ExecuteSequence) and step == 'Yes':
                #     taskNet_._expandProb *= 0.01
                #     expla._prob*=0.01
                # elif step == 'No' and  not (highest_action_PS[0] in ExecuteSequence):
                #     taskNet_._expandProb *= 0.99
                #     expla._prob*=0.99
                # elif step == 'No' and  (highest_action_PS[0] in ExecuteSequence):
                #     taskNet_._expandProb *= 0.01
                #     expla._prob*=0.01

                # taskNet_._expandProb *= (1-p_l)
            expla._prob*= (1 - p_l)
            # expla._prob*= 1
        return
   
    ##################################################################################################    
    ####                                        Part V                                           #####
    ####            Generate the new pending set for each explanation                            #####  
    ####            based on the current tree structure and belief state                         #####
    ##################################################################################################
                 
    def pendingset_generate(self):
        self.normalize()
        for expla in self._explaset:
            expla.create_pendingSet()
           
    ##################################################################################################    
    ####                                        Part VI                                          #####
    ####            Calculate the probability of each node in the explanation                    #####
    ####            Output the probability of each tast and the average level                    #####
    ####            based on the current tree structure and belief state                         #####
    ##################################################################################################

    def task_prob_calculate(self, filename ):
        taskhint = TaskHint(self._output_file_name)
        taskhint.reset()
        for expla in self._explaset:
            expla.generate_task_hint(taskhint)

        taskhint.average_level()

        taskhint.print_taskhintInTable(filename)
        # print("taskhint", taskhint.__dict__)   
        return taskhint  

    ##################################################################################################    
    ####                                        Part VII                                         #####
    ####            Exception handling. This part is used when the probability of                #####
    ####            "otherHappen" is too high.                                                   #####
    ####            Exception handling can deal with (1)sensor die (2)wrong step                 #####
    ##################################################################################################
    def adjust_posterior(self):
        belief_state_repair_summary = {} #record to what degree the belief state should be updated

        for expla in self._explaset:
            expla_repair_result = expla.repair_last_expla(self._sensor_notification, self.highest_action_PS)
            if expla_repair_result[0] != 0.0:
                self.belief_state_repair_summary_extend(belief_state_repair_summary, expla_repair_result)
        
        self.belief_state_repair_execute(belief_state_repair_summary)

    def handle_exception(self):
        # print("into handle exception")
        sensor_cause = {}
        sensor_cause["bad_sensor"] = []
        sensor_cause["sensor_cause"] = False
        wrong_step_cause = False
        
        for step in self._action_posterior_prob:
            operator = db.get_operator(step)
            effect_length = get_effect_length(operator)
            if effect_length >= len(self._sensor_notification):
                this_sensor_cause = operator_sensor_check(operator)
                sensor_cause["bad_sensor"].extend(this_sensor_cause["bad_sensor"])
        
        # bad sensor cause the exception, call the caregiver to repair the sensors
        if len(sensor_cause["bad_sensor"]) > 0:
            print("bad sensor cause sensor exception")
            sensor_cause["sensor_cause"] = True
            call_for_caregiver_sensor_cause(sensor_cause["bad_sensor"], self._output_file_name)
        
        # wrong step cause the exception
        self.handle_wrong_step_exception()
        
    ####----------------------------------------------------------------------------------------#######    
    # the wrong step might violate soft_constraint or hard_constraint
    # 1. soft_constraint:   the wrong step does not impact on objects related to already happened steps' effects. 
    #                       In this case, the wrong step violate soft_constraint (logically wrong, e.g. rinse hand before use soap)
    #                       Do not update the current explanation, give the previous hint, do not update state belief,
    # 2. hard_constraint:   the wrong step does impact on objects related to already happened steps' effects
    #                       In this case, the wrong step violate hard_constraint (destroy the required preconditions for later steps)
    #                       Need to backtrack to the step that create the precondtions and start from that point. 
    #                       Need to update tree structure and the new pendingSet. Need to update belief state
    # 3. state_update:      Finally weather update the state depends the sum probability of update and no-update   
    def handle_wrong_step_exception(self):
        belief_state_repair_summary = {} #record to what degree the belief state should be updated
        
        for expla in self._explaset:
            expla_repair_result = expla.repair_expla(self._sensor_notification)
            if expla_repair_result[0] != 0.0:
                self.belief_state_repair_summary_extend(belief_state_repair_summary, expla_repair_result)
        
        self.belief_state_repair_execute(belief_state_repair_summary)
        
    # add the expla_repair_result into belief_state_repair_summary
    # expla_repair_result [prob, {key:value}]        
    def belief_state_repair_summary_extend(self, belief_state_repair_summary, expla_repair_result):
        for x in expla_repair_result[1]:
            newkey = x + "/" + expla_repair_result[1][x]
            if newkey in belief_state_repair_summary:
                belief_state_repair_summary[newkey] = belief_state_repair_summary[newkey] + expla_repair_result[0]
            else:
                belief_state_repair_summary[newkey] = expla_repair_result[0]

    def belief_state_repair_execute(self, belief_state_repair_summary):
        for effect in belief_state_repair_summary:
            if belief_state_repair_summary[effect] > 0.7:
                belief_state = effect.split("/")
                opposite_attri_value = db.get_reverse_attribute_value(belief_state[0], belief_state[1], belief_state[2])
                new_att_distribution = {}
                new_att_distribution[belief_state[2]] = belief_state_repair_summary[effect]
                new_att_distribution[opposite_attri_value] = 1 - belief_state_repair_summary[effect]
                
                db.update_state_belief(belief_state[0], belief_state[1], new_att_distribution)
        return
