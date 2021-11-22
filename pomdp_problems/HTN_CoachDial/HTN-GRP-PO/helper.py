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
####                        Some helper functions                                                  ####
####                        Also refer to "Interface specification part II"                        ####
#######################################################################################################

import sys
sys.dont_write_bytecode = True
from database import *
import json
db = DB_Object()


########################################################################################################
####                                    normalizing                                                #####
########################################################################################################



###normalize the probability of pending set[action_name, prob]
def my_normalize(act_expla):
    mysum = 0
    for x in act_expla:
        mysum=mysum+x[1]
    for x in act_expla:
        x[1]=x[1]/mysum
    return act_expla
    
##normatlize the probability p[p1, p2, p3]    
def my_normalize_1(prob):
    mysum=0
    for x in prob:
        mysum=mysum+x
    if mysum==0: return
    for i in range(len(prob)):
        prob[i] = float(prob[i]/mysum)
        
##to check if the precondition of a method is satisfied
## the return value is [[prob, [subtasks]]]
def compare_ability(ab1, pre_ab2):
    if pre_ab2[0] == ">=":
        return no_less_than(ab1, pre_ab2)
    return False
   
########################################################################################################
####                                    constraint                                                 #####
########################################################################################################

#constraint: no_less_than
def no_less_than(ab1, pre_ab2):
    for i, x in enumerate(ab1):
        if i==0: continue
        if float(ab1[i]) < float(pre_ab2[i]):
           return False
    return True 

##############################################################
#################return the average of a list###################
def list_average(mylist):
    length = len(mylist)
    if length==0:
        return -1
    mysum = 0
    for x in mylist:
        mysum = mysum+x
    ##the return value is an int now
    return mysum/length


####################################################################################
####                Get the length of effect of an operator                     ####
####################################################################################
# Input: an operator
# Output: the length of the effect of this operator
# Each attribute status change is an effect, rather each object. 
def get_effect_length(op):
    effect_length = 0
    for obj in op["effect"]:
        effect_length = effect_length + len(op["effect"][obj])
    return effect_length
