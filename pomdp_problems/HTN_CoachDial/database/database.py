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
####          The DB_Object class. Provide database communication interfaces                       ####
####          Also refer to "Interface specification part III"                                     ####
#######################################################################################################
# // from pomdp_problems.HTNcoachDial.database cimport DB_Object
import sys
sys.dont_write_bytecode = True
from pymongo import MongoClient
import pymongo
import random
# client = MongoClient()
# db = client.smart_home3
import config

if config.baseline:
    client = MongoClient()
    db = client.smart_homebaseline
else:
    client = MongoClient()
    db = client.smart_home5
class DB_Object(object):
    def __init__(self):
        self._method = db.method
        self._operator = db.operator
        self._state = db.state
        self._sensor = db.sensor
        self._Rstate = db.Rstate
        # self._mcts_sensor = db.mcts_sensor
        self._backup_state = db.backup_state
        self._backup_sensor = db.backup_sensor
    ####################################################################################
    ####                        method related functions                            ####
    ####################################################################################    
    # Find all the method, and return as a list
    def find_all_method(self): 
        return list(self._method.find())
    
    # Find and return the specific method
    def find_method(self, m_name):
        method = list(self._method.find({"m_name":m_name}))
        if len(method)==0:
            return None
        else:
            return method[0]
    
    ##Find and return the start actions of the given method
    ##because the given method is the tag of tree root, it must be a goal 
    def get_start_action(self, m_name):
        method = list(self._method.find({"m_name":m_name}))
        return method[0]["start_action"]
    
    ####################################################################################
    ####                        operator related functions                          ####
    ####################################################################################
    
    ## Find and return the specific action  
    def get_operator(self, op_name):
        op = list(self._operator.find({"st_name":op_name}))
        return op[0]
    '''
    # The effect is store in json format like this
    # {"object_name": {"attribute_name": {"value": v, "step_name": sn}}, 
    #   }
        
    def get_operator_effect(self, op_name):
        op = self.get_operator(op_name)
        effect_summary = {}
        for obj in op["effect"]:
            for att in op["effect"][obj]:
                new_item_att = {}
                new_item = {}
                new_item["value"] = op["effect"][obj][att]
                new_item["step_name"] = op_name
                new_item_att[att] = new_item
                effect_summary[obj] = new_item_att
        return effect_summary 
     '''   
        
    ####################################################################################
    ####                 Belief state related functions                             ####
    ####################################################################################
    
    # Find and return the state of a specific object
    def get_object_status(self, ob_name): 
        ob = list(self._state.find({"ob_name":ob_name}))
        return ob[0]
    
    # Find and return the all possible values for a specific attribute of an object     
    def get_object_attri(self, ob_name, attri_name):
        st = list(self._state.find({"ob_name":ob_name}))
        return (st[0][attri_name])        
         
    # Find and return the attribute value belief from belief state
    def get_attribute_prob(self, s, ob_name, attri_name):
        print("inside database get attribute prob", ob_name, attri_name)
        st = list(self._state.find({"ob_name":ob_name}))
        return float(st[0][attri_name][s])
    
    ##Find and return the attribute value belief from belief state
    ##according to method/operator's precondition!!!!!!!!!!!!!
    ##The difference with get_attribute_prob is that here need to consider
    ##the "ability attribute"
    def get_attribute_prob_1(self, s, ob_name, attri_name):
        st = list(self._state.find({"ob_name":ob_name}))
        if attri_name!="ability":
            return float(st[0][attri_name][s])
        else:
            for x in st[0][attri_name]:
                y=x.split(",")
                if self.ability_check(s, y) == True:
                    return st[0][attri_name][x]
                else:
                    return 1-st[0][attri_name][x]

    # get reverse value of a attribute
    def get_reverse_attribute_value(self, ob_name, attri_name, attri_value):
        ob = self.get_object_status(ob_name)
        for att_value in ob[attri_name]:
            if att_value != attri_value:
                return att_value


    #can only check >= scenario!!!!!!!!!
    def ability_check(self, precond, state):
        for i, x in enumerate(state):
            if i==0: continue
            if float(state[i])<float(precond[i]): return False       
        return True

            
    ####################################################################################
    ####                 Sensor reading related functions                           ####
    ####################################################################################
    
    # Find and return the prob for p(obs|s) ##@@This is p(obs|sensor), prob of sensor notification| state, dependant on realiability, read from sensor_0.99 json
    def get_obs_prob(self, s, ob_name, attri_name):
        sensor = list(self._sensor.find({"ob_name":ob_name, "attri_name":attri_name}))
        sensor = sensor[0]
        if sensor["reliability"] == -1.0:   #this means the sensor is missing
            return 0.5
        elif sensor["value"][0]==s:
            return sensor["reliability"]
        else:
            return (1-sensor["reliability"])/(sensor["value"][1]-1)
    
    
    # Get sensor reliability information
    def get_sensor_reliability(self, ob_name, attri_name):
        sensor = list(self._sensor.find({"ob_name":ob_name, "attri_name":attri_name}))
        sensor = sensor[0]
        if sensor["reliability"] == -1.0:
            return 0.5
        else:
            return sensor["reliability"]
    

    # Update sensor value according to the desired update and the sensor reliability
    # the returned "label" tells if the sensor state is updated. 
    # "False": No, "True": Yes
    def update_sensor_value(self, ob_name, attri_name, value, real_step = False):
        label = False
        # if real_step:
        #     sensor = list(self._sensor.find({"ob_name":ob_name, "attri_name":attri_name}))
        # else:
        #     sensor = list(self._mcts_sensor.find({"ob_name":ob_name, "attri_name":attri_name}))
        
        sensor = list(self._sensor.find({"ob_name":ob_name, "attri_name":attri_name}))
        if len(sensor)!=1:
            print("inside udpate_sensor_value, the number of target ob_name is bad", len(sensor))
            sys.exit(0)
        
        elif sensor[0]["reliability"] == -1.0: ##in this case the sensor is missing just return False
            print("This is an missing sensor")
            return label
        else:
            sensor = sensor[0]
            if real_step:
                randomN = config.randomNs[config.realRandomIndex]
                config.realRandomIndex+=1
            else:
                randomN = config.randomNs[config.randomIndex]
                config.randomIndex+=1
            # print("randomN:",randomN, sensor["reliability"])
            if real_step:
                with open("random_no.txt", 'a') as f:
                    f.write(str(randomN)+'\n')

            if(randomN<=sensor["reliability"]):
                label = True
                valueNum = sensor["value"][1]
                result = self._sensor.update_many(
                    {"ob_name":ob_name, "attri_name":attri_name},
                    {
                        "$set":{
                            "value":[value, valueNum]
                        }
                    
                    }
                )
                
                # newsensor = list(self._sensor.find({"ob_name":ob_name, "attri_name":attri_name}))
                
        return label 
    ####################################################################################
    ####                 Parent node search                                         ####
    ####################################################################################

    # Find and return the parent list
    # Firstly search in the method collection
    # If not find, search the operator collection
    def get_parent_list(self, name):
        #step1: search the method collection
        parent = list(self._method.find({"m_name":name}))
        #step2: search the operator collections
        if len(parent)==0:
            parent = list(self._operator.find({"st_name":name}))
        if len(parent)==0:
            return False
        return parent[0]["parent"]
        
        
    ####################################################################################
    ####                 Belief state update related functions                      ####
    ####################################################################################    
    
    # Update belief state            
    def update_state_belief(self, ob_name, attri_name, attri_distri):
        '''
        print "inside database.py, update_state_belief:    ", ob_name, attri_name
        thestate =list (self._state.find({"ob_name":ob_name}))
        print "before state update, the distribution is", thestate[0]
        '''
        result = self._state.update_many(
            {"ob_name":ob_name},
            {
                "$set":{
                    attri_name:attri_distri
                }
            }
        )
        
        '''
        thestate =list (self._state.find({"ob_name":ob_name}))
        print "after state update, the distribution is", thestate[0]
        '''
        #print "then number of changes is", result.matched_count
        # print "after state update, the distribution is", thestate[0]
    def get_attri_distribution(self, ob_name, attri_name):
        object_state = list(self._state.find({"ob_name": ob_name}))
        object_state = object_state[0]
        return object_state[attri_name]
    
    def update_state_belief_for_exception(self, ob_name, attri_name, attri_value):
        reliability = self.get_sensor_reliability(ob_name, attri_name)
        old_distribution = self.get_attri_distribution(ob_name, attri_name)
        for value in old_distribution:
            if value == attri_value:
                old_distribution[value] = reliability
            else:
                old_distribution[value] = 1 - reliability      
        self.update_state_belief(ob_name, attri_name, old_distribution)
    ####################################################################################
    ####                 Real state update related functions                        ####
    ####################################################################################
    
    # Get the object real state
    def get_obj_Rstate(self, ob_name):
        objList = list(self._Rstate.find({"ob_name":ob_name}))
        if len(objList)!=1:
            print("inside get_obj_Rstate, the number of target ob_name is bad", len(objList))
            sys.exit(0)
        else:
            return objList[0]
            
    # Update the object real state given the ob_name, attri_name, and attri_value        
    def update_obj_Rstate(self, ob_name, attri_name, attri_value):
        result = self._Rstate.update_many(
            {"ob_name":ob_name},
            {
                "$set":{
                    attri_name:attri_value
                }
            }
        )
        
    def language_update_sensor_value(self, ob_name, attri_name, value):
        label = False
        sensor = list(self._sensor.find({"ob_name":ob_name, "attri_name":attri_name}))
        if len(sensor)!=1:
            print("inside udpate_sensor_value, the number of target ob_name is bad", len(sensor))
            sys.exit(0)
        
        elif sensor[0]["reliability"] == -1.0: ##in this case the sensor is missing just return False
            print("This is an missing sensor")
            return label
        else:
            sensor = sensor[0]
            # print(random.getstate())
            # config.seed+=10
            # random.seed(config.seed) 
            label = True
            valueNum = sensor["value"][1]
            result = self._sensor.update_many(
                {"ob_name":ob_name, "attri_name":attri_name},
                {
                    "$set":{
                        "value":[value, valueNum]
                    }
                
                }
            )
                
                #newsensor = list(self._sensor.find({"ob_name":ob_name, "attri_name":attri_name}))
                
        return label 