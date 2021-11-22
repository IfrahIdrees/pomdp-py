"""------------------------------------------------------------------------------------------
Hierarchical Task Recognition and Planning in Smart Homes with Partially Observability
Author: Dan Wang danwangkoala@gmail.com (May 2016 - June 2017)
Supervised by Prof. Jesse Hoey (https://cs.uwaterloo.ca/~jhoey/)
Association: Computer Science, University of Waterloo.
Research purposes only. Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by AGEWELL Networks of Centers of Excellence (NCE).
----------------------------------------------------------------------------------------------"""

###############################################################################################################
#####           This file contains functions that manually check if a sensor is working or not.           #####
#####           Those functions are simulations of real sensor checking process                           #####
###############################################################################################################


import sys
sys.dont_write_bytecode = True
from database import *

db = DB_Object()

# To check if all the sensors related to the operator's effect are working well
# To simulate the physical sensor checking, the sensor is defined as working if 
# the sensor reliability if bigger than 0.8
# How to set the threshhold matters a lot.
# Output: {"bad_sensor": [], "sensor_cause": True / False}
def operator_sensor_check(op):
    bad_sensor = []
    for obj in op["effect"]:
        for att in op["effect"][obj]:
            reliability = db.get_sensor_reliability(obj, att)
            # if reliability != 0.5 and reliability < 0.8: 
            if reliability < 0.5: 
                new_bad_sensor = {}
                new_bad_sensor["object"] = obj
                new_bad_sensor["attribute"] = att
                bad_sensor.append(new_bad_sensor)
                
    sensor_cause = {}
    sensor_cause["bad_sensor"] = bad_sensor
    if len(bad_sensor) > 0:
        sensor_cause["sensor_cause"] = True
    else:
        sensor_cause["sensor_cause"] = False
    return sensor_cause
