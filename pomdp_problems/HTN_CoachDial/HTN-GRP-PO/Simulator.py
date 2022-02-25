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
#####                           Simulate the state change in a real environment                           #####
###############################################################################################################


import sys
sys.dont_write_bytecode = True
from database import *
from helper import *
db = DB_Object()


##given the happened step, update the realState in database
def realStateANDSensorUpdate(step_name, output_file_name, real_step = False):
    if real_step:
        # print("Simulate step: ", step_name)
        if step_name == "dry_hand":
            a=1
    else:
        if step_name == "close_tea_box":
            a=1
        # print("MCTS Simulate step: ", step_name)

    with open(output_file_name, 'a') as f:
        #version changed in March 14, generate a table
        f.write("Notif: "+step_name + "\t")
    sensor_notification = []
    op = db.get_operator(step_name)
    effect = op["effect"]
    # print("op is", op, "effect is", effect)
    for obj in effect:
        for att in effect[obj]:
            if real_step:
                db.update_obj_Rstate(obj, att, effect[obj][att])
            # else 
            update_result = db.update_sensor_value(obj, att, effect[obj][att], real_step)

            if update_result == True:
                new_item = {}
                new_item["object"] = obj
                new_item["attribute"] = att
                new_item["obj_att_value"] = effect[obj][att]
                sensor_notification.append(new_item)
            # print("inside for loop")
    return sensor_notification

