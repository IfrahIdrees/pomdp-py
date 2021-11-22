"""------------------------------------------------------------------------------------------
Hierarchical Task Recognition and Planning in Smart Homes with Partially Observability
Author: Dan Wang danwangkoala@gmail.com (May 2016 - June 2017)
Supervised by Prof. Jesse Hoey (https://cs.uwaterloo.ca/~jhoey/)
Association: Computer Science, University of Waterloo.
Research purposes only. Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by AGEWELL Networks of Centers of Excellence (NCE).
----------------------------------------------------------------------------------------------"""


################################################################################################
####                    This file simulate the actions related to care-giver                ####
################################################################################################

#exit the agent tracking process due to sensor die
def call_for_caregiver_sensor_cause(bad_sensor, output_file_name):
    with open(output_file_name, 'a') as f:
        f.write("\n")
        f.write("The following sensor are not working: \n")
        for sensor in bad_sensor:
            new_line = sensor["object"] + "--------------" + sensor["attribute"]
            f.write(new_line)
        f.write("\n")
        f.write("The tracking process terminate for non-working sensors !")
        exit(0)
        
        new_line = "Simulate step happen:     " + step_name + "\n"
        f.write(new_line)    
    
    
    '''
    print "Some sensor is not working well now. Please fix them"
    print "The non-working sensors are : "
    for sensor in bad_sensor:
        print sensor["object"], "-----", sensor["attribute"]
    exit(0)
    '''
