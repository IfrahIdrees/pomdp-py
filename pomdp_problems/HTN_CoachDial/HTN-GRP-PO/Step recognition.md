"""
Hierarchical Task Recognition and Planning in Smart Homes with Partially Observability
Author: Dan Wang danwangkoala@gmail.com (May 2016 - June 2017)
Supervised by Prof. Jesse Hoey (https://cs.uwaterloo.ca/~jhoey/)
Association: Computer Science, University of Waterloo.
Research purposes only. Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by AGEWELL Networks of Centers of Excellence (NCE).
"""




Step Recognition
================
Step recognition means given the **state change notification**, **knowledge base**, **and the last step environment state**,  identify the just happened step. This is a basic and essential procedure for both **goal recognition** and **on-going task tracking**. 

Assumption
----------

 - one state change notification includes one or more than one object's state change
 - one state change notification only related to one step


The method
----------

 - Step 1: QUERY **candidate steps** from the knowledge base using the first object's state change in the state change notification list.
 - Step 2: FILTER candidate steps by checking if len(state_change_notification) == len(step_effect_list)
 - Step 3: FILTER candidate steps by checking if step_effect_list contributes all state change. 
 - Step 4: FILTER candidate steps by checking if its precondition is satisfied in the *current state collection of the database*. **(When checking the precondition, "person": "ability" precondition should not be checked since this step already happened, it does not matter if the person's ability matches the step's ability requirement. )**
 - Step 5: UPDATE the **state collection** in the database according to state change notification. (**We update the state collection after recognizing the step because**: when doing step recognition, precondition need to be checked according to the last step's environment step. So delay the database updating makes this procedure convenient). 

