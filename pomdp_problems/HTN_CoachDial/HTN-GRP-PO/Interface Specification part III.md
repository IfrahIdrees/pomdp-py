"""
Hierarchical Task Recognition and Planning in Smart Homes with Partially Observability
Author: Dan Wang danwangkoala@gmail.com (May 2016 - June 2017)
Supervised by Prof. Jesse Hoey (https://cs.uwaterloo.ca/~jhoey/)
Association: Computer Science, University of Waterloo.
Research purposes only. Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by AGEWELL Networks of Centers of Excellence (NCE).
"""





Interface Specification part III
================================

**Operator** collection
-----------------------
**Example**

    {
	    "type":"step",
	    "st_name":"use_soap",
	    "precondition":{
	        "hand_1":{
	            "soapy":"no",
	            "dirty":"yes"  
	        },
	        "person_1":{
	            "location":"kitchen",
	            "ability":[">=", "0", "0", "0"]
	        }
	    },
	    "effect":{
	        "hand_1":{
	            "soapy":"yes",
	            "dry":"no"  
	        }
	    },
	    "parent":["clean_hand"]
	}

**Format**

 - "type": "step" -->indicates this is a step
 - "st_name":"use_soap" --> indicates the name of the step
 - "precondition": -->indicates the precondition of the step
	 - "hand_1": -->the object name
	 - "soapy":"no" -->the attribute name and the required value 
 - "effect":-->indicates the effects of the step
 - "parent":["clean_hand"]-->indicate the parent of this step. It's parent should be a list.


**Method** collection
---------------------

**Example**


    {
	    "type":"method",
	    "m_name":"clean_hand",
	    "precondition":[{
	        "hand_1":{
	            "dirty":"yes",
	            "soapy":"no"  
	        },
	        "faucet_1":{
	            "state":"on"
	        },
	        "person_1":{
	            "location":"kitchen",
	            "ability":[">=", "0", "0", "0"]
	        }
	    }],
	    "subtasks":[{
	        "use_soap":{
	            "pre":[],
	            "dec":["rinse_hand"]
	        },
	        "rinse_hand":{
	            "pre":["use_soap"],
	            "dec":[]
	        }
	    }],
	    "parent":["wash_hand"],
	    "start_action":[]
    }

**Format**

 - "type":"method"--> this is a method (for tasks)
 - "m_name":"clean_hand"--> the task name is "clean_hand"
 - "precondition":[{1}, {2}] -->the method precondition, a **list**
	 - each element in the list is a branch. This means the method can be decomposed into different sub-tasks under different condition.
	 - in this example, only one branch
 - "subtasks":[{1},{2}] -->subtasks branches
	 - in this example only one branch
	 - "use_soap": the subtask name
	 - "pre": indicate the predecessors of "use_soap" within subtasks
	 - "dec": the successor of "use_soap" within subtasks
 - "parent":[], the parent tasks of "clean_hand", a **list**
 - "start_action":[]--> if this method stands for a goal, the list should **not** have a length of 0, otherwise it is length 0 list.

**State** collection ##@II initial state of the user probablity distribution.
--------------------

**Example**

    {
	    "ob_name":"hand_1",
	    "ob_type":"hand",
	    "soapy":{
	        "yes":0.001,
	        "no":0.999
	    },
	    "dirty":{
	        "yes":0.999,
	        "no":0.001
	    },
	    "dry":{
	        "yes":0.999,
	        "no":0.001
	    }
	}

**Format**

 - "ob_name":object name
 - "ob_type": object type
 - "soapy": attribute name
	 - "yes": 0.001 --> attribute value and prob
	 - should cover all possible values of attribute "soapy"
 - "dirty": attribute name
 - "dry":attribute name
 - (Need to list all the attributes related to this object) 

**Class** DB_object
-------------------

 - `def` init():
	 - ._method: refer to the method collection
	 - ._operator: refer to the operator collection
	 - ._state: refer to the state collection
	 - ._sensor: refer to the sensor collection
	 

> **Method related**

 - `def`  find_all_method(): return all the method in the collection
 - `def` find_method(m_name): 
	 -  input: method name
	 -  output: 
		 - the specific method in the collection if exist
		 - None, if not exist
 - `def` get_start_action(m_name):
	 - input:m_name
	 - output:
		 - if exist, return the "start_action" value of the find method, a **list**
		 - if not exist, None 
		 

> **Operator related**

 - def get_operator(op_name):
	 - input: op_name
	 - output:
		 - if exist, return the find operator
		 - if not exist, return None
 - def get_object_status(ob_name):
	 - input: ob_name
	 - output:
		 - if exist, return the find state item
		 - if not exist, return None
 - def get_object_attri(ob_name, attri_name):
	 - input: ob_name, attri_name
	 - output:
		 - if exist, return the specific attribute value distribution
		 - if not exist, return None
 - def get_attribute_prob(value, ob_name, attri_name):
	 - input: attribute value, ob_name, attri_name
	 - output:
		 - if exist, return the find probability for the attribute value
		 - if not exist, return None
 - def get_attribute_prob_1(value, ob_name, attri_name):
	 - input: attribute value, ob_name, attri_name
	 - output: 
		 - if exist:
			 - if the attri_name == "ability"
				 - the input "value" is an required ability to completing a step or task, need to check if the current ability of the person match this ability. 
				 - if match, return the probability directly.
				 - if not match, return 1-prob. 
			 - if attri_name !="ability"
				 - return the find probability for the attribute value
		 - if not exist, return None


>**Other**

 - def get_parent_list(name):
	 - input: at step name or a task name
	 - output: 
		 - if exist, return the "parent" value of the input name
		 - if not exist, return None
