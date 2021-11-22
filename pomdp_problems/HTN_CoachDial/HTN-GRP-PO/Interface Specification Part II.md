"""
Hierarchical Task Recognition and Planning in Smart Homes with Partially Observability
Author: Dan Wang danwangkoala@gmail.com (May 2016 - June 2017)
Supervised by Prof. Jesse Hoey (https://cs.uwaterloo.ca/~jhoey/)
Association: Computer Science, University of Waterloo.
Research purposes only. Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by AGEWELL Networks of Centers of Excellence (NCE).
"""



Interface Specification Part II
===============================



Checklist
---------

 - main.py
 - **Class** Tracking_Engine

main.py
-------

 - the entrance of the algorithm, create an instance of a **Tracking_Engine**, and start it. 
 - specify parameters:
	 - *no_notif_trigger_porb*: without notification, the probability of go through the whole updating procedures
	 - *interval*: sleep interval between two running of the algorithm
	 - *cond_satisfy*, cond_notsatisfy: specify conditional probability p(s_t | s_t-1, a_t)
	 - *delete_trigger*: the threshold that an explanation should be dropped. 

**Class** Tracking_Engine
---------------------

 - `def` init(self, no_trigger = 0, sleep_interval = 1, cond_satisfy=1.0, cond_notsatisfy = 0.0, delete_trigger = 0.001)
	 - Initialize necessary parameters
 - `def` start(self):
	 - while loop forever
	 - case 1: has notification, go through the tracking engine algorithm
	 - case 2: has no notification, go through the tracking engine with probability of "**no_trigger**", sleep with probability of "**1-no_trigger**"
	 - tracking engine algorithm procedures:
		 - create ***ExplaSet*** instance
		 - explaInitialize(): Initialize *ExplaSet*
		 - action_posterior(): calculate the posterior probability of an step has happened for steps in the pending set
		 - update_state_belief(): update the belief state
		 - explaSet_expand(): expand the ExplaSet
		 - pendingset_generate(): generate the new pendingSet
		 - task_prob_calculate():calculate the probability of an task should happen in the next step, this is used for prompt. 


**Class** ExplaSet
------------------

 - parameters: explaset, init_label, cond_satisfy, cond_notsatisfy, delete_trigger
 - `def` init(): used for create new ExplaSet intance
 

> **Basic operations**

 - `def` add_exp(e): add a new explanation to the explaset
 - `def` pop(): get and remove an explanation
 - `def` length(): get the current length of the explaset
 - `def` explaInitialize(): Initialize explaset, only happen once
	 - the purpose is to create an pending set.
	 - the set contains "start_action" in method instance
	 - need to remove duplication
	 - need to calculate prior prob (evenly distributed in this step)
	 - create an Explanation instance and append to explaSet
 - `def` print_explaSet(): print out the explaset, is used for testing
 - `def` normalize(): normalize the prob over all explanations in the explaSet

> **Step posterior calculation**

 - `def` action_posterior(): for each explanation in the explaSet, for each step in the pendingSet of an explanation, calculate the posterior prob of happening based on **s_t-1, obs_t**. 
 - `def` cal_posterior(step): calculate the posterior prob for an step.
	 - title: all **object-attribute** pairs occurs in the step's precondition
	 - enum: the enumeration over all possible values on all object_attribute pairs. 
 - `def` myDFS(): call realMyDFS, to get enum
 - `def` realMyDFS(): DFS logic
 - `def` variable_elim(enum, op, title): variable elimination implemention
 - `def` bayesian_expand(before, after, op, title): implementation of the equation I
 - `def` get_ps_actANDs_1(before, after, op, title):  calculate p(s|s_t-1, a_t) (the step has happened)
	 - check precondition
	 - check effect
 - `def` get_ps_actANDs_2(before, after): calculate p(s|s_t-1, a_t) (the step not happen)

>  **ExplaSet expand**

 - `def` explaSet_expand:
	 - get step level explanations (**see** action_level_explanation())
	 - for each step level explanation, generate new explanation and add it to explaset 
		 - If "nothing", create new explanation instance, need to update explanation.prob.
		 - else,  **see** Explanation.generate_new_expla(step))
 - `def` action_level_explanation(pendingset):
	 - calculate the prob of nothing happen
	 - normalize on something happen
	 - delete action with prob **smaller than** delete_trigger

>  **Generate Pending Set**

 - `def` pendingset_generate(): 
	 - normalize the explanations in the explaSet
	 - create the new pendingSet for each explanation (**see** Explanation.create_pendingSet())

>  **Calculate task probability**

 - `def` task_prob_calculate(): Calculate the probability of node in each explanation, output the possible prompt task name, probabilities of tasks, and the task's average level. (**See** Explanation.generate_task_hint())


**Class** Explanation
------------------
 - `def` init(): construction function. 
	 - _prob: the probability of this explanation
	 - _forest: the tree structures for this explanation, each element in _forest is a **TaskNet** (**see** class TaskNet)
	 - _pendingSet: the pending steps of this explanation
	 - _start_action: this is come from a "method" format "start_action" keyword

> **Basic Operations**

 - `def` set_prob(v):
 - `def` set_forest(forest):
 - `def` set_pendingSet(pendingset):
 - `def` add_tasknet_ele(tasknet):
 - `def` add_pendSet_ele(val):
 - `def` update_forest(forest):
 - `def` update_pendSet(pendingSet):
 

> **Generate new explanation**

 - `def` generate_new_expla(self, act_expla):
	 - input act_expla: step level explanation. That is, given the current explanation, if step "act_expla" happens, what are the new explanations when adding this new observation ("act_expla" happens)
	 - the result is a list: one explanation might be expanded into many
	 - if the input step (act_expla) does not exist in the pendingset of this explanation, trigger exception
	 - if the input step(act_expla) in the Explanation._start_action list, initialize a tree structure (**see** Explanation.initialize_tree_structure), and generate new explanation. 
	 - if the input step (act_expla) exist in the taskNet._pendingset._pending_actions, 
		 - generate new taskNet(**see** TaskNet.generate_new_taskNet())
		 - get the newforest (remove the current one, add the new one)
		 - create the new explanation
		 - should be careful about the case: when the new TaskNet has a "complete" status, need to reset this goal.
 - `def` initialize_tree_structure(action): given an step name, initialize all the tree structures using a bottom-up process. bottom-up tree structure construction
	 - need to check the precondition of a method(**see** Explanation.method_precond_check), return the list of satisfy branch [prob, branch_subtask]
	 - need to create new node (see my_create_new_node), for newly added node, need to initialize Node_data information.

> **Create Pending Set**

 - `def` create_pendingSet():
	 - when add an newly generate explanation, set the pendingSet as [], an exception is that explanations generated from "nothing"
	 - if len(Explanation._pendingSet)==0, go to (**see**) Explanation.real_create_pendingSet()
	 - if len(Explanation._pendingSet)>0 , it means this explanation comes from "nothing", just normalize the priors of steps in the pending set. **See** normalize_pendingSet_prior().
 - `def` real_create_pendingSet() :
	 - using python set() to remove duplication
	 - steps comes from _forest->taskNet->taskNet._pendingset->_pending_actions **and** _start_acction
 - `def` normalize_pendingSet_prior():normalize priors for steps, the sum should be the Explanation._prob
 - `def` generate_task_hint(): calculate the prob for all the node in _forest->taskNet->taskNet._pendingset._tree
	 - Get all node
	 - Select nodes whose completeness is False, and readiness is True
	 - for each node, generate a [task_name, level] pair, and store in task_Name_Level
	 - each element in the task_Name_Level into taskhint (**see** class TaskHint)


**Class** Node_data
-------------------
 - `def` init()
	 - _completeness: complete or not
	 - _ready: 
	 - _pre:
	 - _dec:
	 - _branch:
 - `def` print_property():



**Class** TaskNet
-------------------
 - Each element in the explanation's forest is a TaskNet
 - `def` init(goalName="", tree=Tree(), expandProb = 1, pendingset=TaskNetPendingSet(), complete = False):
	 - _goalName
	 - _tree
	 - _expandPro
	 - _pendingset: each element is an **TaskNetPendingSet**
	 - _complete
 - `def` update():
	 - update the completeness information for all nodes in the tree, by searching on the tree. **See** TaskNet.complete_update()
	 - update the readiness information for all nodes in the tree, by searching on the tree. **See** TaskNet.readiness_update()
	 - update the pendingset for this tasknet, **see** TaskNet.pendingset_update
 - `def` complete_update(node, tree): recursion, 
 - `def` readiness_update(root_id, tree):from top to bottom, BFS
 - `def` pendingset_update(node, tree):
	 - input: the current TaskNet tree structure
	 - output: a list of **TaskNetPendingSet**
	 - decompose tree structure using method_decompose(**see** TaskNet.method_decompose)
	 - add the decompose result into the tree structrue. (See TaskNet.add_child(tree, node_id, branch))
 - `def` method_decompose(method): need to check the "method" knowledge base
 - `def` add_child(tree, node_id, branch): when adding a new node, need to initialize the node data


**Class** TaskNetPendingSet
-------------------
 - `def` init(tree = Tree(), branch_factor = 1, pending_actions = []):
	 - _tree:
	 - _branch_factor: newly added prob when decompose the tree structure of the corresponding TaskNet
	 - _pending_actions:
 - `def` generate_new_taskNet(action):  the action exist in the pending_actions of the TaskNetPendingSet, and now this action has happened. generate a new TaskNet based on this decomposition.
	 - update corresponding not completeness information
	 - create a new TaskNet instance
	 - **update** the newTaskNet

**Class** State
---------------

 - `def` init(cons_satisfy, cond_notsatisfy):
 - `def` update_state_belief(): update the belief state according the posterior prob of steps in the current pending set, previous belief state. Using the operators knowledge base
	 - Get all the attribute in the effect list of all steps in the current pending set. (see State.get_attr_in_effect())
	 -  Update the attribute value distribution. (see update_attri_status_belief())
 - `def` get_attr_in_effect():
	 - action_list: need to accumulate the prob if a step occurs in many explanations' pending set
	 - my_set: using python Set() to remove duplication
	 - title: [ob_name, attribute_name]
	 - return value [action_list, title]
 - `def` update_attri_status_belief(att, index, action_list, title): update the attribute status belief for a specific attribute (Based on Equation 2)
 - `def` get_ps_actANDs(after, before, action, index, title): calculate p(s|s_t-1, a_t) happen

**Class** TaskHint
--------------

 - Will provide API for prompt display module, can provide task name, task probability and the average task level information
 - One Tracking_Engine instance only have one TaskHint
 - Parameters: prompt_task={}
 - `def` reset(): reset prompt_task to empty dict
 - `def` add_task(): add a task information into the dict
	 - if task.tag exist: sum prob, and append the level list
	 - if task.tag not exist: create new item in the dict
 - `def` average_level(): calculate the average level for each task in the dict
 - `def` print_taskhint(): for testing. Can be changed into API in the future.

**Class** helper
------------

 - helper functions used in the tracking engine algorithm
 - `def` my_normalize(act_expla): normalize on pending set
 - `def` my_normalize_1(prob): normalize on a list of prob
 - `def` compare_ability(ab1, pre_ab2): used to compare if the person's ability is good enough to implement a task or step
 - `def` no_less_than: used to explain ">="
 - `def` list_average(): get the average of a list

