"""The classic Tiger problem.

This is a POMDP problem; Namely, it specifies both
the POMDP (i.e. state, action, observation space)
and the T/O/R for the agent as well as the environment.

The description of the tiger problem is as follows: (Quote from `POMDP:
Introduction to Partially Observable Markov Decision Processes
<https://cran.r-project.org/web/packages/pomdp/vignettes/POMDP.pdf>`_ by
Kamalzadeh and Hahsler )

A tiger is put with equal probability behind one
of two doors, while treasure is put behind the other one.
You are standing in front of the two closed doors and
need to decide which one to open. If you open the door
with the tiger, you will get hurt (negative reward).
But if you open the door with treasure, you receive
a positive reward. Instead of opening a door right away,
you also have the option to wait and listen for tiger noises. But
listening is neither free nor entirely accurate. You might hear the
tiger behind the left door while it is actually behind the right
door and vice versa.

States: tiger-left, tiger-right
Actions: open-left, open-right, listen
Rewards:
    +10 for opening treasure door. -100 for opening tiger door.
    -1 for listening.
Observations: You can hear either "tiger-left", or "tiger-right".

Note that in this example, the HTNCoachDial is a POMDP that
also contains the agent and the environment as its fields. In
general this doesn't need to be the case. (Refer to more complicated
examples.)

"""

import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
import sys
from human_simulator import *
import math
from database import *
from helper import *
from env import *
db = DB_Object()
import config
# import copy
# from ExplaSet import *
# class TigerState(pomdp_py.State):
#     def __init__(self, name):
#         self.name = name
#         # self.exp = exp 
#     def __hash__(self):
#         return hash(self.name)
#     def __eq__(self, other):
#         if isinstance(other, TigerState):
#             return self.name == other.name
#         return False
#     def __str__(self):
#         return self.name
#     def __repr__(self):
#         return "TigerState(%s)" % self.name
#     def other(self):
#         if self.name.endswith("left"):
#             return TigerState("tiger-right")
#         else:
#             return TigerState("tiger-left")


class ObjectAttrAndLangState(pomdp_py.ObjectState):
    def __init__(self, object_class, obj_attr_dict):
        # """Note: camera_direction is None unless the robot is looking at a direction,
        # in which case camera_direction is the string e.g. look+x, or 'look'"""
        super().__init__(object_class, obj_attr_dict)
        ## object_class: explaset -  action posterior value, sensor_notification, feedback, object - {obj_name:, attr_name: atr_val}, question_asked for language utterance, 
        ## language - {feedback: , question_asked:}
        ## question_asked
    def __str__(self):
        # return 'ObjectState(%s,%s|%s' % (str(self.objclass), str(self.pose), str(self.objects_found))
        return 'ObjectAttrAndLangState(%s|%s' % (str(self.objclass), str(self.attributes))
    def __repr__(self):
        return str(self)
    # @property
    # def pose(self):
    #     return self.attributes['ob_name']
    # # @property
    # def pose(self):
    #     return self.attributes['pose']
    # @property
    # def robot_pose(self):
    #     return self.attributes['pose']
    # @property
    # def objects_found(self):
    #     return self.attributes['objects_found']

class HTNCoachDialState(pomdp_py.OOState):
    def __init__(self, htn_explaset, object_states):
        self.htn_explaset = htn_explaset
        super().__init__(object_states)
    # def object_pose(self, objid):
    #     return self.object_states[objid]["pose"]
    # def pose(self, objid):
    #     return self.object_pose(objid)
    # @property
    # def object_poses(self):
    #     return {objid:self.object_states[objid]['pose']
    #             for objid in self.object_states}
    def __str__(self):
        return 'HTNCoachDialState%s' % (str(self.object_states))
    def __repr__(self):
        return str(self)
    def append_object_attribute(self, objid, attr, new_val):
        attr_val_list = self.object_states[objid][attr]
        attr_val_list.append(new_val)
    def set_htn_explaset(self,explaset):
        self.htn_explaset = explaset
        return


# class HTNCoachDialState(pomdp_py.State):
#     # TODO:check this?
#     def __init__(self, world_state, exp): 
#         self._world_state = _world_state
#         self._explaset = exp #use exp._prob and exp._action_posterior
#         self._question_asked =  None
#         self._sensor_notification = None
#     def __hash__(self):
#         return hash(self.world, self.exp, self._question_asked, self._sensor_notification)
#     def __eq__(self,other):
#         if isinstance(self,State):
#              ## TODO:check that object is same
#             return self._world_state == other._world_state\
#                 and list(self._explaset) == list(other._explaset)\
#                 and all([expla1._prob == expla2._prob for expla1, expla2 in zip(self._explaset, other._explaset)])\
#                 and all([expla1._start_task == expla2._start_task for expla1, expla2 in zip(self._explaset, other._explaset)])\
#                 and self._explaset._action_posterior_prob == other.exp._action_posterior_prob\
#                 and self._explaset._prior == other._explaset._prior\
#                 and self._explaset._otherHappen == other._explaset._otherHappen\
#                 and self._question_asked == other._question_asked\
#                 and self._sensor_notification == other._sensor_notification
#         # and self.exp == self.exp\ 
#         # self._non_happen = non_happen
#         # self.__sensor_notification = []
#         # self._output_file_name = output_file_name
#         #  = {}
#         # self._language_notification = []
#         # self.highest_action_PS = []
#         # self._forest = forest
#         # self._pendingSet = pendingSet
#         # self._start_task = start_task
#         else:
#             return False
#     def __str__(self):
#         return self.__repr__()

#     def __repr__(self):
#         return "State(%s | %s | %s | %s | %s | %s)" % (str(self._world_state), str(self._explaset.__dict__), self._question_asked, self._sensor_notification)
    


class Action(pomdp_py.Action):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "Action(%s)" % self.name
    
# class TigerAction(pomdp_py.Action):
#     def __init__(self, name):
#         self.name = name
#     def __hash__(self):
#         return hash(self.name)
#     def __eq__(self, other):
#         if isinstance(other, TigerAction):
#             return self.name == other.name
#         return False
#     def __str__(self):
#         return self.name
#     def __repr__(self):
#         return "TigerAction(%s)" % self.name
# class AgentGiveNextInstructionAction(Action):
#     """
#     Robot action for giving the next instruction
#     """
#     ##@II need to code that it increases instruction by 1. As in MoveAction East is (1,0) (just defining)

#     def __init__(self):
#         super().__init__("give-next-instruction")

# def

class AgentAskClarificationQuestion(Action):
    """
    Robot action for giving the next instruction
    """
    ##@II need to code that it increases instruction by 1. As in MoveAction East is (1,0) (just defining)

    def __init__(self):
        super().__init__("ask-clarification-question")

    def update_question_asked(self, state):
        # print("state in action is", state)
        # explaset_title = "explaset_action"
        # explaset_title = explaset_title.split("_")
        # question_asked_state = state.get_object_state(title)
        # question_asked_state.attributes[title_split[1]] = question_asked
        
        highest_action_PS = ["", float('-inf')]
        if state.htn_explaset == None:
            return None
        for k, v in state.htn_explaset._action_posterior_prob.items():
            if v > highest_action_PS[1]:
                highest_action_PS = [k,v]
        #TODO: actual epxlaset not updated
        # exp.highest_action_PS = highest_action_PS 

        return highest_action_PS[0]
        # return state

# class AgentWaitAction(Action):
#     """
#     Robot action for waiting for user's utterances.
#     """
#     def __init__(self):
#         super().__init__("wait")


# class Observation(pomdp_py.Observation):
#     def __init__(self, name):
#         # self.name = name
#         self._world_state = {}
#         self._question_asked = None
#         self._explaset = None
#     def __hash__(self):
#         return hash(self.name)
#     def __eq__(self, other):
#         if isinstance(other, Observation):
#             return self.name == other.name
#         return False
#     def __str__(self):
#         return self.name
#     def __repr__(self):
#         return "TigerObservation(%s)" % self.name


class ObjectAttrAndLangObservation(pomdp_py.Observation):
    """The xy pose of the object is observed; or NULL if not observed"""
    NULL = None
    def __init__(self, obj_name, attr_name, attr_val):
        # self.obj_type = obj_type #object, language
        self.obj_name = obj_name #faucet1, ""
        self.attr_name = attr_name #state, feedback
        self.attr_val = attr_val #on, yes
        # if type(pose) == tuple and len(pose) == 2\
        #    or pose == ObjectObservation.NULL:
        #     self.pose = pose
        # else:
        #     raise ValueError("Invalid observation %s for object"
        #                      % (str(pose), objid))
    def __hash__(self):
        return hash((self.obj_name, self.attr_name, self.attr_val))
    def __eq__(self, other):
        if not isinstance(other, ObjectObservation):
            return False 
        else:
            return self.obj_name == other.obj_name\
                and self.attr_name == other.attr_name\
                and self.attr_val == other.attr_val\
                # and self.obj_type == other.obj_type\


# class LanguageObservation(pomdp_py.Observation):
#     """The xy pose of the object is observed; or NULL if not observed"""
#     NULL = None
#     def __init__(self, feedback):
#         self.feedback = feedback
#         # self.attr_name = attr_name
#         # self.attr_val = attr_val
#         # if type(pose) == tuple and len(pose) == 2\
#         #    or pose == ObjectObservation.NULL:
#         #     self.pose = pose
#         # else:
#         #     raise ValueError("Invalid observation %s for object"
#         #                      % (str(pose), objid))
#     def __hash__(self):
#         return hash(self.feedback)
#     def __eq__(self, other):
#         if not isinstance(other, LanguageObservation):
#             return False
#         else:
#             return self.feedback == other.feedback

class HTNCoachDialObservation(pomdp_py.OOObservation):
    """Observation for Mos that can be factored by objects;
    thus this is an OOObservation."""
    def __init__(self, lang_objattrs):
        """
        objposes (dict): map from objid(obj_name-attr_name) to state value or NULL (not ObjectObservation!).
        """
        self._hashcode = hash(frozenset(lang_objattrs.items()))
        self.lang_objattrs = lang_objattrs

    def for_obj(self, objid):
        if objid in self.lang_objattrs:
            return ObjectAttrAndLangObservation(objid, self.lang_objattrs[objid])
        # else:
        #     return ObjectAttrAndLangObservation(objid, ObjectObservation.NULL)
        
    def __hash__(self):
        return self._hashcode
    
    def __eq__(self, other):
        if not isinstance(other, HTNCoachDialObservation):
            return False
        else:
            return self.lang_objattrs == other.lang_objattrs

    def __str__(self):
        return "HTNCoachDialObservation(%s)" % str(self.lang_objattrs)

    def __repr__(self):
        return str(self)

    def get_lang_objattr(self, objid):
        return self.lang_objattrs[objid]
    # def factor(self, next_state, *params, **kwargs):
    #     """Factor this OO-observation by objects"""
    #     # return {objid: ObjectObservation(objid, self.objposes[objid])
    #     #         for objid in next_state.object_states
    #     #         if objid != next_state.robot_id}

    #     # return 
    #     factored_observations = {}
    #     for objid in self.lang_objattrs.keys():
    #         title = objid.split("-")
    #         factored_observations[objid]=ObjectAttrAndLangObservation(title[0], title[1], self.lang_objattrs[objid])
    #     return factored_observations
    # #     return {objid:
    # #             ObjectAttrAndLangObservation(obj_type, obj_name, attr_name, attr_val)
    # #             ObjectObservation(objid, self.objposes[objid])
    # #             for objid in self.lang_objattrs.keys()}
    # #             # if objid != next_state.robot_id}
    # # #
    
    # @classmethod
    # def merge(cls, object_observations, next_state, *params, **kwargs):
    #     """Merge `object_observations` into a single OOObservation object;
        
    #     object_observation (dict): Maps from objid to ObjectObservation"""
    #     merge_dict = {}
    #     for obs in object_observations:
    #         title = "_".join([obs.obj_name, obs.attr_name])
    #         merge_dict[title] = obs.attr_val
    #     return HTNCoachDialObservation(merge_dict)
    #     # return HTNCoachDialObservation({objid: object_observations[objid].pose
    #     #                          for objid in object_observations
    #     #                          if objid != next_state.object_states[objid].objclass != "robot"})

    #     # ({objid: object_observations[objid].pose


class HTNCoachDialObservationModel(pomdp_py.ObservationModel):
    '''
    Sample language observation form the state
    '''
    def __init__(self):
        pass

    def probability(self, observation, next_state, action):
        # if action.name == "ask-clarification-question":
        # elif action.name == "wait" and observation.get_lang_objattrget_lang_:
        a=1
        pass

        

    def sample(self, next_state, action):
        feedback_title = config.feedback_title
        if action.name == "ask-clarification-question":
            question_title = config.question_title
            question_title_split = question_title.split("-")
            # question_asked_state = next_state.get_object_state(question_title)
            question_index = next_state.get_object_attribute(question_title, question_title_split[1])
            

            explaset_title = config.explaset_title

            explaset_title_split = explaset_title.split("-")
            sequence_actions = next_state.get_object_attribute(explaset_title, explaset_title_split[1])
            # sensor_state = next_state.get_object_state(explaset_title)
            lang_objattrs = {}
            
            if question_index == sequence_actions[-2]:
                lang_objattrs[feedback_title] = "yes"
            else:
                lang_objattrs[feedback_title] = "no"
            
            ##nextstate is not updated with environment changes for now, just old changes
        else:
            lang_objattrs = {feedback_title: None}
        return HTNCoachDialObservation(lang_objattrs)
            
            # question_asked_state.attributes[title_split[1]] = question_asked
            # question_asked_state.__setitem__(question_title_split[1], question_asked)
        

        # if action.name == "listen":
        #     thresh = 1.0 - self.noise
        # else:
        #     thresh = 0.5

        # if random.uniform(0,1) < thresh:
        #     # return Type(Observation
        #     return wrapperobservation([facucet_on, hand_soapy, yes])
        #     return TigerObservation(next_state.name)
        # else:
        #     return TigerObservation(next_state.other().name)
        '''
        sample_observation = {}
        for key, prob_dist in self._lang_objattrs_prob.items():
            choice = random.choices(list(prob_dist.keys()), list(prob_dist.values()))
            sample_observation[key] = choice[0]
        # return random.choices()
        return HTNCoachDialObservation(sample_observation)
        '''
        

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [TigerObservation(s) for s in {"tiger-left", "tiger-right"}]

    def set_lang_objattrs_prob(self,lang_objattrs_prob):
        # self._hashcode = hash(frozenset(lang_objattrs_prob.items()))
        for key, value in lang_objattrs_prob.items():
            self._lang_objattrs_prob[key] = value

'''
class HTNCoachDialObservationModel(pomdp_py.ObservationModel):
    # Sample language observation form the state
    
    def __init__(self,lang_objattrs):
        # self.noise = noise
        # self.hs = human_simulator()
        # self._hashcode = hash(frozenset(lang_objattrs.items()))
        self._lang_objattrs_prob = lang_objattrs

    def probability(self, observation, next_state, action):
        #joint probablity of all the 
        # if action.name == "listen":
        #     if observation.name == next_state.name: # heard the correct growl
        #         return 1.0 - self.noise
        #     else:
        #         return self.noise
        # else:
        #     return 0.5

        for objid, val in observation.lang_objattrs:
            return self._lang_objattrs_prob[objid][val]

        # retu/rn 


    def sample(self, next_state, action):
        # if action.name == "listen":
        #     thresh = 1.0 - self.noise
        # else:
        #     thresh = 0.5

        # if random.uniform(0,1) < thresh:
        #     # return Type(Observation
        #     return wrapperobservation([facucet_on, hand_soapy, yes])
        #     return TigerObservation(next_state.name)
        # else:
        #     return TigerObservation(next_state.other().name)
        sample_observation = {}
        for key, prob_dist in self._lang_objattrs_prob.items():
            choice = random.choices(list(prob_dist.keys()), list(prob_dist.values()))
            sample_observation[key] = choice[0]
        # return random.choices()
        return HTNCoachDialObservation(sample_observation)



    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [TigerObservation(s) for s in {"tiger-left", "tiger-right"}]

    def set_lang_objattrs_prob(self,lang_objattrs_prob):
        # self._hashcode = hash(frozenset(lang_objattrs_prob.items()))
        for key, value in lang_objattrs_prob.items():
            self._lang_objattrs_prob[key] = value
        # self._lang_objattrs_prob = lang_objattrs_prob
'''

# Observation model
# class ObservationModel(pomdp_py.ObservationModel):
#     def __init__(self, noise=0.8):
#         self.noise = noise
#         # self.hs = human_simulator()

#     def probability(self, observation, next_state, action):
#         #joint probablity of all the 
#         if action.name == "listen":
#             if observation.name == next_state.name: # heard the correct growl
#                 return 1.0 - self.noise
#             else:
#                 return self.noise
#         else:
#             return 0.5

#     def sample(self, next_state, action):
#         if action.name == "listen":
#             thresh = 1.0 - self.noise
#         else:
#             thresh = 0.5

#         if random.uniform(0,1) < thresh:
#             # return Type(Observation
#             return wrapperobservation([facucet_on, hand_soapy, yes])
#             return TigerObservation(next_state.name)
#         else:
#             return TigerObservation(next_state.other().name)

#     def get_all_observations(self):
#         """Only need to implement this if you're using
#         a solver that needs to enumerate over the observation space (e.g. value iteration)"""
#         return [TigerObservation(s) for s in {"tiger-left", "tiger-right"}]

# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
    def __init__(self, hs):
        self.human_simulator = hs

    def probability(self, next_state, state, action):
        return self.human_simulator.probablity(next_state, state)

    def sample(self, state, action):
        # set_object_state(self, objid, object_state)
        # next_state = HTNCoachDialState()
        # get_object_state(self, objid)
        # print("state", state)
        if self.human_simulator.mcts_check_terminal_state():
            return state
        if action.name == "ask-clarification-question":
            question_asked = action.update_question_asked(state)

        else:
            question_asked = None
        question_title = config.question_title
        question_title_split = question_title.split("-")
        question_asked_state = state.get_object_state(question_title)
        # question_asked_state.attributes[title_split[1]] = question_asked
        question_asked_state.__setitem__(question_title_split[1], question_asked)
        
        explaset_title = config.explaset_title
        explaset_title_split = explaset_title.split("-")
        sensor_state = state.get_object_state(explaset_title)
        step,_ = self.human_simulator.curr_step(sensor_state.attributes[explaset_title_split[1]], action.name)
        state.append_object_attribute(explaset_title, explaset_title_split[1], step)

        # TODO: transition explaset
        # sensor_state.attributes[explaset_title_split[1]] =  
        

        # title = "sensor-notif"
        # title_split = title.split("-")
        # sensor_state = state.get_object_state(title)
        # sensor_state.attributes[title_split[1]] =  self.human_simulator.curr_step(sensor_state.attributes[title_split[1]], action.name)
        # # question_asked_state.attributes[title_split[1]] = self.human_simulator.curr_step(question_asked_state.attributes[title_split[1]], action.name)
        # state.set_object_state(title, sensor_state)
        
        # next_state._question_asked = question_asked
        # next_state._sensor_notification = hs.curr_step(self, state, action)
        return state
        
    def get_all_states(self):
        ##TODO: states are notifs or world states? need to convert notifs to states.
        return self.human_simulator.get_all_states()

# class TransitionModel(pomdp_py.TransitionModel):
    
#     def probability(self, next_state, state, action):
#         """According to problem spec, the world resets once
#         action is open-left/open-right. Otherwise, stays the same"""
#         if action.name.startswith("open"):
#             return 0.5
#         else:
#             if next_state.name == state.name:
#                 return 1.0 - 1e-9
#             else:
#                 return 1e-9

#     def sample(self, state, action):
#         if action.name.startswith("open"):
#             return random.choice(self.get_all_states())
#         else:
#             return TigerState(state.name)

#     def get_all_states(self):
#         """Only need to implement this if you're using
#         a solver that needs to enumerate over the observation space (e.g. value iteration)"""
#         return [TigerState(s) for s in {"tiger-left", "tiger-right"}]


class RewardModel(pomdp_py.RewardModel):
    def __init__(self, hs):
        self.human_simulator = hs
    
    # ACTIONS = {Action(s) for s in {"ask-clarification-question", "wait"}}

    def sample(self, state, action, next_state, normalized=False, **kwargs):
        ''' Sensor notif is taken from the previous state based on which the question is asked and the
        # question asked index is extracted from next state because after the action of asking question it will be updated
        # in the next state '''
        explaset_title = config.explaset_title
        explaset_title_split = explaset_title.split("-")
        sensor_state = state.get_object_state(explaset_title)
        sensor_notification = sensor_state.attributes[explaset_title_split[1]]
        
        question_title = config.question_title
        question_title_split = question_title.split("-")
        question_asked_state = next_state.get_object_state(question_title)
        question_asked = question_asked_state.attributes[question_title_split[1]]
        if self.human_simulator._notifs[self.human_simulator.index_test_case]._notif.empty():
            return 10
        elif action.name == "wait":
            return -1
        elif action.name == "ask-clarification-question" and sensor_notification == question_asked:
            return 5
        else:
            return -5 #-100s

    def argmax(self, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError

    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError



# Reward Model
# class RewardModel(pomdp_py.RewardModel):
#     def _reward_func(self, state, action):
#         # if action_name == 
#         if action.name == "open-left":
#             if state.name == "tiger-right":
#                 return 10
#             else:
#                 return -100
#         elif action.name == "open-right":
#             if state.name == "tiger-left":
#                 return 10
#             else:
#                 return -100
#         else: # listen
#             return -1

#     def sample(self, state, action, next_state):
#         # deterministic
#         return self._reward_func(state, action)

# Policy Model
class PolicyModel(pomdp_py.RandomRollout):
    """This is an extremely dumb policy model; To keep consistent
    with the framework."""
    # A stay action can be added to test that POMDP solver is
    # able to differentiate information gathering actions.
    # ACTIONS = {Action(s) for s in {"ask-clarification-question", "wait"}}
    ACTIONS = {Action("wait"), AgentAskClarificationQuestion() }
    # {Action(s) for s in {"ask-clarification-question", "wait"}}

    def sample(self, state, **kwargs):
        return random.sample(self.get_all_actions(), 1)[0]

    def get_all_actions(self, **kwargs):
        return PolicyModel.ACTIONS

    def probability(self, action, state, normalized=False, **kwargs):
        raise NotImplementedError

    def argmax(self, state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        raise NotImplementedError

class HTNCoachDialBelief(pomdp_py.OOBelief):
    """This is needed to make sure the belief is sampling the right
    type of State for this problem."""
    def __init__(self, object_beliefs):
        """
        robot_id (int): The id of the robot that has this belief.
        object_beliefs (objid -> GenerativeDistribution)
            (includes robot)
        """
        # self.robot_id = robot_id
        self.htn_explaset  = None
        super().__init__(object_beliefs)

    def mpe(self, **kwargs):
        return HTNCoachDialState(pomdp_py.OOBelief.mpe(self, **kwargs).object_states)

    def random(self, **kwargs):
        object_states = pomdp_py.OOBelief.random(self, **kwargs).object_states
        return HTNCoachDialState(self.htn_explaset, pomdp_py.OOBelief.random(self, **kwargs).object_states)

    def set_htn_explaset(self,explaset):
        self.htn_explaset = explaset
        return

def convert_object_belief_to_histogram(init_worldstate_belief):
    '''Convert belief object to histogram dictionary'''
    # TODO:Tian
    #input two lists
    #output a big dict
    from collections import defaultdict
    print(init_worldstate_belief)
    # obj_belief_dict = defaultdict(list)  # {obj_name: state_name: {state1: prob, ..}}
    obj_belief_dict = defaultdict(pomdp_py.Histogram)  # {obj_name: state_name: {state1: prob, ..}}
    HTN_object_state_dict = defaultdict(ObjectAttrAndLangState)
    # [objid]=histogram
    observation_prob ={}
    observe_distribute ={}
    for obj in init_worldstate_belief:
        obj_name = obj["ob_name"]

        for key, value in obj.items():
            if key in ["_id", "ob_name", "ob_type"]:
                # Ignore keys that are not an object state
                continue

            obj_state_dict = {}  # {ObjectState: state_prob}
            prev_observation_prob_key = None
            for state, state_prob in value.items():
                # if state_prob == 1.0:
                #     # Skip a state when it only has one possible value.
                #     continue
                #flat_dict[f"{obj_name}_{key}_{state}"] = state_prob
                # obj_state = pomdp_py.ObjectState(obj_name, {key: state})
                attr_dict = {"obj_name": obj_name, "attr_name":key, "attr_val":state}
                obj_state = ObjectAttrAndLangState("object", attr_dict)
                # obj_id = "_".join(obj_state.attributes.keys())
                # HTN_object_state_dict[obj_id] = obj_state
                obj_state_dict[obj_state] = state_prob

                # for value in attri_distribute:
                # observe_distribute[value] = db.get_obs_prob(value, item[0], item[1])
                observation_prob_key = "-".join([obj_name, key])
                if observation_prob_key == prev_observation_prob_key:
                    observe_distribute[state] = db.get_obs_prob(state, obj_name, key) 
                    observation_prob[observation_prob_key] = observe_distribute
                    observe_distribute = {}
                else:
                    observe_distribute[state] = db.get_obs_prob(state, obj_name, key) 
                    prev_observation_prob_key = observation_prob_key

            if observe_distribute:
                observation_prob[observation_prob_key] = observe_distribute

            

            if obj_state_dict != {}:
                # obj_belief_dict[obj_name].append(obj_state_dict)
                init_object_state = max(obj_state_dict, key=obj_state_dict.get)
                obj_id = "-".join(list(init_object_state.attributes.values())[:-1])
                obj_belief_dict[obj_id] = pomdp_py.Histogram(obj_state_dict)
                HTN_object_state_dict[obj_id] = init_object_state
        # ObjectAttrAndLangState("object", )
        ##init_object_state
        # attr_dict = {"obj_name": obj_name, key:state}
    print(obj_belief_dict)
    return obj_belief_dict, HTN_object_state_dict, observation_prob

def update_belief(HTNCoachDial_problem,action, real_observation, prob_lang):
    # print(HTNCoachDial_problem.agent.curr_belief, HTNCoachDial_problem.agent.observation_model,
                    # HTNCoachDial_problem.agent.transition_model)
    # pass
    if real_observation.get_lang_objattr(config.feedback_title) == None:
        pass



class HTNCoachDial(pomdp_py.POMDP):
    """
    In fact, creating a HTNCoachDial class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, init_true_state, init_belief, output_filename, observation_prob, explaset = None):
        """init_belief is a Distribution."""
        
        self.hs = human_simulator(output_filename)
        # self.hs.read_files()
        # self.hs.goal_selection()

        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(self.hs),
                               HTNCoachDialObservationModel(),
                               RewardModel(self.hs))
        # env = pomdp_py.Environment(init_true_state,
        #                            TransitionModel(),
        #                            RewardModel())
        # self, human_simulator, explaset, init_state, RewardModel, TransitionModel
        env =  HTNCoachDialEnvironment(self.hs, explaset, init_true_state,
                                   RewardModel(self.hs), TransitionModel(self.hs))
        # self.environment_reward_model = RewardModel(self.hs)
        # env = None
        super().__init__(agent, env, name="HTNCoachDial") ## ask kaiyu

    @staticmethod
    def create(state="tiger-left", belief=0.5, obs_noise=0.15):
        """
        Args:
            state (str): could be 'tiger-left' or 'tiger-right'; True state of the environment
            belief (float): Initial belief that the target is on the left; Between 0-1.
            obs_noise (float): Noise for the observation model (default 0.15)
        """
        init_true_state = TigerState(state)
        init_belief = pomdp_py.Histogram({TigerState("tiger-left"): belief,
                                          TigerState("tiger-right"): 1.0 - belief})
        HTNCoachDial_problem = HTNCoachDial(obs_noise,  # observation noise
                                     init_true_state, init_belief)
        HTNCoachDial_problem.agent.set_belief(init_belief, prior=True)
        return HTNCoachDial_problem


def planner_one_loop(HTNCoachDial_problem, planner, nsteps=3, debug_tree=False, discount=0.95, gamma = 1.0, total_reward = 0, total_discounted_reward = 0, i=0, true_state = None, prob_lang = 0.95):
    action = planner.plan(HTNCoachDial_problem.agent)
    if debug_tree:
        from pomdp_py.utils import TreeDebugger
        dd = TreeDebugger(HTNCoachDial_problem.agent.tree)
        import pdb; pdb.set_trace()

    print("==== Step %d ====" % (i+1))
    ## true state, get from simulator
    # if i == 0:
    #     curr_step = HTNCoachDial_problem.hs.curr_step("none", action, True)
    # else:
    #     curr_step = HTNCoachDial_problem.hs.curr_step(curr_step, action, True)
    ## env state update accordnig to basics state transition
    env_reward = HTNCoachDial_problem.env.state_transition(action, execute=True)
    true_next_state = copy.deepcopy(HTNCoachDial_problem.env.state)
    
    # env_reward = HTNCoachDial_problem.env.reward_model.sample(HTNCoachDial_problem.env.state, action, None)
    # env_reward = HTNCoachDial_problem.environment_reward_model.sample(HTNCoachDial_problem.env.state, action, None)
    ##TODO: need env state variable in the coachdial probelm. update the state variable with true state info.
    print("True state: %s" % true_state)
    # print("True state: %s" % HTNCoachDial_problem.env.state)
    print("Belief: %s" % str(HTNCoachDial_problem.agent.cur_belief))
    print("Action: %s" % str(action))
    print("Reward: %s" % str(env_reward))
    total_reward += env_reward
    total_discounted_reward += env_reward * gamma
    gamma *= discount
    print("Reward (Cumulative): %s" % str(total_reward))
    print("Reward (Cumulative Discounted): %s" % str(total_discounted_reward))
        
    if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)
            print("__plan_time__: %.5f" % planner.last_planning_time)
    if isinstance(planner, pomdp_py.PORollout):
        print("__best_reward__: %d" % planner.last_best_reward)
        
    # Let's create some simulated real observation; Update the belief
    # Creating true observation for sanity checking solver behavior.
    # In general, this observation should be sampled from agent's observation model.
    real_observation = HTNCoachDial_problem.env.provide_observation(HTNCoachDial_problem.agent.observation_model,
                                                              action)
    # real_observation = TigerObservation(HTNCoachDial_problem.env.state.name)
    print(">> Observation: %s" % real_observation)
    HTNCoachDial_problem.agent.update_history(action, real_observation)

    # If the planner is POMCP, planner.update also updates agent belief.
    planner.update(HTNCoachDial_problem.agent, action, real_observation)
    if isinstance(planner, pomdp_py.POUCT):
        print("Num sims: %d" % planner.last_num_sims)
        print("Plan time: %.5f" % planner.last_planning_time)

    # TODO: belief update for now update the explaset

    update_belief(HTNCoachDial_problem,
                    action, real_observation, prob_lang) 
    
    # if isinstance(HTNCoachDial_problem.agent.cur_belief, pomdp_py.Histogram):
    #     new_belief = pomdp_py.update_histogram_belief(HTNCoachDial_problem.agent.cur_belief,
    #                                                     action, real_observation,
    #                                                     HTNCoachDial_problem.agent.observation_model,
    #                                                     HTNCoachDial_problem.agent.transition_model)
    #     HTNCoachDial_problem.agent.set_belief(new_belief)

    # if action.name.startswith("open"):
        # Make it clearer to see what actions are taken until every time door is opened.
        # print("\n")
    return total_reward, total_discounted_reward


def test_planner(HTNCoachDial_problem, planner, nsteps=3, debug_tree=False):
    """
    Runs the action-feedback loop of Tiger problem POMDP

    Args:
        HTNCoachDial_problem (HTNCoachDial): an instance of the tiger problem.
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
    """
    gamma = 1.0
    total_reward = 0
    total_discounted_reward = 0
    print("Test Case Index is: %s" % HTNCoachDial_problem.hs.index_test_case)
    for i in range(nsteps):
        action = planner.plan(HTNCoachDial_problem.agent)
        if debug_tree:
            from pomdp_py.utils import TreeDebugger
            dd = TreeDebugger(HTNCoachDial_problem.agent.tree)
            import pdb; pdb.set_trace()

        print("==== Step %d ====" % (i+1))
        ## true state, get from simulator
        if i == 0:
            curr_step = HTNCoachDial_problem.hs.curr_step("none")
        else:
            curr_step = HTNCoachDial_problem.hs.curr_step(curr_step)
        print("True state: %s" % curr_step)
        # print("True state: %s" % HTNCoachDial_problem.env.state)
        print("Belief: %s" % str(HTNCoachDial_problem.agent.cur_belief))
        print("Action: %s" % str(action))
        print("Reward: %s" % str(HTNCoachDial_problem.env.reward_model.sample(HTNCoachDial_problem.env.state, action, None)))

        # Let's create some simulated real observation; Update the belief
        # Creating true observation for sanity checking solver behavior.
        # In general, this observation should be sampled from agent's observation model.
        real_observation = TigerObservation(HTNCoachDial_problem.env.state.name)
        print(">> Observation: %s" % real_observation)
        HTNCoachDial_problem.agent.update_history(action, real_observation)

        # If the planner is POMCP, planner.update also updates agent belief.
        planner.update(HTNCoachDial_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims: %d" % planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        if isinstance(HTNCoachDial_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(HTNCoachDial_problem.agent.cur_belief,
                                                          action, real_observation,
                                                          HTNCoachDial_problem.agent.observation_model,
                                                          HTNCoachDial_problem.agent.transition_model)
            HTNCoachDial_problem.agent.set_belief(new_belief)

        if action.name.startswith("open"):
            # Make it clearer to see what actions are taken until every time door is opened.
            print("\n")

def main():
    init_true_state = random.choice([TigerState("tiger-left"),
                                     TigerState("tiger-right")])
    init_belief = pomdp_py.Histogram({TigerState("tiger-left"): 0.5,
                                      TigerState("tiger-right"): 0.5})
    HTNCoachDial_problem = HTNCoachDial(0.15,  # observation noise
                                 init_true_state, init_belief)

    # print("** Testing value iteration **")
    # vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
    # test_planner(HTNCoachDial_problem, vi, nsteps=3)

    # # Reset agent belief
    # HTNCoachDial_problem.agent.set_belief(init_belief, prior=True)

    print("\n** Testing POUCT **")
    pouct = pomdp_py.POUCT(max_depth=3, discount_factor=0.95,
                           num_sims=4096, exploration_const=50,
                           rollout_policy=HTNCoachDial_problem.agent.policy_model,
                           show_progress=True)
    test_planner(HTNCoachDial_problem, pouct, nsteps=10, debug_tree=False)
    TreeDebugger(HTNCoachDial_problem.agent.tree).pp

    # Reset agent belief
    # HTNCoachDial_problem.agent.set_belief(init_belief, prior=True)
    # HTNCoachDial_problem.agent.tree = None

    # print("** Testing POMCP **")
    # HTNCoachDial_problem.agent.set_belief(pomdp_py.Particles.from_histogram(init_belief, num_particles=100), prior=True)
    # pomcp = pomdp_py.POMCP(max_depth=3, discount_factor=0.95,
    #                        num_sims=1000, exploration_const=50,
    #                        rollout_policy=HTNCoachDial_problem.agent.policy_model,
    #                        show_progress=True, pbar_update_interval=500)
    # test_planner(HTNCoachDial_problem, pomcp, nsteps=10)
    # TreeDebugger(HTNCoachDial_problem.agent.tree).pp

if __name__ == '__main__':
    main()
