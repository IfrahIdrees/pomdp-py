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

from re import X
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
from State import *
from Simulator import languageStateANDSensorUpdate

import logging

# import numpy as np
import pandas as pd
import os
import argparse
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
    def __init__(self, step_index, htn_explaset, object_states):
        self.step_index = step_index
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
        explaset_title = config.explaset_title
        explaset_title_split = explaset_title.split("-")
        # print)
        sensor_state = self.get_object_state(explaset_title)
        sensor_notification = sensor_state.attributes[explaset_title_split[1]]

        question_title =config.question_title
        question_title_split = question_title.split("-")
        # print)
        question_state = self.get_object_state(question_title)
        # print(question_state.attributes, question_title_split)
        question_notification = question_state.attributes[question_title_split[1]]


        return 'HTNCoachDialState:%s,%s, question_asked%s, explaset_other_happen%s' % (str(self.step_index), str(sensor_notification), str(question_notification), str(self.htn_explaset._other_happen)) #, str(self.object_states))
    def __repr__(self):
        return str(self)
    #state.append_object_attribute(explaset_title, explaset_title_split[1], step)
    def append_object_attribute(self, objid, attr, new_val):
        attr_val_list = self.object_states[objid][attr]
        attr_val_list.append(new_val)
    def set_htn_explaset(self,explaset):
        self.htn_explaset = explaset
        return
    def set_step_index(self, step_index):
        self.step_index = step_index
        return
    def diff_state(self, other):
        diff = {}
        for key, objlangstate in self.object_states.items():
            if key in other.object_states.keys() and \
                other.object_states[key].objclass == self.object_states[key].objclass and \
                other.object_states[key].attributes == self.object_states[key].attributes:
                continue
            else:
                diff[key]=other.object_states[key]
        return diff

    def __hash__(self) -> int:
        hashint = 0
        
        for key, objlangstate in self.object_states.items():
            hashint += hash(str(objlangstate.objclass))
            if key != config.explaset_title:
                hashint += hash(json.dumps(objlangstate.attributes, sort_keys=True))
            else:
                explaset_title_split = config.explaset_title.split("-")
                hashint += hash("".join(objlangstate[explaset_title_split[1]]))
        return hashint
        # return super().__hash__()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HTNCoachDialState) and not self.diff_state(other):
            return True
        else:
            return False

        # return super().__eq__(__o)





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
        self.question_asked = None

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
        self.question_asked = highest_action_PS[0]
        return highest_action_PS
        # return highest_action_PS
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
            # print("sensor_notification is", sequence_actions)
            if question_index[0] == sequence_actions[-2]:
                lang_objattrs[feedback_title] = "yes"
            else:
                lang_objattrs[feedback_title] = "no"
            
            ##nextstate is not updated with environment changes for now, just old changes
        else:
            lang_objattrs = {feedback_title: None}

        world_observations = next_state.htn_explaset._sensor_notification

        for world_observation in world_observations:
            key = world_observation["object"]+"-"+world_observation["attribute"]
            lang_objattrs[key] = world_observation["obj_att_value"]

        # return HTNCoachDialObservation({**lang_objattrs, **world_observation})
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
        self.all_hash = set()
        self.human_simulator = hs

    def probability(self, next_state, state, action):
        return self.human_simulator.probablity(next_state, state)

    def sample(self, state, action, execute = False):
        # set_object_state(self, objid, object_state)
        # next_state = HTNCoachDialState()
        # get_object_state(self, objid)
        # print("state", state)
        # current_state = state
        state = copy.deepcopy(state)
        # state_hash = hash(state)
        # if state_hash in self.all_hash:
        #     print("Current state's hash is already here")
        # else:
        #     print("A new state")
        #   self.all_hash.add(state_hash)
        if self.human_simulator.check_terminal_state(state.step_index+1): ##check if next step index is out of length 
            return state
        if action.name == "ask-clarification-question":
            question_asked = action.update_question_asked(state)[0]

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
        # step, sensor_notification = self.human_simulator.curr_step(sensor_state.attributes[explaset_title_split[1]][-1], action.name)
        if execute:
            step_index, step, sensor_notifications = self.human_simulator.curr_step(state.step_index, action.name, real_step=True)
        else:
            step_index, step, sensor_notifications = self.human_simulator.curr_step(state.step_index, action.name)
        state.append_object_attribute(explaset_title, explaset_title_split[1], step)
        # print("state after append is", state)
        ## update the state_variable object_states
        state.step_index = step_index
        # sensor_notification
        for sensor_notification in sensor_notifications:
            notif = sensor_notification['object']+"-"+sensor_notification['attribute']
            notif_state = state.get_object_state(notif)
            notif_ObjectAttrAndLangState = ObjectAttrAndLangState('object',{'obj_name':sensor_notification['object'], 'attr_name':sensor_notification['attribute'], 'attr_val':sensor_notification['obj_att_value'] })
            # (object|{'obj_name': 'hand_1', 'attr_name': 'dry', 'attr_val': 'yes'}(object|{'obj_name': 'hand_1', 'attr_name': 'dry', 'attr_val': 'yes'}
            notif_state.__setitem__('object', sensor_notification['object'])
            notif_state.__setitem__('attr_name', sensor_notification['attribute'])
            notif_state.__setitem__('attr_val',sensor_notification['obj_att_value'])
            # notif_state.__setitem__(notif, notif_ObjectAttrAndLangState)
        # notif_state = state.get_object_state(notif)
        ## set explanation
        exp = state.htn_explaset
        exp.setSensorNotification(sensor_notifications)
        otherHappen, observation_prob = exp.action_posterior(execute)

        '''Executing transition of exp'''
        if otherHappen > config._other_happen:
                    # wrong step handling
            # print("action posterior after bayseian inference is",  exp._action_posterior_prob)
            exp.handle_exception()
            
        # correct step procedure
        else:
            # print("updating explaset")
            length = len(exp._explaset)
            
            # input step start a new goal (bottom up procedure to create ongoing status)
            # include recognition and planning
            exp.explaSet_expand_part1(length)

            # belief state update
            world_state = State()
            world_state.update_state_belief(exp)
            ## TODO: user the above function to update belief over the state.
            
            # input step continues an ongoing goal
            # include recognition and planning 
            exp.explaSet_expand_part2(length)
            

                    
        exp.pendingset_generate()
        
        # compute goal recognition result PROB and planning result PS
        if execute:
            exp.task_prob_calculate(self.human_simulator.real_output_filename)
            exp.print_explaSet()
        else:
            exp.task_prob_calculate(self.human_simulator.mcts_output_filename)
            exp.mcts_print_explaSet()
        #output PROB and PS in a file
        # if execute: 
        # else:
            
        state.set_htn_explaset(exp)


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
        # print("state after exp expand", state)
        # current_state = state
        
        # print(f"&&&&This is the difference: {current_state.diff_state(state)}")
        return state
        
    def get_all_states(self):
        ##TODO: states are notifs or world states? need to convert notifs to states.
        # pass
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
    def __init__(self, hs, args):
        self.human_simulator = hs
        self.goal_reward = args.gr
        self.wait_penalty = args.wp 
        self.question_penalty = args.qp 
        self.question_reward = args.qr 
    
    # ACTIONS = {Action(s) for s in {"ask-clarification-question", "wait"}}

    def sample(self, state, action, next_state, normalized=False, **kwargs):
        ''' Sensor notif is taken from the previous state based on which the question is asked and the
        # question asked index is extracted from next state because after the action of asking question it will be updated
        # in the next state '''
        # state s1, state s2
        explaset_title = config.explaset_title
        explaset_title_split = explaset_title.split("-")
        sensor_state = state.get_object_state(explaset_title)
        sensor_notification = sensor_state.attributes[explaset_title_split[1]]
        
        question_title = config.question_title
        question_title_split = question_title.split("-")
        question_asked_state = next_state.get_object_state(question_title)
        question_asked = question_asked_state.attributes[question_title_split[1]]
        # print("Reward, state", state)
        # print("Reward, next_state", next_state)
        # if action.name == "ask-clarification-question":
            # print("action", action.name, sensor_notification[-1] in self.human_simulator.all_wrong_actions, sensor_notification[-1] == question_asked, sensor_notification[-1], question_asked)
        # print(list(self.human_simulator.wrong_actions.values()))
        # print(self.human_simulator.all_wrong_actions)
        # [for i in ]self.human_simulator._notifs[self.human_simulator.index_test_case]._notif.empty()
        # print()
        if self.human_simulator.check_terminal_state(state.step_index+1) and action.name == "wait":
            return self.goal_reward 
        # elif self.human_simulator.check_terminal_state(state.step_index+1) and action.name == "ask-clarification-question":
            # return self.goal_reward 
        elif action.name == "wait":
            return self.wait_penalty
        elif action.name == "ask-clarification-question" and self.human_simulator.check_wrong_step(state.step_index):  #and sensor_notification[-1] in self.human_simulator.all_wrong_actions and sensor_notification[-1] == question_asked:
            return self.question_reward
        elif action.name == "ask-clarification-question" and sensor_notification[-1] != question_asked :  #and sensor_notification[-1] in self.human_simulator.all_wrong_actions and sensor_notification[-1] == question_asked:
            return self.question_reward
        elif action.name == "ask-clarification-question":  #not(sensor_notification[-1] in self.human_simulator.all_wrong_actions and sensor_notification[-1] == question_asked):
            return self.question_penalty
        # else:
            # return -5 #-100s

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
        return self.ACTIONS

    def probability(self, action, state, normalized=False, **kwargs):
        raise NotImplementedError

    def argmax(self, state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        raise NotImplementedError

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

class PreferredPolicyModel(PolicyModel):
    """The same with PolicyModel except there is a preferred rollout policypomdp_py.RolloutPolicy"""
    def __init__(self, action_prior, num_visits_init, val_init):
        self.action_prior = action_prior
        # super().__init__(self.action_prior.robot_id,
        #                  self.action_prior.grid_map,
        #                  no_look=self.action_prior.no_look)
        # self.action_prior.set_motion_actions(ALL_MOTION_ACTIONS)
        self.ACTIONS = {Action("wait"), AgentAskClarificationQuestion() }
        self.action_prior.set_all_actions(self.ACTIONS)
        self.num_visits_init = num_visits_init
        self.val_init = val_init
        
    def rollout(self, state, history):
        # Obtain preference and returns the action in it.
        preferences = self.action_prior.get_preferred_actions(state, history)
        if len(preferences) > 0:
            if len(preferences) > 1:
                print("here")
            return random.sample(preferences, 1)[0][0]
        else:
            ##wait is threex more likely since three times check if action_ask will be added or not.
            # preferences.add((Action("wait"), self.num_visits_init, self.val_init))
            action = np.random.choice(list(self.ACTIONS), 1, p=[0.75,0.25]) 
            return action[0]
            # if action[0] == Action("wait"):
                # return (Action("wait"), self.num_visits_init, self.val_init)
            # else:
                # return (AgentAskClarificationQuestion() , self.num_visits_init, self.val_init)


            # return action[0]
            # return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

    def get_all_actions(self, **kwargs):
        return self.ACTIONS
        # return super().get_all_actions(**kwargs)

class ActionPrior(pomdp_py.ActionPrior):
    """greedy action prior for 'xy' motion scheme"""
    def __init__(self, num_visits_init, val_init):
        # self.robot_id = robot_id
        # self.grid_map = grid_map
        self.all_actions = None
        self.num_visits_init = num_visits_init
        self.val_init = val_init
        # self.no_look = no_look

    def set_all_actions(self, motion_actions):
        self.all_actions = motion_actions

    def get_belief_repair_summary(self, explaset):
        # explaset = copy.deepcopy(explaset)

        belief_state_repair_summary = {} #record to what degree the belief state should be updated
        
        for expla in explaset._explaset:
            expla_repair_result = expla.repair_expla(explaset._sensor_notification)
            if expla_repair_result[0] != 0.0:
                explaset.belief_state_repair_summary_extend(belief_state_repair_summary, expla_repair_result)
        
        return belief_state_repair_summary
        # self.belief_state_repair_execute(belief_state_repair_summary)
        
    def check_correct_explanations(self, explaset, sensor_notification):
        correct_taskNets = 0
        for expla in explaset._explaset:
            found = False
            # goal_prob = expla._prob
            correct_taskNets = 0
            
            for taskNet_ in expla._forest:
                # for taskNet_ in forest:
                ExecuteSequence =  taskNet_._execute_sequence._sequence
                if ExecuteSequence == []:
                    # taskNet_._expandProb *= 0.01
                    # expla._prob*=0.01
                    continue
                if sensor_notification == ExecuteSequence[-1]: 
                    found =True
            if found == True:
                correct_taskNets+=1

        if correct_taskNets > 0:
            return True
        else:
            return False
        
                # taskNet_._expandProb *= p_l
                # expla._prob*=p_l
            # delta = 0.001
            # if len(expla._forest) == 0:
            #     weight = 0+delta
            # else:
            #     weight = float(correct_taskNets)/len(expla._forest)+delta
            

    def get_preferred_actions(self, state, history):
        """Get preferred actions. This can be used by a rollout policy as well."""
        # Prefer actions that move the robot closer to any
        # undetected target object in the state. If
        # cannot move any closer, look. If the last
        # observation contains an unobserved object, then Find.
        #
        # Also do not prefer actions that makes the robot rotate in place back
        # and forth.
        if self.all_actions is None:
            raise ValueError("Unable to get preferred actions because"\
                             "we don't know what motion actions there are.")

        

        ##check if terminal
        preferences = set()
        htn_explaset = copy.deepcopy(state.htn_explaset)

        explaset_title = config.explaset_title
        explaset_title_split = explaset_title.split("-")
        # print)
        sensor_state = state.get_object_state(explaset_title)
        ExecuteSequence = sensor_state.attributes[explaset_title_split[1]]
        sensor_notification = ExecuteSequence[-1]

        belief_confidence = False
        highest_belief_action  = [None, 0]

        # if sensor_notification == "turn_on_faucet_1":
            # print("here")
        
        if len(history) == 0:
            print("history is 0, here")
            preferences.add((Action("wait"), self.num_visits_init, self.val_init))
            return preferences


        for action, prob in htn_explaset._action_posterior_prob.items():
            if prob> highest_belief_action[1]:
                highest_belief_action=[action, prob]

        if not htn_explaset._sensor_notification:
            preferences.add((AgentAskClarificationQuestion(), self.num_visits_init, self.val_init))
            return preferences

        belief_state_repair = self.get_belief_repair_summary(htn_explaset)
        if len(belief_state_repair) > 0:
            preferences.add((AgentAskClarificationQuestion(), self.num_visits_init, self.val_init))
            return preferences
        
        if len(history) > 0:
            correct_explaset = self.check_correct_explanations(htn_explaset, sensor_notification)
            if not correct_explaset and sensor_notification != highest_belief_action[0]:
                preferences.add((AgentAskClarificationQuestion(), self.num_visits_init, self.val_init))
                return preferences

        if htn_explaset._other_happen > config._other_happen:
            preferences.add((AgentAskClarificationQuestion(), self.num_visits_init, self.val_init))
            return preferences


        
        # else:
            
        # robot_state = state.object_states[self.robot_id]

        # last_action = None
        # if len(history) > 0:
        #     last_action, last_observation = history[-1]
        #     for objid in last_observation.objposes:
        #         if objid not in robot_state["objects_found"]\
        #            and last_observation.for_obj(objid).pose != ObjectObservation.NULL:
        #             # We last observed an object that was not found. Then Find.
        #             return set({(FindAction(), self.num_visits_init, self.val_init)})

        # if self.no_look:
        #     # No Look action; It's embedded in Move.
        #     preferences = set()
        # else:
        #     # Always give preference to Look
        #     preferences = set({(LookAction(), self.num_visits_init, self.val_init)})
        # for objid in state.object_states:
        #     if objid != self.robot_id and objid not in robot_state.objects_found:
        #         object_pose = state.pose(objid)
        #         cur_dist = euclidean_dist(robot_state.pose, object_pose)
        #         neighbors =\
        #             self.grid_map.get_neighbors(
        #                 robot_state.pose,
        #                 self.grid_map.valid_motions(self.robot_id,
        #                                             robot_state.pose,
        #                                             self.all_motion_actions))
        #         for next_robot_pose in neighbors:
        #             if euclidean_dist(next_robot_pose, object_pose) < cur_dist:
        #                 action = neighbors[next_robot_pose]
        #                 preferences.add((action,
        #                                  self.num_visits_init, self.val_init))
        return preferences

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
        self.step_index = -1
        self.htn_explaset  = None
        super().__init__(object_beliefs)

    def mpe(self, **kwargs):
        return HTNCoachDialState(pomdp_py.OOBelief.mpe(self, **kwargs).object_states)

    def random(self, **kwargs):
        # TODO: reset MCTS environment
        object_states = copy.deepcopy(pomdp_py.OOBelief.random(self, **kwargs).object_states)
        return HTNCoachDialState(self.step_index,copy.deepcopy(self.htn_explaset), object_states)

    def set_htn_explaset(self,explaset):
        self.htn_explaset = copy.deepcopy(explaset)
        return

    def set_step_index(self,step_index):
        self.step_index = copy.deepcopy(step_index)
        return
    
    def append_object_belief(self, objid, attr, val):
        """set_object_belief(self, objid, belief)
        Sets the belief of object to be the given `belief` (GenerativeDistribution)"""
        distribution = self.object_beliefs[objid] #{objectstate:1}
        object_state =  list(distribution.histogram.keys())[0]
        # object_state.__getitem__[attr]
        # dis__setitem__
        # distribution.append(belief)
        object_state.attributes[attr].append(val)
        return
    

def convert_object_belief_to_histogram(init_worldstate_belief):
    '''Convert belief object to histogram dictionary'''
    # TODO:Tian
    #input two lists
    #output a big dict
    from collections import defaultdict
    # print(init_worldstate_belief)
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
    # print(obj_belief_dict)
    return obj_belief_dict, HTN_object_state_dict, observation_prob

def update_belief(HTNCoachDial_problem,action, real_observation, prob_lang, execute=True):
    # print(HTNCoachDial_problem.agent.curr_belief, HTNCoachDial_problem.agent.observation_model,
                    # HTNCoachDial_problem.agent.transition_model)
    # pass
    '''
    In Past: Planner has planned and has chosen action.
    we have gotten reward from the environment for that
    and a real observation given the action. Now we need to update the belief of what we think 
    1) prob of world state is copied, 
    2)explaset-tile update, question-title update?
    2) exp prob is updated based on language recived or not'''
    # Todo: #update step index??
    curr_belief = HTNCoachDial_problem.agent.cur_belief
    ### update the world belief
    

    # key = config.explaset_title
    

    ### update exp prob
    exp = curr_belief.htn_explaset ## feedback is for s1 so update s1's expset
    
    ##perform langauge update is now using original results
    feedback_title = config.feedback_title
    feedback = real_observation.get_lang_objattr(feedback_title)
    
    # return highest_action_PS[0]

    if HTNCoachDial_problem.agent_type != "htn_baseline":
        if action.name == "ask-clarification-question" and feedback != None:
            if exp._other_happen> config._other_happen and not config._last_sensor_notification_dict:
                #update the sensor value
                print("sensor notif is,", config._last_sensor_notification)
                config._last_sensor_notification_dict = languageStateANDSensorUpdate(config._last_sensor_notification, config._output_file_name)
                exp.setSensorNotification(config._last_sensor_notification_dict)

                otherHappen, observation_prob = exp.action_posterior()

                # if feedback == "No":
                #     self._other_happen/=1.005#1.26
            
                # wrong step detect
                if otherHappen > config._other_happen:
                    # wrong step handling
                    print("action posterior after bayseian inference is",  exp._action_posterior_prob)
                    exp.handle_exception()
                    
                # correct step procedure
                else:
                    length = len(exp._explaset)
                    
                    # input step start a new goal (bottom up procedure to create ongoing status)
                    # include recognition and planning
                    exp._delete_trigger = config._real_delete_trigger
                    exp.explaSet_expand_part1(length)

                    # belief state update
                    state = State()
                    state.update_state_belief(exp)
                    # input step continues an ongoing goal
                    # include recognition and planning 
                    exp.explaSet_expand_part2(length)
                    exp._delete_trigger = config._delete_trigger

                    # exp.update_without_language_feedback(prob_lang)
        
            # else:
            # exp.update_with_language_feedback(feedback, exp.highest_action_PS, self._p_l)
            exp.update_with_language_feedback(feedback, exp.highest_action_PS, prob_lang)
    exp.pendingset_generate()
    # compute goal recognition result PROB and planning result PS
    taskhint = exp.task_prob_calculate(HTNCoachDial_problem.hs.real_output_filename)
    print("taskhint is", taskhint.__dict__)
    #output PROB and PS in a file
    exp.print_explaSet()
            
            #output PROB and PS in a file
            ## @II here decide prob of recognizing each task
            # exp.print_explaSet1()
            # exp.print_explaSet()
            # index+=1
            # print "go into the next loop"
            # print 
            # print
        ## remove without langauge update from else to doing everytime before the state change
        # else:
            # exp.update_without_language_feedback(prob_lang)
    


    s1_step_index = curr_belief.step_index 
    s2_step_index = s1_step_index+1
    s1_step_name  = HTNCoachDial_problem.env.human_simulator.return_step(s1_step_index) 
    print("s1 is", s1_step_index, s1_step_name)
    s2_step_name = None
    sensor_notification = None
    if not HTNCoachDial_problem.env.human_simulator.check_terminal_state(s2_step_index):
        _, s2_step_name , sensor_notification = HTNCoachDial_problem.env.human_simulator.curr_step(s1_step_index, None, real_step=True)
        config._last_sensor_notification = s2_step_name
        config._last_sensor_notification_dict = sensor_notification
        
        # s2_step_name = HTNCoachDial_problem.env.human_simulator.return_step(s2_step_index) 
        ## add last sensor_notification to explaset.
        key = config.explaset_title
        attribute =  key.split("-")[1]
        curr_belief.append_object_belief(key, attribute, s2_step_name)
    
    
        ## find exp for s2
        # sensor_notification = HTNCoachDial_problem.hs.sensor_notification_dict[s2_step_name]

        exp.setSensorNotification(sensor_notification)
        otherHappen, observation_prob = exp.action_posterior(execute)

        '''Executing transition of exp'''
        if otherHappen > config._other_happen:
                    # wrong step handling
            # print("action posterior after bayseian inference is",  exp._action_posterior_prob)
            exp.handle_exception()
            
        # correct step procedure
        else:
            # print("updating explaset")
            length = len(exp._explaset)
            
            # input step start a new goal (bottom up procedure to create ongoing status)
            # include recognition and planning
            exp._delete_trigger = config._real_delete_trigger
            exp.explaSet_expand_part1(length)

            # belief state update
            world_state = State()
            world_state.update_state_belief(exp)
            ## TODO: user the above function to update belief over the state.
            
            # input step continues an ongoing goal
            # include recognition and planning 
            exp.explaSet_expand_part2(length)

            ##incorporate probaility of not getting language feedback
            if HTNCoachDial_problem.agent_type != "htn_baseline":
                exp.update_without_language_feedback(prob_lang)
            exp._delete_trigger = config._delete_trigger
            

                    
        exp.pendingset_generate()
        
        # compute goal recognition result PROB and planning result PS
        if execute:
            # taskhint = exp.task_prob_calculate(HTNCoachDial_problem.hs.real_output_filename)
            taskhint = exp.task_prob_calculate("")
            # exp.cout_taskhintInTable(taskhint)
            # exp.print_explaSet()
        # else:
        #     taskhint = exp.task_prob_calculate(HTNCoachDial_problem.hs.mcts_output_filename)
        #     exp.mcts_print_explaSet()

        
    worldstate_belief, _, _ = convert_object_belief_to_histogram(list(db._state.find()))
    
    for key, hist in worldstate_belief.items():
        curr_belief.set_object_belief(key, hist)
    
    ## add last sensor_notification to explaset.
    # key = config.explaset_title
    # attribute =  key.split("-")[1]
    # curr_belief.append_object_belief(key, attribute, last_sensor_notification)

    # step_index+=1
    # if not HTNCoachDial_problem.env.human_simulator.check_terminal_state(step_index):
    #     next_sensor_notification = HTNCoachDial_problem.env.human_simulator.return_step(step_index) 
    #     ## add last sensor_notification to explaset.
    #     key = config.explaset_title
    #     attribute =  key.split("-")[1]
    #     curr_belief.append_object_belief(key, attribute, next_sensor_notification)

    print("s1_step_name ",s1_step_name , "s2_step_name ", s2_step_name  )
    if s2_step_name != None:
        
        # if s1_step_name == exp.highest_action_PS[0]:
        #     step = "Yes"
        # else:
        #     step = "No"
        # # print("step is", step)

        # if real_observation.get_lang_objattr(config.feedback_title) == None:
        #     exp.update_without_language_feedback(prob_lang)
        # else:
        #     exp.update_with_language_feedback(step, exp.highest_action_PS, prob_lang)

        # HTNCoachDial_problem.agent.cur_belief.htn_explaset = exp
        
        HTNCoachDial_problem.agent.cur_belief.set_htn_explaset(exp)
    else:
        exp = None
    HTNCoachDial_problem.agent.cur_belief.set_step_index(s2_step_index)
    return exp,s2_step_index, sensor_notification

class HTNCoachDial(pomdp_py.POMDP):
    """
    In fact, creating a HTNCoachDial class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, init_true_state, init_belief, observation_prob, explaset = None, hs = None, args = None):
        """init_belief is a Distribution."""
        self.agent_type = args.agent_type
        self.hs = hs
        self.reward_output_filename = None
        # self.hs.read_files()
        # self.hs.goal_selection()

        reward_model = RewardModel(self.hs, args)
        num_visits=10
        val_init = reward_model.goal_reward
        action_prior = ActionPrior(num_visits, val_init)

        agent = pomdp_py.Agent(init_belief,
                               PreferredPolicyModel(action_prior, num_visits, val_init),
                               TransitionModel(self.hs),
                               HTNCoachDialObservationModel(),
                               reward_model)
        # env = pomdp_py.Environment(init_true_state,
        #                            TransitionModel(),
        #                            RewardModel())
        # self, human_simulator, explaset, init_state, RewardModel, TransitionModel
        env =  HTNCoachDialEnvironment(self.hs, explaset, init_true_state,
                                   reward_model, TransitionModel(self.hs))
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

def planner_one_loop(HTNCoachDial_problem, planner, nsteps=3, debug_tree=True, discount=0.95, gamma = 1.0, total_reward = 0, total_discounted_reward = 0, i=0, true_state = None, prob_lang = 0.95, num_question_asked=0):
    # planner._db = db
    # if agent == standard
    # action = planner.plan(HTNCoachDial_problem.agent)
    # action = Action("wait") 
    # action_dict={
    #     1:0
    #     2:0
    #     3:1
    #     4:1
    #     5:
    #     6:
    #     7:
    #     8:
    #     9:
    #     10:
    #     11:
    #     12:
    #     13:
    # }
    # action_policy = [AgentAskClarificationQuestion(),Action("wait"),AgentAskClarificationQuestion(),AgentAskClarificationQuestion(),Action("wait")]
    print("==== Step %d ====" % (i+1))

    if HTNCoachDial_problem.agent_type == "standard":
        # if i == 0:
        #     action = Action("wait")
        # else:
        # action = action_policy[i]
        action = planner.plan(HTNCoachDial_problem.agent)
    elif HTNCoachDial_problem.agent_type == "htn_baseline":
        action = Action("wait")
    elif HTNCoachDial_problem.agent_type == "fixed_always_ask":
        action = AgentAskClarificationQuestion()
    elif HTNCoachDial_problem.agent_type == "random":
        action = HTNCoachDial_problem.agent.policy_model.sample(HTNCoachDial_problem.agent.cur_belief)
        
    #random
    # action = choses_action_randomly()
    #fixed policy -always asking question
    # belief =
    # fixed plocu -  always wait() (HTN baseline)
    # action = A

    '''Restoring the state for next iteration, env variable in HTNcoachproblem should be reset'''

    pipeline = [ {"$match": {}}, 
                {"$out": "state"},
    ]
    db._backup_state.aggregate(pipeline)

    pipeline = [ {"$match": {}}, 
                {"$out": "sensor"},
    ]
    db._backup_sensor.aggregate(pipeline)
    
    if debug_tree:
        dd = TreeDebugger(HTNCoachDial_problem.agent.tree)
        # import pdb; pdb.set_trace()
        TreeDebugger(HTNCoachDial_problem.agent.tree).pp

    if action == AgentAskClarificationQuestion():
        ##update the question
        curr_belief = HTNCoachDial_problem.agent.cur_belief
        # exp = curr_belief.htn_explaset ## feedback is for s1 so update s1's expset

        # highest_action_PS = ["", float('-inf')]

        # for k, v in exp._action_posterior_prob.items():
        #     if v > highest_action_PS[1]:
        #         highest_action_PS = [k,v]
        # #TODO: actual epxlaset not updated
        # exp.highest_action_PS = highest_action_PS 
        
        highest_action = action.update_question_asked(curr_belief)
        curr_belief.htn_explaset.highest_action_PS = highest_action
        num_question_asked+=1

    
    ## true state, get from simulator
    # if i == 0:
    #     curr_step = HTNCoachDial_problem.hs.curr_step("none", action, True)
    # else:
    #     curr_step = HTNCoachDial_problem.hs.curr_step(curr_step, action, True)
    ## env state update accordnig to basics state transition
    true_state = copy.deepcopy(HTNCoachDial_problem.env.state)
    env_reward = HTNCoachDial_problem.env.state_transition(action, execute=True)
    true_next_state = copy.deepcopy(HTNCoachDial_problem.env.state)
    
    # env_reward = HTNCoachDial_problem.env.reward_model.sample(HTNCoachDial_problem.env.state, action, None)
    # env_reward = HTNCoachDial_problem.environment_reward_model.sample(HTNCoachDial_problem.env.state, action, None)
    ##TODO: need env state variable in the coachdial probelm. update the state variable with true state info.
    with open(HTNCoachDial_problem.reward_output_filename, 'a') as f:
        f.write("True state: %s" % true_state + "\n")
        f.write("Belief: %s" % HTNCoachDial_problem.agent.cur_belief.__str__ + "\n")
        f.write("Action: %s" % str(action) + "\n")
        f.write("Reward: %s" % str(env_reward)+ "\n")

    '''Print Statements'''
    # print("True state: %s" % true_state)
    # # print("True state: %s" % HTNCoachDial_problem.env.state)
    # print("Belief: %s" % str(HTNCoachDial_problem.agent.cur_belief))
    # print("Action: %s" % str(action))
    # print("Reward: %s" % str(env_reward))
    total_reward += env_reward
    total_discounted_reward += env_reward * gamma
    gamma *= discount

    with open(HTNCoachDial_problem.reward_output_filename, 'a') as f:
        f.write("Reward (Cumulative): %s" % str(total_reward) + "\n")
        f.write("Reward (Cumulative Discounted): %s" % str(total_discounted_reward) + "\n")

        if isinstance(planner, pomdp_py.POUCT):
            f.write("__num_sims__: %d" % planner.last_num_sims + "\n")
            f.write("__plan_time__: %.5f" % planner.last_planning_time + "\n")
            f.write("\n\n\n")
        if isinstance(planner, pomdp_py.PORollout):
            print("__best_reward__: %d" % planner.last_best_reward)
    # print("Reward (Cumulative): %s" % str(total_reward))
    # print("Reward (Cumulative Discounted): %s" % str(total_discounted_reward))
        
    
        
    # Let's create some simulated real observation; Update the belief
    # Creating true observation for sanity checking solver behavior.
    # In general, this observation should be sampled from agent's observation model. 
    real_observation = HTNCoachDial_problem.env.provide_observation(HTNCoachDial_problem.agent.observation_model,
                                                              action)
    # real_observation = TigerObservation(HTNCoachDial_problem.env.state.name)
    print(">> Observation: %s" % real_observation)
    HTNCoachDial_problem.agent.update_history(action, real_observation)

    # If the planner is POMCP, planner.update also updates agent belief.
    # print("==== Step", i+1,"Tree before update:", HTNCoachDial_problem.agent.tree)
    planner.update(HTNCoachDial_problem.agent, action, real_observation)
    if isinstance(planner, pomdp_py.POUCT):
        print("Num sims: %d" % planner.last_num_sims)
        print("Plan time: %.5f" % planner.last_planning_time)
    # print("==== Step", i+1,"Tree after update:", HTNCoachDial_problem.agent.tree )
    # TODO: belief update for now update the explaset

    # if true_next_state.step_index == 5:
    #     print("here")
    #     action = AgentAskClarificationQuestion()
    exp,step_index, sensor_notifications = update_belief(HTNCoachDial_problem,
                    action, real_observation, prob_lang) 

    HTNCoachDial_problem.env.state.set_htn_explaset(exp)
    HTNCoachDial_problem.env.state.set_step_index(step_index)
    ExecuteSequence = list(HTNCoachDial_problem.agent.cur_belief.object_beliefs[config.explaset_title].get_histogram().keys())[0].attributes['action']
    if not ExecuteSequence:
        # HTNCoachDial_problem.agent.cur_belief.object_beliefs[config.explaset_title].get_histogram()
        HTNCoachDial_problem.env.state.set_object_state(config.explaset_title, ExecuteSequence)
    if sensor_notifications:
        for sensor_notification in sensor_notifications:
                notif = sensor_notification['object']+"-"+sensor_notification['attribute']
                notif_state = HTNCoachDial_problem.env.state.get_object_state(notif)
                notif_ObjectAttrAndLangState = ObjectAttrAndLangState('object',{'obj_name':sensor_notification['object'], 'attr_name':sensor_notification['attribute'], 'attr_val':sensor_notification['obj_att_value'] })
                # (object|{'obj_name': 'hand_1', 'attr_name': 'dry', 'attr_val': 'yes'}(object|{'obj_name': 'hand_1', 'attr_name': 'dry', 'attr_val': 'yes'}
                notif_state.__setitem__('object', sensor_notification['object'])
                notif_state.__setitem__('attr_name', sensor_notification['attribute'])
                notif_state.__setitem__('attr_val',sensor_notification['obj_att_value'])
    # explaset_state_dict = {"object":self.explaset_title[0], "attribute": self.explaset_title[1] , "obj_att_value": step}
    # self.update_true_state(self.HTNCoachDial_problem.env.state, explaset_state_dict)## update explaset action which stores the current sensor (TODO: MAYBE remove sensor state)
    
    
    # if isinstance(HTNCoachDial_problem.agent.cur_belief, pomdp_py.Histogram):
    #     new_belief = pomdp_py.update_histogram_belief(HTNCoachDial_problem.agent.cur_belief,
    #                                                     action, real_observation,
    #                                                     HTNCoachDial_problem.agent.observation_model,
    #                                                     HTNCoachDial_problem.agent.transition_model)
    #     HTNCoachDial_problem.agent.set_belief(new_belief)

    # if action.name.startswith("open"):
        # Make it clearer to see what actions are taken until every time door is opened.
        # print("\n")
    # explaset_title = config.explaset_title
    # explaset_title_split = explaset_title.split("-")
    # # print)
    # sensor_state = HTNCoachDial_problem.env.state.get_object_state(explaset_title)
    # sensor_notification = sensor_state.attributes[explaset_title_split[1]]

    i+=1
    return total_reward, total_discounted_reward,i, gamma,num_question_asked


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
