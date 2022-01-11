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

class HTNCoachDialState(pomdp_py.State):
    # TODO:check this?
    def __init__(self, world_state, exp): 
        self._world_state = _world_state
        self._explaset = exp #use exp._prob and exp._action_posterior
        self._question_asked =  None
        self._sensor_notification = None
    def __hash__(self):
        return hash(self.world, self.exp, self._question_asked, self._sensor_notification)
    def __eq__(self,other):
        if isinstance(self,State):
             ## TODO:check that object is same
            return self._world_state == other._world_state\
                and list(self._explaset) == list(other._explaset)\
                and all([expla1._prob == expla2._prob for expla1, expla2 in zip(self._explaset, other._explaset)])\
                and all([expla1._start_task == expla2._start_task for expla1, expla2 in zip(self._explaset, other._explaset)])\
                and self._explaset._action_posterior_prob == other.exp._action_posterior_prob\
                and self._explaset._prior == other._explaset._prior\
                and self._explaset._otherHappen == other._explaset._otherHappen\
                and self._question_asked == other._question_asked\
                and self._sensor_notification == other._sensor_notification
        # and self.exp == self.exp\ 
        # self._non_happen = non_happen
        # self.__sensor_notification = []
        # self._output_file_name = output_file_name
        #  = {}
        # self._language_notification = []
        # self.highest_action_PS = []
        # self._forest = forest
        # self._pendingSet = pendingSet
        # self._start_task = start_task
        else:
            return False
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "State(%s | %s | %s | %s | %s | %s)" % (str(self._world_state), str(self._explaset.__dict__), self._question_asked, self._sensor_notification)
    

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

    def update_question_asked(state):
        highest_action_PS = ["", float('-inf')]
        for k, v in state._explaset._action_posterior_prob.items():
            if v > highest_action_PS[1]:
                highest_action_PS = [k,v]
        #TODO: actual epxlaset not updated
        # exp.highest_action_PS = highest_action_PS 

        return highest_action_PS
        # return state

# class AgentWaitAction(Action):
#     """
#     Robot action for waiting for user's utterances.
#     """
#     def __init__(self):
#         super().__init__("wait")


class Observation(pomdp_py.Observation):
    def __init__(self, name):
        # self.name = name
        self._world_state = {}
        self._question_asked = None
        self._explaset = None
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "TigerObservation(%s)" % self.name

# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0.15):
        self.noise = noise

    def probability(self, observation, next_state, action):
        if action.name == "listen":
            if observation.name == next_state.name: # heard the correct growl
                return 1.0 - self.noise
            else:
                return self.noise
        else:
            return 0.5

    def sample(self, next_state, action):
        if action.name == "listen":
            thresh = 1.0 - self.noise
        else:
            thresh = 0.5

        if random.uniform(0,1) < thresh:
            return TigerObservation(next_state.name)
        else:
            return TigerObservation(next_state.other().name)

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [TigerObservation(s) for s in {"tiger-left", "tiger-right"}]

# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
    def __init__(self, hs):
        self.human_simulator = hs

    def probability(self, next_state, state, action):
        return hs.probablity(next_state, state)

    def sample(self, state, action):
        next_state = HTNCoachDialState()
        if action.name == "ask-clarification-question":
            question_asked = action.update_question_asked(state)
        else:
            question_asked = None
        next_state._question_asked = question_asked
        next_state._sensor_notification = hs.curr_step(self, state, action)
        return next_state
        
    def get_all_states(self):
        ##TODO: states are notifs or world states? need to convert notifs to states.
        return hs.get_all_states()

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
        if len(self.human_simulator.notifs[self.human_simulator.index_test_case]) == 0:
            return 10
        elif action.name == "wait":
            return -1
        elif action.name == "ask-clarification-question" and state._sensor_notification == state._question_asked:
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
    ACTIONS = {Action("wait"), AgentAskClarificationQuestion }
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



def convert_object_belief_to_histogram(init_belief):
    '''Convert belief object to histogram dictionary'''
    # TODO:Tian
    #input two lists
    #output a big dict
    pass



class HTNCoachDial(pomdp_py.POMDP):
    """
    In fact, creating a HTNCoachDial class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, obs_noise, init_belief):
        """init_belief is a Distribution."""
        
        self.hs = human_simulator()
        self.hs.goal_selection()

        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(self.hs),
                               ObservationModel(obs_noise),
                               RewardModel(self.hs))
        # env = pomdp_py.Environment(init_true_state,
        #                            TransitionModel(),
        #                            RewardModel())
        env = None
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
