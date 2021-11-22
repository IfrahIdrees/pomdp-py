"""
CoachDial(n,k) problem

Origin: Heuristic Search Value Iteration for POMDPs (UAI 2004)

Description:

State space:

    Position {(1,1),(1,2),...(n,n)}
    :math:`\\times` RockType_1 :math:`\\times` RockType_2, ..., :math:`\\times` RockType_k
    where RockType_i = {Good, Bad}
    :math:`\\times` TerminalState

    (basically, the positions of rocks are known to the robot,
     but not represented explicitly in the state space. Check_i
     will smartly check the rock i at its location.)

Action space:

    North, South, East, West, Sample, Check_1, ..., Check_k
    The first four moves the agent deterministically
    Sample: samples the rock at agent's current location
    Check_i: receives a noisy observation about RockType_i
    (noise determined by eta (:math:`\eta`). eta=1 -> perfect sensor; eta=0 -> uniform)

Observation: observes the property of rock i when taking Check_i.

Reward: +10 for Sample a good rock. -10 for Sampling a bad rock.
        Move to exit area +10. Other actions have no cost or reward.

Initial belief: every rock has equal probability of being Good or Bad.
"""
import random
import math
import sys
import copy
import argparse
import os
import logging

import numpy as np
import pandas as pd

import pomdp_py

EPSILON = 1e-9
MAX_INSTRUCTION_STEP = 1
#IS_RANDOM_AGENT = False
#IS_HEURISTIC_AGENT = False
update_i_step_condition_dict ={1:{'w':'0000'}, 2:{'w':'0001'}, 3:{'w':'0011'}, 4:{'w':'1011'}, 5:{'w':'1111'}}
HYPERPARAMS = {}
# TIME_STEP = 0


def parseArguments():
    parser = argparse.ArgumentParser()

    # Necessary variables
    #parser.add_argument("is_random_agent", action="store_true")
    #parser.add_argument("is_heuristic_agent", action="store_true")
    parser.add_argument("--belief", type=str, default="uniform")
    parser.add_argument("--agent_type", type=str, default="standard",
                        help="standard, random, heuristic")
    parser.add_argument("--nsteps", type=int, default=200)
    parser.add_argument("--num_sims", type=int, default=500,
                        help="num_sims for POMCP")
    parser.add_argument("--give_next_instr_reward", type=int, default=20)
    parser.add_argument("--give_next_instr_penalty", type=int, default=-10)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--print_log", action="store_true")
    parser.add_argument("--output_results", action="store_true")
    args = parser.parse_args()

    # I/O parameters
    output_name, output_dir = get_output_path(args)
    parser.add_argument("--output_name", type=str, default=output_name)
    parser.add_argument("--output_dir", type=str, default=output_dir)
    args = parser.parse_args()

    return args

def get_logger(args):
    os.makedirs("../../../logs/", exist_ok=True)
    os.makedirs("../../../logs/demo", exist_ok=True)
    os.makedirs("../../../logs/demo/{}".format(args.agent_type), exist_ok=True)
    os.makedirs("../../../logs/demo/{}/{}".format(args.agent_type, args.output_name), exist_ok=True)
    logging.basicConfig(level = logging.DEBUG, \
            format = '%(asctime)s %(levelname)s: %(message)s', \
            datefmt = '%m/%d %H:%M:%S %p', \
            filename = '../../../logs/demo/{}/{}/{}.log'.format(
                args.agent_type, args.output_name, args.output_name
            ), \
            filemode = 'w'
    )
    return logging.getLogger()

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def get_output_path(args):
    output_name = "R{}_B{}_N{}_S{}_instrR{}_instrP{}".format(
            args.num_runs,
            args.belief,
            args.nsteps, 
            args.num_sims,
            args.give_next_instr_reward, 
            args.give_next_instr_penalty,
    )
    output_dir = "../../../outputs/demo/{}/{}".format(args.agent_type, output_name)

    os.makedirs("../../../outputs", exist_ok=True)
    os.makedirs("../../../outputs/demo", exist_ok=True)
    os.makedirs("../../../outputs/demo/{}".format(args.agent_type), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return output_name, output_dir

def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class ObjectState:
    
    TABLE = "0"
    STOVE = "1"
#    OFF = 0
#    ON  = 1

    @staticmethod
    def change_location(location):
        if location == TABLE:
            return STOVE
        else:
            return TABLE
    
#    @staticmethod
#    def change_state(curr_state):
#        if curr_state == ON:
#            return OFF
#        else:
#            return ON

    @staticmethod
    def initialize():
        """
        Assume appliances are off, and objects (e.g. egg, oil, pan) are on the table.
        """
        return "0"


class ApplianceState:
    """
    The world state (s) is a binary vector representing objects' status.
    """

    ON = 1
    OFF = 0

    @staticmethod
    def change_state(appliance_state):
        if appliance_state == ON:
             appliance_state = OFF
        else:
            appliance_state = ON
        return appliance_state
    
    @staticmethod
    def random(p=0.5):
        if random.uniform(0,1) >= p:
            return ON
        else:
            return OFF

    @staticmethod
    def initialize():
        """
        Currently we suppose everything is "off". This will be more complex later.
        """
        return 0
        

class State(pomdp_py.State):
    def __init__(self, instruction_step, user_intent, user_activity, world_state):
        """
        TODO: Fix the comment here?
        positions (list of tuples): a list (x,y) positions of the appliances in the kitchen.
        rocktypes: tuple of size k. Each is either Good or Bad.
        is_terminal (bool): The robot is at the terminal state.
        """
        self.instruction_step = instruction_step
        self.user_intent = user_intent
        #self.goal = goal
        self.user_activity = user_activity
        self.world_state = world_state
        self.is_terminal = False
        #self.time_step_entered = time_step

    def __hash__(self):
        return hash((self.instruction_step, self.user_intent, self.user_activity, self.world_state, self.is_terminal))
    def __eq__(self, other):
        if isinstance(other, State):
            return self.instruction_step == other.instruction_step\
                and self.user_intent == other.user_intent\
                and self.user_activity == other.user_activity\
                and self.world_state == other.world_state\
                and self.is_terminal == other.is_terminal
                # and self.time_step_entered == other.time_step_entered
        else:
            return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "State(%s | %s | %s | %s | %s)" % (str(self.instruction_step), str(self.user_intent), str(self.user_activity), str(self.world_state), str(self.is_terminal))
    
    def other_user_intent(self):
        if self.user_intent.name == "saying-nothing":
            return UserAffirmDoingStepIntent()
        else:
            return UserSayingNothingIntent()

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

"""
TODO: How should we track human's move action? 
class MoveAction(Action):
    EAST = (1, 0)  # x is horizontal; x+ is right. y is vertical; y+ is up.
    WEST = (-1, 0)
    NORTH = (0, -1)
    SOUTH = (0, 1)
    def __init__(self, motion, name):
        if motion not in {MoveAction.EAST, MoveAction.WEST,
                          MoveAction.NORTH, MoveAction.SOUTH}:
            raise ValueError("Invalid move motion %s" % motion)
        self.motion = motion
        super().__init__("move-%s" % str(name))

MoveEast = MoveAction(MoveAction.EAST, "EAST")
MoveWest = MoveAction(MoveAction.WEST, "WEST")
MoveNorth = MoveAction(MoveAction.NORTH, "NORTH")
MoveSouth = MoveAction(MoveAction.SOUTH, "SOUTH")
"""

# Adding actions for robot and human
class AgentGiveNextInstructionAction(Action):
    """
    Robot action for giving the next instruction
    """
    ##@II need to code that it increases instruction by 1. As in MoveAction East is (1,0) (just defining)

    def __init__(self):
        super().__init__("give-next-instruction")

class AgentWaitAction(Action):
    """
    Robot action for waiting for user's utterances.
    """
    def __init__(self):
        super().__init__("wait")

class AgentByeAction(Action):
    """
    Robot action for saying bye (terminating the interaction).
    """
    def __init__(self):
        super().__init__("bye")

class AgentFlipWorldStateValueAction(Action):
    """
    Robot action for flipping the world state (e.g. if the stove is on, turn it off)
    NOTE: Not included in the current demo
    """
    ##@II should include change state/ change location of the state. (just defining)
    ## I can change. 
    def __init__(self):
        super().__init__("flip-world-state-value")

class AgentRepeatAction(Action):
    """
    Robot action for repeating the instruction
    NOTE: Not included in the current demo
    """
    def __init__(self):
        super().__init__("repeat")

class AgentStartTimerAction(Action):
    """
    Robot action for starting the timer
    NOTE: Not included in the current demo
    """
    def __init__(self):
        super().__init__("start-timer")

class UserIntent:
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, UserIntent):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "UserIntent(%s)" % self.name

class UserAffirmDoingStepIntent(UserIntent):
    """
    User action for following the agent's instruction
    """
    ##@II (should we have some thing to store progress of current step?)
    ## progress bar...
    def __init__(self):
        super().__init__("affirm-doing-step")

class UserSayingNothingIntent(UserIntent):
    """
    User action for saying nothing (e.g. they could be receiving the agent's instructions)
    """
    def __init__(self):
        super().__init__("saying-nothing")


class UserActivity:
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, UserActivity):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "UserActivity(%s)" % self.name

class UserDoingNothingActivity(UserActivity):
    """
    User activity for doing nothing.
    """
    def __init__(self):
        super().__init__("doing-nothing")

class UserPuttingSaucepanOnStoveActivity(UserActivity):
    """
    User activity for putting saucepan on stove.
    """
    def __init__(self):
        super().__init__("putting-saucepan-on-stove")

class UserAddingOilInPanActivity(UserActivity):
    """
    User activity for adding oil in the pan.
    """
    def __init__(self):
        super().__init__("adding-oil-in-the-pan") 

class UserTurningOnStoveActivity(UserActivity):
    """
    User activity for tuning on the stove.
    """
    def __init__(self):
        super().__init__("turning-on-stove")

class UserCrackingEggInPanActivity(UserActivity):
    """
    User activity for cracking egg in the pan.
    """
    def __init__(self):
        super().__init__("cracking-egg-in-pan")

class UserTurningOffStoveActivity(UserActivity):
    """
    User activity for turning off the stove.
    """
    def __init__(self):
        super().__init__("turning-off-stove")


class CDTransitionModel(pomdp_py.TransitionModel):

    """ The model is stochastic but transitions are limited """
    def __init__(self, k, transition_constant):
        """Read in the transition table and return the corresponding world state in the simulation table 

        Args:
            k (integer): Number of objects in the kitchen
            transition_constant (Integer): The probability for a world state if it matches the simulation table
            simulation_table (pd.DataFrame): 
        
        Returns:
            world_state (binary vector)
        """
        self.k = k
        self.transition_constant = transition_constant

        # Total number of combinations of world state
        self.num_world_states = 2**k
        
        # Pandas dataframe with sensical actions and transitions
        self.simulation_table = self._read_simulation_table()

    def _read_simulation_table(self):
        """Read simulation table and return a pd.DataFrame"""
        table = pd.read_csv('simulation_table.csv', dtype={"w": str, "w'": str})
        return table

    def _change_world(self, prev_state, prev_agent_action):
        """Return the corresponding world state in the simulation table

        Args:
            prev_state (State): state at t-1
            prev_agent_action (Action): agent's action at t-1

        Returns:
            curr_state: state at t if the transition is sensical (else, return prev_state)
        """
        # Get the strings for agent action and user intent for querying the table
        # @II shouldn't line 359-362 be prev_state instead of state?
        # logger.info("TIMESTEP IS %s", str(TIME_STEP))
        # global TIME_STEP
        prev_i_step = prev_state.instruction_step
        prev_u = prev_state.user_intent.name
        prev_w = prev_state.world_state
        prev_actv = prev_state.user_activity.name
        prev_a = prev_agent_action.name
        
        # lookup conditions
        condition_i = self.simulation_table.i==prev_i_step
        condition_u = self.simulation_table.u==prev_u
        condition_w = self.simulation_table.w==prev_w
        condition_actv = self.simulation_table.actv==prev_actv
        condition_a = self.simulation_table.a==prev_a

        # logger.info(condition_i)
        # logger.info(condition_u)
        # logger.info(condition_w)
        # logger.info(condition_actv)
        # logger.info(condition_a)
        condition_all = condition_i & condition_u & condition_w & condition_actv & condition_a
        if any(condition_all):
            # if i,u,w,actv,a are in the simulation table, return (i',u', w',actv'), i.e., (i_t,u_t, w_t,actv_t)
            # Simulation table lookup
            candidate_states = self.simulation_table[condition_all]
            num_candidate_states = len(candidate_states)

            # Random sampling from all the possible next states
            sampled_state_id = np.random.choice(candidate_states.index)
            sampled_state = self.simulation_table.iloc[sampled_state_id]

            curr_i = int(sampled_state["i'"])

            curr_u_name = sampled_state["u'"]
            curr_u = UserIntent(curr_u_name) #TODO: Change to specific action, e.g. UserDoingNothingActivity

            curr_w = sampled_state["w'"]

            curr_actv_name = sampled_state["actv'"]
            curr_actv = UserActivity(curr_actv_name)
            # logger.info("TIMESTEP IS %s", str(TIME_STEP))
            return State(curr_i, curr_u, curr_actv, curr_w)
        else:
            # otherwise, return previous state
            return prev_state
        
    def probability(self, next_state, state, agent_action):
        """[summary]

        Args:
            next_state ([type]): [description]
            state ([type]): [description]
            agent_action ([type]): [description]

        Returns:
            [type]: [description]
        """
        if next_state != self.sample(state, action):
            return (1 - self.transition_constant) / self.num_world_states
        else:
            return self.transition_constant

    def sample(self, prev_state, prev_action):
        """Return the expected next state and check for terminal condition. 
        
        Args:
            state (State): The previous state.
            action (Action): The previous agent action.
        
        Returns:
            (State): The next state.
        """
        # global TIME_STEP
        # TIME_STEP+=1
        prev_is_terminal = prev_state.is_terminal
        logger.info("In transition function prev state is %s and prev action is %s", prev_state.__str__(), prev_action.__str__())

        ## condition to check if it is the end. 
        if prev_is_terminal:
            # Already terminated, so no state transition happens.
            prev_is_terminal = True
            return prev_state
        else:
            # Otherwise, retrieve the expected (world state, instr_step)
            next_state = self._change_world(prev_state, prev_action)

            # Check for the terminal conditions:
            #   1. the curr instruction step is larger than the max instruction step
            #   2. prev_action is bye.

            ##@II should remain stuck after bye action
            # if (next_state.instruction_step > MAX_INSTRUCTION_STEP and
                # prev_action.name == "bye")  :
            if(prev_action.name == "bye"):
                prev_is_terminal = True
                next_state.is_terminal = True
            
            return next_state #@II returned without isterminal being set
        
    def argmax(self, state, action):
        """Returns the most like next state"""
        return np.argmax(self.sample(state, action))


class Observation(pomdp_py.Observation):
    def __init__(self, user_intent, world_state, user_activity):
        self.user_intent = user_intent
        self.world_state = world_state
        self.user_activity = user_activity
    def __hash__(self):
        return hash((self.user_intent, self.world_state, self.user_activity))
    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.user_intent == other.user_intent and self.world_state == other.world_state and self.user_activity == other.user_activity
        else:
            return False
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return "Observation(%s | %s | %s)" % (str(self.user_intent), str(self.world_state), str(self.user_activity))

class CDObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0):
        # NOTE: set the noise in the future
        self.noise = noise

    def probability(self, observation, curr_state):
        #@II: for consistency make it next state. this is the observation we get in the next state.
        if curr_state.user_intent == observation.user_intent:
            return 1.0 - self.noise
        else:
            return self.noise

    def sample(self, curr_state, action):
#        print("cur_state", curr_state, "\naction", action)
        
        
        thresh = 1.0 - self.noise
        
        if random.uniform(0,1) < thresh:
            return Observation(curr_state.user_intent, curr_state.world_state, curr_state.user_activity)
        else:
            return Observation(curr_state.other_user_intent(), curr_state.world_state, curr_state.user_activity)

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        
        all_obs = []
        for w in range(16):
            binary_w = bin(w)[2:]  # "0b10" -> "10"
            w = "{:04d}".format(int(binary_w))  # "10" -> "0010"
            for u in [UserSayingNothingAction(), UserAffirmDoingStepAction()]:
                all_obs.append(u, w)
        return all_obs


class CDRewardModel(pomdp_py.RewardModel):
    def __init__(self, update_i_step_condition_dict):
#        self._appliance_locs = appliance_locs
        self._last_instruction_step = False
        self._update_i_step_condition_dict = update_i_step_condition_dict
	#update_i_step_condition_dict ={instruction_step: {end_world_state}}
	#self.instruction_step = instruction_step
#        self.user_intent = user_intent
#        self.user_activity = user_activity
#        self.world_state = world_state
#        self.is_terminal = False

    def sample(self, state, action, next_state, normalized=False, **kwargs):
        # deterministic
        # if state.is_terminal and state.instruction_step > MAX_INSTRUCTION_STEP:
        #     return 0  # terminated. No reward
        # elif state.is_terminal and state.instruction_step <= MAX_INSTRUCTION_STEP:
        #     return -100 #said bye before completing the step
        if state.is_terminal and state.instruction_step > MAX_INSTRUCTION_STEP:
            return 200

        if state.is_terminal:
            return 0

        if IS_RANDOM_AGENT:
            return -1 # Random agent. Always return -1.

        if isinstance(action, AgentGiveNextInstructionAction):
            if (state.instruction_step != 6 and 
            update_i_step_condition_dict[state.instruction_step]['w'] == state.world_state and 
            state.user_activity == "doing-nothing"):
                return args.give_next_instr_reward
            else:
                return args.give_next_instr_penalty
        elif isinstance(action, AgentByeAction):
            if next_state.instruction_step > MAX_INSTRUCTION_STEP:
                return 100
            else:
                return -100
        return -1


    def argmax(self, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError

    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError


class CDPolicyModel(pomdp_py.RolloutPolicy):
    """Simple policy model according to problem description."""


    def __init__(self):
        self._all_actions = {AgentGiveNextInstructionAction(), AgentWaitAction(), AgentByeAction()}

    def sample(self, state, normalized=False, **kwargs):
        return random.sample(self.get_all_actions(state=state), 1)[0]

    def probability(self, action, state, normalized=False, **kwargs):
        raise NotImplementedError

    def argmax(self, state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        raise NotImplementedError

    def get_all_actions(self, **kwargs):
        return  self._all_actions

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state), 1)[0]


class CoachDialProblem(pomdp_py.POMDP):

    @staticmethod
    def random_free_location(n, not_free_locs):
        """returns a random (x,y) location in nxn grid that is free."""
        while True:
            loc = (random.randint(0, n-1),
                   random.randint(0, n-1))
            if loc not in not_free_locs:
                return loc

    def in_exit_area(self, pos):
        return pos[0] == self._n

    ## other helper functions such as checking that current instruction step is final instruction step. 

    @staticmethod
    def generate_instance(n, k):
        """Returns init_state and rock locations for an instance of CoachDial(n,k)"""

        # TODO: Add user position + appliance position later (not used for the current demo)
        user_position = (0, random.randint(0, n-1))

        ##@II why appliance location.
        appliance_locs = {}  
        for i in range(k):
            loc = CoachDialProblem.random_free_location(n, set(appliance_locs.keys()) | set({user_position}))
            appliance_locs[loc] = i
                
        # Initial world state: [0,table,table,table] -> "0000"
        world_state = ""
        for _ in range(k): 
            world_state += ObjectState.initialize()

        # Initial State Space (instruction_step, user_intent, user_activity, world_state)
        instruction_step = 1
        user_intent = UserSayingNothingIntent()
        user_activity = UserDoingNothingActivity()
        init_state = State(instruction_step, user_intent, user_activity, world_state) 

        return init_state, appliance_locs

    def print_state(self):
        print("i | u | actv | w | is_terminal")
        print(self.env.state.__str__())
    
    def log_state(self):
        logger.info("i | u | actv | w | is_terminal")
        logger.info(self.env.state.__str__())
 
    def __init__(self, n, k, init_state, init_belief, transition_constant, update_i_step_condition_dict, appliance_locs):
        self._n, self._k = n, k
        agent = pomdp_py.Agent(init_belief,
                               CDPolicyModel(),
                               CDTransitionModel(k, transition_constant),
                               CDObservationModel(),
                               CDRewardModel(update_i_step_condition_dict))
        env = pomdp_py.Environment(init_state,
                                   CDTransitionModel(k, transition_constant),
                                   CDRewardModel(update_i_step_condition_dict))
        self._appliance_locs = appliance_locs
        self.update_i_step_condition_dict = update_i_step_condition_dict
        super().__init__(agent, env, name="CoachDialProblem")

def heuristic_planner(prev_state, simulation_table):
    """The heuristic for selecting the agent's action at state t-1
    
    Our heuristic is based on the likelihood of the agent taking an action 
    given the observation (i.e., the user's intent and world state) in the simulation table. 
        - For an observation, we select all the matching rows from the simulation table. 
        - We sample an action based on the probability that each action may appear in the matching rows

    Args:
        prev_state (State): State at t-1
        simulation_table (pandas dataframe): Lookup table for transitions in our simulation

    Returns:
        Action: agent's action selected by the heuristic at t-1
    """
    prev_u = prev_state.user_intent.name
    prev_w = prev_state.world_state
    
    condition_u = simulation_table.u==prev_u
    condition_w = simulation_table.w==prev_w
    condition_both = condition_u & condition_w
    
    # Retrieve all matching rows
    matched_transitions = simulation_table[condition_both]
    total_num_transitions = len(matched_transitions)

    # Calculate the probabilities for wait and bye actions
    wait_action_cnts = (matched_transitions.a == "wait").sum()
    prob_wait_action = wait_action_cnts/total_num_transitions

    bye_action_cnts = (matched_transitions.a == "bye").sum()
    prob_bye_action = bye_action_cnts/total_num_transitions
    
    # sample an action based on the probabilities
    random_var = random.uniform(0,1)
    
    if random_var < prob_wait_action:
        return AgentWaitAction()
    elif random_var < prob_wait_action+prob_bye_action:
        return AgentByeAction()
    else:
        return AgentGiveNextInstructionAction()

def random_planner(prev_state):
    """Random planner that selects the agent's action at state t-1
    
    Args:
        prev_state (State): State at t-1

    Returns:
        Action: agent's action selected by the heuristic at t-1
    """
    # sample an action based on the probabilities
    random_var = random.uniform(0,1)
    
    if random_var < 1/3:
        return AgentWaitAction()
    elif random_var < 2/3:
        return AgentByeAction()
    else:
        return AgentGiveNextInstructionAction()

def test_planner(coachdial, planner, nsteps=3, discount=0.95):
    gamma = 1.0
    total_reward = 0
    total_discounted_reward = 0

    step_id_list = [0]
    total_reward_list = [total_reward]
    total_discounted_reward_list = [total_discounted_reward]

    for i in range(nsteps):
        print("==== Step %d ====" % (i+1))
        if args.print_log:
            logger.info(f"==== Step {i+1} ====")

        # pomdp_py.visual.visualize_pouct_search_tree(coachdial.agent.tree,
        #                                             max_depth=5, anonymize=False)

        true_state = copy.deepcopy(coachdial.env.state)
        
        if IS_HEURISTIC_AGENT:
            action = heuristic_planner(true_state, coachdial.agent.transition_model.simulation_table)
            print("Using heuristic planner")
            if args.print_log: logger.info("Using heuristic planner")

            env_reward = coachdial.env.state_transition(action, execute=True)
            real_observation = coachdial.env.provide_observation(coachdial.agent.observation_model,
                                                              action)
            coachdial.agent.update_history(action, real_observation)
        if IS_RANDOM_AGENT:
            action = random_planner(true_state)
            print("Using random planner")
            if args.print_log: logger.info("Using random planner")

            env_reward = coachdial.env.state_transition(action, execute=True)
            real_observation = coachdial.env.provide_observation(coachdial.agent.observation_model,
                                                              action)
            coachdial.agent.update_history(action, real_observation)
        else:
            action = planner.plan(coachdial.agent)
            print("In test planner agent.tree", coachdial.agent.tree)
            if args.print_log: logger.info(f"In test planner agent.tree {str(coachdial.agent.tree)}")

            env_reward = coachdial.env.state_transition(action, execute=True)
            true_next_state = copy.deepcopy(coachdial.env.state)
            real_observation = coachdial.env.provide_observation(coachdial.agent.observation_model,
                                                              action)

            coachdial.agent.update_history(action, real_observation)
            planner.update(coachdial.agent, action, real_observation)

        total_reward += env_reward
        total_discounted_reward += env_reward * gamma
        gamma *= discount
        
        # Appending intermediate rewards
        step_id_list.append(i+1)
        total_reward_list.append(total_reward)
        total_discounted_reward_list.append(total_discounted_reward)
        
        # Printing states
        print("True state: %s" % true_state)
        print("Action: %s" % str(action))
        print("Observation: %s" % str(real_observation))
        print("Reward: %s" % str(env_reward))
        print("Reward (Cumulative): %s" % str(total_reward))
        print("Reward (Cumulative Discounted): %s" % str(total_discounted_reward))
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)
            print("__plan_time__: %.5f" % planner.last_planning_time)
        if isinstance(planner, pomdp_py.PORollout):
            print("__best_reward__: %d" % planner.last_best_reward)
        print("World:")
        coachdial.print_state()

        if args.print_log:
            logger.info("True prev state: %s" % true_state)
            logger.info("Action: %s" % str(action))
            logger.info("True curr state: %s" % true_next_state)
            logger.info("Observation: %s" % str(real_observation))
            logger.info(f"Belief state: {coachdial.agent.belief}")
            logger.info("Reward: %s" % str(env_reward))
            logger.info("Reward (Cumulative): %s" % str(total_reward))
            logger.info("Reward (Cumulative Discounted): %s" % str(total_discounted_reward))
            if isinstance(planner, pomdp_py.POUCT):
                logger.info("__num_sims__: %d" % planner.last_num_sims)
                logger.info("__plan_time__: %.5f" % planner.last_planning_time)
            if isinstance(planner, pomdp_py.PORollout):
                logger.info("__best_reward__: %d" % planner.last_best_reward)
            logger.info("World:")
            coachdial.log_state()

        if coachdial.env.state.is_terminal:
            break

    return {
            "total_reward": total_reward, 
            "total_discounted_reward": total_discounted_reward, 
            "step_id_list": step_id_list, 
            "total_reward_list": total_reward_list, 
            "total_discounted_reward_list": total_discounted_reward_list
    }

def init_particles_belief(num_particles, init_state, update_i_step_condition_dict, belief="uniform"):
    num_particles = 200
    particles = []
    for _ in range(num_particles): 
        if belief == 'uniform':
            instruction_step = random.randint(1, MAX_INSTRUCTION_STEP)
            user_intent = random.choice([UserAffirmDoingStepIntent(),UserAffirmDoingStepIntent()])
            user_activity = random.choice([UserDoingNothingActivity(), UserPuttingSaucepanOnStoveActivity(), UserAddingOilInPanActivity(), UserTurningOnStoveActivity(), UserCrackingEggInPanActivity(), UserTurningOffStoveActivity() ])
            all_world_states = [i[1]['w'] for i in list(update_i_step_condition_dict.items())]
            world_state = random.choice(all_world_states)
            particles.append(State(instruction_step, user_intent, user_activity, world_state)) # is terminal set to false automatically.
        elif belief == "groundtruth":
            particles.append(init_state)
    init_belief = pomdp_py.Particles(particles)
    return init_belief


def main(args):
    run_id_list = []
    all_step_id_list = [] 
    all_cumu_reward_list = []
    all_cumu_discounted_reward_list = []
    last_cumu_reward_list = []
    last_cumu_discounted_reward_list = []
    nsteps_list = []

    for run_id in range(args.num_runs):
        
        logger.info("="*70)
        logger.info(f"RUN {run_id}")
        logger.info("="*70)

        n, k = 5, 4  # n->side of the grid, k->number of appliances
        init_state, _ = CoachDialProblem.generate_instance(n, k)
        # # For debugging purpose
        # n, k = 2,2
        # user_position = (0, 0)
        # appliance_locs = {}  # map from rock location to rock id
        # appliance_locs[(0,1)] = 0
        # appliance_locs[(1,1)] = 1
        # rocktypes = ('good', 'good')
        # Ground truth state
        # init_state = State(user_position, rocktypes, False)
        # belief = "uniform"

        #update_i_step_condition_dict ={1:{'w':'0000'}, 2:{'w':'0001'}, 3:{'w':'0011'}, 4:{'w':'1011'}, 5:{'w':'1111'}}
        belief = args.belief
        
        init_state_copy = copy.deepcopy(init_state)

        # init belief (uniform), represented in particles;
        # We don't factor the state here; We are also not doing any action prior.
        init_belief = init_particles_belief(200, init_state, update_i_step_condition_dict, belief=belief)
        transition_constant = 0.9
        coachdial = CoachDialProblem(n, k, init_state, init_belief, transition_constant, update_i_step_condition_dict, appliance_locs=None)

        # TODO: print everything in coachdial problem
        coachdial.print_state()

        print("*** Testing POMCP ***")
        if args.print_log: logger.info("*** Testing POMCP ***")
        pomcp = pomdp_py.POMCP(max_depth=6, discount_factor=0.95,
                               num_sims=args.num_sims, exploration_const=10,
                               rollout_policy=coachdial.agent.policy_model,
                               num_visits_init=1)
        outputs = test_planner(coachdial, pomcp, nsteps=args.nsteps, discount=0.95)
        #coachdial.env.state.instruction_step = init_state_copy.instruction_step
        #coachdial.env.state.world_state = init_state_copy.world_state
        #coachdial.env.state.user_intent = init_state_copy.user_intent
        #coachdial.env.state.user_activity = init_state_copy.user_activity
        
        # init_belief = init_particles_belief(200, coachdial.env.state, belief=belief)
        # coachdial.agent.set_belief(init_belief)
        
        logger.info("DONE!")
        run_id_list.extend([run_id]*len(outputs["step_id_list"]))
        all_step_id_list.extend(outputs["step_id_list"])
        all_cumu_reward_list.extend(outputs["total_reward_list"])
        all_cumu_discounted_reward_list.extend(outputs["total_discounted_reward_list"])
        last_cumu_reward_list.append(outputs["total_reward"])
        last_cumu_discounted_reward_list.append(outputs["total_discounted_reward"])
        nsteps_list.append(outputs["step_id_list"][-1] - 1)  # -1 for removing step0

    logger.info("="*70)
    logger.info("="*70)
    logger.info("")
    
    nsteps_list = np.array(nsteps_list)
    last_cumu_reward_list = np.array(last_cumu_reward_list)
    last_cumu_discounted_reward_list = np.array(last_cumu_discounted_reward_list)
    logger.info(f"{args.num_runs:4d} runs") 
    logger.info(f"nsteps_list: {nsteps_list}")
    logger.info(f"cumu_reward_list: {last_cumu_reward_list}")
    logger.info(f"cumu_discounted_reward_list: {last_cumu_discounted_reward_list}")
    logger.info(f"Mean nsteps: {np.mean(nsteps_list):8.6f}")
    logger.info(f"Standard deviation nsteps: {np.std(nsteps_list):8.6f}")
    logger.info(f"Mean cumu_reward: {np.mean(last_cumu_reward_list):8.6f}")
    logger.info(f"Standard deviation cumu_reward: {np.std(last_cumu_reward_list):8.6f}")
    logger.info(f"Mean cumu_discounted_reward: {np.mean(last_cumu_discounted_reward_list):8.6f}")
    logger.info(f"Standard deviation cumu_discounted_reward: {np.std(last_cumu_discounted_reward_list):8.6f}")

    if args.output_results:
        logger.info(f"Output results to {args.output_dir}")
        df = pd.DataFrame(
                {
                    "nsteps": nsteps_list,
                    "cumu_reward": last_cumu_reward_list,
                    "cumu_discounted_reward": last_cumu_discounted_reward_list,
                }
        )
        df.to_csv("{}/overall_stats.csv".format(args.output_dir))

        df = pd.DataFrame(
                {
                    "run_id": run_id_list,
                    "step_id": all_step_id_list,
                    "cumu_reward": all_cumu_reward_list,
                    "cumu_discounted_reward": all_cumu_discounted_reward_list,
                }
        )
        df.to_csv("{}/run_details.csv".format(args.output_dir))
    logger.info("DONE!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)
    
    # Set hyperparams
    global IS_RANDOM_AGENT
    global IS_HEURISTIC_AGENT
    IS_RANDOM_AGENT = True if args.agent_type=="random" else False
    IS_HEURISTIC_AGENT = True if args.agent_type=="heuristic" else False
    
    main(args)
