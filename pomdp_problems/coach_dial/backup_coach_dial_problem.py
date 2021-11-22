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
import pomdp_py
import random
import math
import numpy as np
import pandas as pd
import sys
import copy

EPSILON = 1e-9
MAX_INSTRUCTION_STEP = 5
IS_RANDOM_AGENT = False
IS_HEURISTIC_AGENT = False

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

    def __hash__(self):
        return hash((self.instruction_step, self.user_intent, self.user_activity, self.world_state, self.is_terminal))
    def __eq__(self, other):
        if isinstance(other, State):
            return self.instruction_step == other.instruction_step\
                and self.user_intent == other.user_intent\
                and self.user_activity == other.user_activity\
                and self.world_state == other.world_state\
                and self.is_terminal == other.is_terminal
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
        prev_is_terminal = prev_state.is_terminal

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

            if (next_state.instruction_step > MAX_INSTRUCTION_STEP and
                prev_action.name == "bye"):
                prev_is_terminal = True
                next_state.is_terminal = True
            
            return next_state #@II returned without isterminal being set
        
    def argmax(self, state, action):
        """Returns the most like next state"""
        return np.argmax(self.sample(state, action))


class Observation(pomdp_py.Observation):
    def __init__(self, user_intent, world_state):
        self.user_intent = user_intent
        self.world_state = world_state
    def __hash__(self):
        return hash((self.user_intent, self.world_state))
    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.user_intent == other.user_intent and self.world_state == other.world_state
        else:
            return False
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return "Observation(%s | %s)" % (str(self.user_intent), str(self.world_state))

class CDObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0.15):
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
            return Observation(curr_state.user_intent, curr_state.world_state)
        else:
            return Observation(curr_state.other_user_intent(), curr_state.world_state)

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
        if state.is_terminal:
            return 0  # terminated. No reward
        if IS_RANDOM_AGENT:
            return -1 # Random agent. Always return -1.

        if isinstance(action, AgentGiveNextInstructionAction):
            if (state.instruction_step != 6 and 
            update_i_step_condition_dict[state.instruction_step]['w'] == state.world_state and 
            state.user_activity == "doing-nothing"):
                return 20
            else:
                return -10
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

def test_planner(coachdial, planner, nsteps=3, discount=0.95):
    gamma = 1.0
    total_reward = 0
    total_discounted_reward = 0
    for i in range(nsteps):
        print("==== Step %d ====" % (i+1))

        # pomdp_py.visual.visualize_pouct_search_tree(coachdial.agent.tree,
        #                                             max_depth=5, anonymize=False)

        true_state = copy.deepcopy(coachdial.env.state)
        
        if IS_HEURISTIC_AGENT:
            action = heuristic_planner(true_state, coachdial.agent.transition_model.simulation_table)
            print("Using heuristic planner")

            env_reward = coachdial.env.state_transition(action, execute=True)
            real_observation = coachdial.env.provide_observation(coachdial.agent.observation_model,
                                                              action)
            coachdial.agent.update_history(action, real_observation)
        else:
            action = planner.plan(coachdial.agent)
            print("In test planner agent.tree", coachdial.agent.tree)

            env_reward = coachdial.env.state_transition(action, execute=True)
            true_next_state = copy.deepcopy(coachdial.env.state)
            real_observation = coachdial.env.provide_observation(coachdial.agent.observation_model,
                                                              action)

            coachdial.agent.update_history(action, real_observation)                                                  
            planner.update(coachdial.agent, action, real_observation)
        
        total_reward += env_reward
        total_discounted_reward += env_reward * gamma
        gamma *= discount
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

        if coachdial.env.state.is_terminal:
            break
    return total_reward, total_discounted_reward


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


if __name__ == '__main__':

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

    update_i_step_condition_dict ={1:{'w':'0000'}, 2:{'w':'0001'}, 3:{'w':'0011'}, 4:{'w':'1011'}, 5:{'w':'1111'}}
    belief = "groundtruth"
    
    init_state_copy = copy.deepcopy(init_state)

    # init belief (uniform), represented in particles;
    # We don't factor the state here; We are also not doing any action prior.
    init_belief = init_particles_belief(200, init_state, update_i_step_condition_dict, belief=belief)
    transition_constant = 0.9
    coachdial = CoachDialProblem(n, k, init_state, init_belief, transition_constant, update_i_step_condition_dict, appliance_locs=None)

    # TODO: print everything in coachdial problem
    coachdial.print_state()

    print("*** Testing POMCP ***")
    pomcp = pomdp_py.POMCP(max_depth=6, discount_factor=0.95,
                           num_sims=500, exploration_const=20,
                           rollout_policy=coachdial.agent.policy_model,
                           num_visits_init=1)
    tt, ttd = test_planner(coachdial, pomcp, nsteps=100, discount=0.95)

    coachdial.env.state.instruction_step = init_state_copy.instruction_step
    coachdial.env.state.world_state = init_state_copy.world_state
    coachdial.env.state.user_intent = init_state_copy.user_intent
    coachdial.env.state.user_activity = init_state_copy.user_activity
    
    # init_belief = init_particles_belief(200, coachdial.env.state, belief=belief)
    # coachdial.agent.set_belief(init_belief)
