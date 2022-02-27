import pomdp_py
import copy
from HTNCoachDial import *
from human_simulator import *

class HTNCoachDialEnvironment(pomdp_py.Environment):

    def __init__(self, human_simulator, explaset, init_state, RewardModel, TransitionModel):
        self.human_simulator = human_simulator
        # self.explaset = explaset
        super().__init__(init_state,
                         TransitionModel,
                         RewardModel)

    # def set_htn_explaset(self,state, htn_explaset):
    #     state.set_htn_explaset(htn_explaset)
    #     self.explaset = explaset

    # def set_htn_explaset(self,state, htn_explaset):
    #     state.set_htn_explaset(htn_explaset)
    #     self.explaset = explaset

    # def set_true_state(self, state):
    #     self.state = state

    def state_transition(self, action, execute=True):
        """state_transition(self, action, execute=True, **kwargs)

        Overriding parent class function.
        Simulates a state transition given `action`. If `execute` is set to True,
        then the resulting state will be the new current state of the environment.

        Args:
            action (Action): action that triggers the state transition
            execute (bool): If True, the resulting state of the transition will
                            become the current state.

        Returns:
            float or tuple: reward as a result of `action` and state
            transition, if `execute` is True (next_state, reward) if `execute`
            is False.

        """
        # assert robot_id is not None, "state transition should happen for a specific robot"
        state = copy.deepcopy(self.state)
        # next_state = self.transition_model.sample(self.state, action, execute)
        next_state = self.partial_state_transition(self.state, action, execute)
        reward = self.reward_model.sample(state, action, next_state)
        # nsteps += 1

        # next_state = copy.deepcopy(self.state)
        # next_state.object_states[robot_id] =\
        #     self.transition_model[robot_id].sample(self.state, action)
        
        # reward = self.reward_model.sample(self.state, action, next_state,
        #                                   robot_id=robot_id)
        if execute:
            self.apply_transition(next_state)
            return reward
        else:
            return next_state, reward

    def partial_state_transition(self,state, action, execute):
        '''set question asked, explaset title and step_index'''
        state = copy.deepcopy(state)
        if self.human_simulator.check_terminal_state(state.step_index+1): ##check if next step index is out of length 
            return state
        if action.name == "ask-clarification-question":
            question_asked = action.update_question_asked(state)

        else:
            question_asked = None

        # set
        # set explaset highest
        # state.htn_explaset.highest_action_PS = question_asked_tuple

        question_title = config.question_title
        question_title_split = question_title.split("-")
        question_asked_state = state.get_object_state(question_title)
        # question_asked_state.attributes[title_split[1]] = question_asked
        question_asked_state.__setitem__(question_title_split[1], question_asked)

        
        
        explaset_title = config.explaset_title
        explaset_title_split = explaset_title.split("-")
        sensor_state = state.get_object_state(explaset_title)
        # step, sensor_notification = self.human_simulator.curr_step(sensor_state.attributes[explaset_title_split[1]][-1], action.name)
        '''just need the step_index and step name to fill not update the db.'''
        state.step_index+=1
        step =  self.human_simulator.return_step(state.step_index)
        
        # if execute:
        #     step_index, step, sensor_notification = self.human_simulator.curr_step(state.step_index, action.name, real_step=True)
        # else:
        #     step_index, step, sensor_notification = self.human_simulator.curr_step(state.step_index, action.name)
        state.append_object_attribute(explaset_title, explaset_title_split[1], step)
        # print("state after append is", state)
        # state.step_index = step_index
        
        return state #htn_explaset is not set

