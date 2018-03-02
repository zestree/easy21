from collections import defaultdict
import numpy as np
from env.actions import Action
from agent import Agent


class AgentTD(Agent):

    def run_episodes(self, num_episodes, discount_factor=1.0):

        N_s = defaultdict(float)
        N_sa = self.get_defaultdict_state_action(self.env.action_spaces)
        Q = self.get_defaultdict_state_action(self.env.action_spaces)
        N0 = 100
        policy = None

        for episode_idx in range(1, num_episodes):
            state = self.env.getInitState()
            action_id = self.choose_action(policy, state)

            self.run_td0(state, action_id, self.env, Q, N_s, N_sa, N0, discount_factor)
        return Q, policy

    def run_td0(self, state, action_id, env, Q, N_s, N_sa, N0, discount_factor):

        epsilon = N0 / (N0 + N_s[state])
        policy = self.make_epsilon_greedy_policy(Q, epsilon, env.action_spaces)
        new_state, reward, done = env.step(state, Action(action_id))

        N_s[state] += 1
        N_sa[state][action_id] += 1

        step_size = 1 / N_sa[state][action_id]

        if not done:
            new_action_id = self.choose_action(policy, new_state)
            error = discount_factor * Q[new_state][new_action_id] - Q[state][action_id]
            Q[state][action_id] += (step_size * (reward + error))
            self.run_td0(new_state, new_action_id, env, Q, N_s, N_sa, discount_factor)
        else:
            error = reward - Q[state][action_id]
            Q[state][action_id] += (step_size * error)
