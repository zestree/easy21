from collections import defaultdict
from env.actions import Action
from agent import Agent


class AgentMC(Agent):

    def run_episodes(self, num_episodes, discount_factor=1.0):

        N_s = defaultdict(float)
        N_sa = self.get_defaultdict_state_action(self.env.action_spaces)
        Q = self.get_defaultdict_state_action(self.env.action_spaces)
        N0 = 100
        policy = None

        for episode_idx in range(1, num_episodes):
            state = self.env.getInitState()
            self.run_mc_step(state, Q, N_s, N_sa, N0, discount_factor)
        return Q, policy

    def run_mc_step(self, state, Q, N_s, N_sa, N0, discount_factor):
        epsilon = N0 / (N0 + N_s[state])
        policy = self.make_epsilon_greedy_policy(Q, epsilon, self.env.action_spaces)

        selected_action_id = self.choose_action(policy, state)
        new_state, reward, done = self.env.step(state, Action(selected_action_id))

        N_s[state] += 1
        N_sa[state][selected_action_id] += 1

        if not done:
            new_state, reward = self.run_mc_step(new_state, Q, N_s, N_sa, N0, discount_factor)

        # incremental update
        error = reward - Q[state][selected_action_id]
        step_size = 1 / N_sa[state][selected_action_id]
        Q[state][selected_action_id] += (step_size * error)

        return new_state, reward * discount_factor
