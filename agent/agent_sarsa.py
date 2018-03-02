from collections import defaultdict
from env.actions import Action
from agent import Agent


class AgentSarsa(Agent):
    def run_episodes(self, num_episodes, var_lambda=0, discount_factor=1.0):

        N_s = defaultdict(float)
        N_sa = self.get_defaultdict_state_action(self.env.action_spaces)
        Q = self.get_defaultdict_state_action(self.env.action_spaces)

        for episode_idx in range(1, num_episodes):
            init_state = self.env.getInitState()

            episode = []
            eligibility_trace = self.get_defaultdict_state_action(self.env.action_spaces)
            self.run_sarsa(episode, init_state, Q, N_s, 100)

            print episode
            for step_id, value in enumerate(episode):

                delta_step_size, delta = self.compute_vars(episode, step_id, Q, N_sa, discount_factor)

                for i in range(0, step_id + 1):
                    state = episode[i][0]
                    action = episode[i][1]
                    eligibility_trace[state][action] = discount_factor * var_lambda * eligibility_trace[state][action]

                    if i == step_id:
                        eligibility_trace[state][action] = 1

                    Q[state][action] += delta_step_size * delta * eligibility_trace[state][action]

        return Q, N_sa

    def run_sarsa(self, episode, state, Q, N_s, N0=100):
        N_s[state] += 1
        epsilon = N0 / (N0 + N_s[state])
        policy = self.make_epsilon_greedy_policy(Q, epsilon, self.env.action_spaces)

        selected_action_id = self.choose_action(policy, state)
        new_state, reward, done = self.env.step(state, Action(selected_action_id))

        episode.append((state, selected_action_id, reward))

        if not done:
            self.run_sarsa(episode, new_state, Q, N_s, N0)
        return reward

    @staticmethod
    def compute_vars(episode, step_id, Q, N_sa, discount_factor):
        step = episode[step_id]
        state = step[0]
        action = step[1]
        reward = step[2]
        N_sa[state][action] += 1
        delta_step_size = 1 / N_sa[state][action]

        next_Q = 0
        if step_id < len(episode) - 1:
            next_step = episode[step_id + 1]
            next_state = next_step[0]
            next_action = next_step[1]
            next_Q = Q[next_state][next_action]

        delta = reward + discount_factor * next_Q - Q[state][action]

        return delta_step_size, delta
