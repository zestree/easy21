import numpy as np
from agent import Agent
from env.actions import Action

class AgentSarsaFa(Agent):

    def __init__(self, env, estimator):
        Agent.__init__(self, env)
        self.estimator = estimator
        pass

    def run_episodes(self, num_episodes, var_lambda=0, discount_factor=1.0):

        Q = self.get_defaultdict_state_action(self.env.action_spaces)
        epsilon = 0.05
        delta_step_size = 0.01

        for episode_idx in range(1, num_episodes):
            init_state = self.env.getInitState()

            episode = []
            eligibility_trace = self.get_defaultdict_state_action(self.env.action_spaces)
            self.run_sarsa(self.estimator, episode, init_state, epsilon)

            print episode
            for step_id, value in enumerate(episode):

                delta = self.compute_vars(episode, step_id, Q, discount_factor)

                for i in range(0, step_id + 1):
                    state = episode[i][0]
                    action = episode[i][1]
                    eligibility_trace[state][action] = discount_factor * var_lambda * eligibility_trace[state][action]

                    if i == step_id:
                        eligibility_trace[state][action] = 1

                    Q[state][action] += delta_step_size * delta * eligibility_trace[state][action]

        return Q

    def run_sarsa(self, estimator, episode, state, epsilon):

        policy = self.make_policy(estimator, epsilon, self.env.action_spaces)
        selected_action_id = self.choose_action(policy, state)

        new_state, reward, done = self.env.step(state, Action(selected_action_id))
        episode.append((state, selected_action_id, reward))

        if not done:
            self.run_sarsa(estimator, episode, new_state, epsilon)

    @staticmethod
    def make_policy(estimator, epsilon, action_spaces):
        def policy_fn(state):
            q_sa = estimator.predict(state)
            if np.max(q_sa) == np.min(q_sa):
                return np.ones(action_spaces) / action_spaces

            prob = np.ones(action_spaces) * epsilon / action_spaces
            best_action = np.argmax(q_sa)
            prob[best_action] += 1 - epsilon
            return prob
        return policy_fn

    @staticmethod
    def compute_vars(episode, step_id, Q, discount_factor):
        step = episode[step_id]
        state = step[0]
        action = step[1]
        reward = step[2]

        next_Q = 0
        if step_id < len(episode) - 1:
            next_step = episode[step_id + 1]
            next_state = next_step[0]
            next_action = next_step[1]
            next_Q = Q[next_state][next_action]

        delta = reward + discount_factor * next_Q - Q[state][action]

        return delta
