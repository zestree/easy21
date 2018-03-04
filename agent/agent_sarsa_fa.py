import numpy as np
from agent import Agent
from env.actions import Action

class AgentSarsaFa(Agent):

    def __init__(self, env, estimator):
        Agent.__init__(self, env)
        self.estimator = estimator
        pass

    def run_episodes(self, num_episodes, var_lambda=0, discount_factor=1.0):

        epsilon = 0.05
        delta_step_size = 0.01

        for episode_idx in range(1, num_episodes):
            init_state = self.env.getInitState()
            estimator = self.estimator

            episode = []
            eligibility_trace = np.zeros(36)
            policy = self.make_policy(estimator, epsilon, self.env.action_spaces)
            self.run_sarsa(estimator, episode, policy, init_state, epsilon)

            print episode
            for step_id, value in enumerate(episode):

                delta = self.compute_delta(estimator, episode, step_id, discount_factor)

                for i in range(0, step_id + 1):
                    state = episode[i][0]
                    action = episode[i][1]

                    eligibility_trace = discount_factor * var_lambda * eligibility_trace

                    if i == step_id:
                        x = estimator.featurize_sa(state, action)
                        eligibility_trace += x

                dw = delta_step_size * delta * eligibility_trace
                estimator.update_weight(dw)
        return self.compute_Q()

    def run_sarsa(self, estimator, episode, policy, state, epsilon):

        selected_action_id = self.choose_action(policy, state)

        new_state, reward, done = self.env.step(state, Action(selected_action_id))
        episode.append((state, selected_action_id, reward))

        if not done:
            self.run_sarsa(estimator, episode, policy, new_state, epsilon)

        return

    def make_policy(self, estimator, epsilon, action_spaces):
        def policy_fn(state):
            q_sa = estimator.predict(state)
            if np.max(q_sa) == np.min(q_sa):
                return np.ones(action_spaces) / action_spaces

            prob = np.ones(action_spaces) * epsilon / action_spaces
            best_action = np.argmax(q_sa)
            prob[best_action] += 1 - epsilon
            return prob
        return policy_fn

    def compute_delta(self, estimator, episode, step_id, discount_factor):
        step = episode[step_id]
        state = step[0]
        action = step[1]
        reward = step[2]

        next_q_sa = 0
        if step_id < len(episode) - 1:
            next_step = episode[step_id + 1]
            next_state = next_step[0]
            next_action = next_step[1]
            next_q_sa = estimator.predict_sa(next_state, next_action)

        delta = reward + discount_factor * next_q_sa - estimator.predict_sa(state, action)
        return delta

    def compute_Q(self):
        Q = self.get_defaultdict_state_action(self.env.action_spaces)

        for i in range(1,21):
            for j in range(1,10):
                Q[(i,j)] = self.estimator.predict((i,j))
        return Q

