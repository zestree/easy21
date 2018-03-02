from collections import defaultdict
import numpy as np

class Agent:

    def __init__(self, env):
        self.env = env
        return

    @staticmethod
    def choose_action(policy, state):
        prob = policy(state)
        return np.random.choice(np.arange(len(prob)), p=prob)

    @staticmethod
    def make_epsilon_greedy_policy(Q, epsilon, nA):
        def policy_fn(state):
            if np.max(Q[state]) == np.min(Q[state]):
                return np.ones(nA, dtype=float) / nA

            best_action = np.argmax(Q[state])
            prob = np.ones(nA, dtype=float) * epsilon / nA
            prob[best_action] += (1 - epsilon)
            return prob

        return policy_fn

    @staticmethod
    def get_defaultdict_state_action(action_spaces):
        return defaultdict(lambda: np.zeros(action_spaces))