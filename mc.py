from collections import defaultdict
import numpy as np



def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    Q = getInitQ(env.actionSpaces)


def getInitQ(action_spaces):
    return defaultdict(lambda: np.zeros(action_spaces))


def make_epsilon_greedy_policy(Q, epsilon, nA):

    def policy_fn(state):

        return Q(state)

    return policy_fn
