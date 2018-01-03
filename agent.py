from collections import defaultdict
import numpy as np
from env.actions import Action


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
#    W = defaultdict(float)
    N_s = defaultdict(float)
    W = get_defaultdict_stateaction(env.action_spaces)
    N_sa = get_defaultdict_stateaction(env.action_spaces)
    Q = get_defaultdict_stateaction(env.action_spaces)
    
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_spaces)

    for episode in range(1, num_episodes):
        state = env.getInitState()
        run_episode(policy, state, env, Q, N_s, N_sa, W, discount_factor)

    return Q, policy


def run_episode(policy, state, env, Q, N_s, N_sa, W, discount_factor):
    actions = policy(state)
    selected_action_id = np.argmax(actions)
    new_state, reward = env.step(state, Action(selected_action_id))

    state_key = tuple(state)
    N_s[state_key] += 1
    N_sa[state_key][selected_action_id] += 1

    if reward is None:
        new_state, reward = run_episode(policy, new_state, env, Q, N_s, N_sa, W, discount_factor)

    W[state_key][selected_action_id] += reward
    Q[state_key][selected_action_id] = W[state_key][selected_action_id] / N_sa[state_key][selected_action_id]
    return new_state, reward * discount_factor


def get_defaultdict_stateaction(action_spaces):
    return defaultdict(lambda: np.zeros(action_spaces))


def make_epsilon_greedy_policy(Q, epsilon, nA):

    def policy_fn(state):

        return np.array([np.random.rand(), np.random.rand()])

    return policy_fn
