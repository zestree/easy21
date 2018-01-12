from collections import defaultdict
import numpy as np
from env.actions import Action


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):

    N_s = defaultdict(float)
    N_sa = get_defaultdict_state_action(env.action_spaces)
    Q = get_defaultdict_state_action(env.action_spaces)
    N0 = 100
    policy = None

    for episode in range(1, num_episodes):
        state = env.getInitState()
        epsilon = N0 / (N0 + N_s[state])

        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_spaces)

        run_step(policy, state, env, Q, N_s, N_sa, discount_factor)

    return Q, policy


def run_step(policy, state, env, Q, N_s, N_sa, discount_factor):
    prob = policy(state)

    selected_action_id = np.random.choice(np.arange(len(prob)), p=prob)
    new_state, reward, done = env.step(state, Action(selected_action_id))

    N_s[state] += 1
    N_sa[state][selected_action_id] += 1

    if not done:
        new_state, reward = run_step(policy, new_state, env, Q, N_s, N_sa, discount_factor)

    # incremental update
    error = reward - Q[state][selected_action_id]
    step_size = 1 / N_sa[state][selected_action_id]
    Q[state][selected_action_id] += (step_size * error)

    return new_state, reward * discount_factor


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(state):
        best_action = np.argmax(Q[state])
        prob = np.ones(nA, dtype=float) * epsilon / nA
        prob[best_action] += (1 - epsilon)
        return prob
    return policy_fn


def get_defaultdict_state_action(action_spaces):
    return defaultdict(lambda: np.zeros(action_spaces))
