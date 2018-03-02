from collections import defaultdict
import numpy as np
from env.actions import Action


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


def td_control(env, num_episodes, discount_factor=1.0):

    N_s = defaultdict(float)
    N_sa = get_defaultdict_state_action(env.action_spaces)
    Q = get_defaultdict_state_action(env.action_spaces)
    N0 = 100
    policy = None

    for episode_idx in range(1, num_episodes):
        state = env.getInitState()
        epsilon = N0 / (N0 + N_s[state]) # should update epsilon on every step, now it's only computed by initial state of each episode

        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_spaces)
        action_id = choose_action(policy, state)

        run_td0(policy, state, action_id, env, Q, N_s, N_sa, discount_factor)

    return Q, policy


def run_td0(policy, state, action_id, env, Q, N_s, N_sa, discount_factor):

    new_state, reward, done = env.step(state, Action(action_id))

    N_s[state] += 1
    N_sa[state][action_id] += 1

    step_size = 1 / N_sa[state][action_id]

    if not done:
        new_action_id = choose_action(policy, new_state)
        error = discount_factor * Q[new_state][new_action_id] - Q[state][action_id]
        Q[state][action_id] += (step_size * (reward + error))
        run_td0(policy, new_state, new_action_id, env, Q, N_s, N_sa, discount_factor)
    else:
        error = reward - Q[state][action_id]
        Q[state][action_id] += (step_size * error)


def sarsa(env, num_episodes, var_lambda = 0, discount_factor = 1.0):

    N_s = defaultdict(float)
    N_sa = get_defaultdict_state_action(env.action_spaces)
    Q = get_defaultdict_state_action(env.action_spaces)

    for episode_idx in range(1, num_episodes):
        init_state = env.getInitState()

        episode = []
        eligibility_trace = get_defaultdict_state_action(env.action_spaces)
        run_episode(episode, init_state, env, Q, N_s, 100)

        print episode
        for step_id, value in enumerate(episode):

            step_size, delta = compute_vars(episode, step_id, Q, N_sa, discount_factor)

            for i in range(0, step_id + 1):
                state = episode[i][0]
                action = episode[i][1]
                eligibility_trace[state][action] = discount_factor * var_lambda * eligibility_trace[state][action]

                if i == step_id:
                    eligibility_trace[state][action] = 1

                Q[state][action] += step_size * delta * eligibility_trace[state][action]

    return Q, N_sa


def compute_vars(episode, step_id, Q, N_sa, discount_factor):
    step = episode[step_id]
    state = step[0]
    action = step[1]
    reward = step[2]
    N_sa[state][action] += 1
    step_size = 1 / N_sa[state][action]

    next_Q = 0
    if step_id < len(episode) - 1:
        next_step = episode[step_id + 1]
        next_state = next_step[0]
        next_action = next_step[1]
        next_Q = Q[next_state][next_action]

    delta = reward + discount_factor * next_Q - Q[state][action]

    return step_size, delta


def run_episode(episode, state, env, Q, N_s, N0 = 100):
    N_s[state] += 1
    epsilon = N0 / (N0 + N_s[state])
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_spaces)

    selected_action_id = choose_action(policy, state)
    new_state, reward, done = env.step(state, Action(selected_action_id))

    episode.append((state, selected_action_id, reward))

    if not done:
        run_episode(episode, new_state, env, Q, N_s, N0)
    return reward


def choose_action(policy, state):
    prob = policy(state)
    return np.random.choice(np.arange(len(prob)), p=prob)


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(state):
        if np.max(Q[state]) == np.min(Q[state]):
            return np.ones(nA, dtype=float) / nA

        best_action = np.argmax(Q[state])
        prob = np.ones(nA, dtype=float) * epsilon / nA
        prob[best_action] += (1 - epsilon)
        return prob
    return policy_fn


def get_defaultdict_state_action(action_spaces):
    return defaultdict(lambda: np.zeros(action_spaces))
