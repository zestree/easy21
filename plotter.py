from collections import defaultdict
from lib import plotting
import numpy as np

def plot(Q):
    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[tuple([state[1], state[0]])] = action_value
    plotting.plot_value_function(V, title="Optimal Value Function")