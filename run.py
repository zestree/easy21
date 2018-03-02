import sys, os
from env.easy21 import Easy21Env
import agent as agent
import plotter

_PATH_ = os.path.dirname(os.path.dirname(__file__))


if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

if __name__ == "__main__":
    env = Easy21Env()
#    Q, policy = agent.mc_control_epsilon_greedy(env, 50000)
#    Q, policy = agent.td_control(env, 50000)
    Q, N_sa = agent.sarsa(env, 10000, 0.5)

    plotter.plot(Q)
