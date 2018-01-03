import sys, os
from env.easy21 import Easy21Env
import agent as agent
import plotter

_PATH_ = os.path.dirname(os.path.dirname(__file__))


if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

if __name__ == "__main__":
    env = Easy21Env()

    #state = env.getInitState()
    #state, reward = env.step(state, 'stick')
    #print state, reward


    Q, policy = agent.mc_control_epsilon_greedy(env, 1000)

    plotter.plot(Q)
