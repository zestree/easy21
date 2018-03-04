import os
import sys

import plotter
from env.easy21 import Easy21Env
from agent.agent_mc import AgentMC
from agent.agent_sarsa import AgentSarsa
from agent.agent_td import AgentTD
from agent.agent_sarsa_fa import AgentSarsaFa
from estimator import Estimator

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

if __name__ == "__main__":
    env = Easy21Env()
    # agent = AgentMC(env)
    # Q, N_sa = agent.run_episodes(50000)

    # agent_sarsa_fa = AgentSarsa(env)
    # Q, N_sa = agent.run_episodes(50000, 1)
    # plotter.plot(Q)

    estimator = Estimator(env)
    agent = AgentSarsaFa(env, estimator)
    Q = agent.run_episodes(1000, 0.5)

    plotter.plot(Q)

#    features = env.get_features((4,4), 0)
#    print features
