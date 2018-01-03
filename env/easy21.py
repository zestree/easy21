import random
import numpy as np
import copy
from actions import Action

class Easy21Env:
    action_spaces = 2

    def __init__(self):
        return

    def getInitState(self):
        ''' state = [dealerPoints, playPoints] '''
        return np.array([self.drawUnsignedCard(), self.drawUnsignedCard()])

    def step(self, state, action):
        state = copy.copy(state)
        reward = None
        if action == Action.HIT:
            self.playerAction(state)
            reward = self.evaluateLimit(state)
        else:
            self.dealerAction(state)
            reward = self.evaluateTerminalReward(state)

        return state, reward

    def evaluateLimit(self, state):
        dealerPoints = state[0]
        playerPoints = state[1]

        reward = None
        if playerPoints < 1 or 21 < playerPoints:
            reward = -1
        if dealerPoints < 1 or 21 < dealerPoints:
            reward = 1

        return reward

    def evaluateTerminalReward(self, state):
        reward = self.evaluateLimit(state)
        if reward != None:
            return reward

        if state[0] == state[1]:
            return 0
        return 1 if state[0] < state[1] else -1


    def drawCard(self):
        sign = -1 if random.randint(1,3) == 1 else 1
        return sign * self.drawUnsignedCard()

    def drawUnsignedCard(self):
        return random.randint(1, 10)

    def dealerAction(self, state):

        while 1 <= state[0] and state[0] < 17:
            state[0] += self.drawCard()

    def playerAction(self, state):
        state[1] += self.drawCard()
