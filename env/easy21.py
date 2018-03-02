import random

from actions import Action

class Easy21Env:
    action_spaces = 2

    def __init__(self):
        return

    def getInitState(self):
        # state = [playPoints, dealerPoints]
        return tuple([self.draw_unsigned_card(), self.draw_unsigned_card()])

    def step(self, state, action):

        if action == Action.HIT:
            state = self.player_action(state)
            reward = self.evaluate_limit(state)
        else:
            state = self.dealer_action(state)
            reward = self.evaluate_terminal_reward(state)

        done = reward is not None
        if not done:
            reward = 0
        return state, reward, done

    def evaluate_limit(self, state):
        playerPoints = state[0]
        dealerPoints = state[1]

        reward = None
        if playerPoints < 1 or 21 < playerPoints:
            reward = -1
        if dealerPoints < 1 or 21 < dealerPoints:
            reward = 1

        return reward

    def evaluate_terminal_reward(self, state):
        reward = self.evaluate_limit(state)
        if reward is not None:
            return reward

        if state[0] == state[1]:
            return 0
        return 1 if state[1] < state[0] else -1


    def draw_card(self):
        sign = -1 if random.randint(1,3) == 1 else 1
        return sign * self.draw_unsigned_card()

    def draw_unsigned_card(self):
        return random.randint(1, 10)

    def dealer_action(self, state):
        while 1 <= state[1] and state[1] < 17:
            state = tuple([state[0], state[1] + self.draw_card()])

        return state

    def player_action(self, state):
        return tuple([state[0] + self.draw_card(), state[1]])
