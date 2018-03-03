import random
import numpy as np

from actions import Action

class Easy21Env:
    action_spaces = 2

    def __init__(self):
        return

    def getInitState(self):
        # state = [playPoints, dealerPoints]
        return tuple([self.draw_unsigned_card(), self.draw_unsigned_card()])

    def get_features(self, state, action):
        features = np.zeros(36)
        dealer_feature_definition = np.array([[1, 4], [4, 7], [7, 10]])
        player_feature_definition = np.array([[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]])
        player_points = state[0]
        dealer_points = state[1]

        map_player = np.vectorize(lambda v: 1 if v[0] <= player_points <= v[1] else 0, signature='(m)->()')
        map_dealer = np.vectorize(lambda v: 1 if v[0] <= dealer_points <= v[1] else 0, signature='(m)->()')

        player_features = map_player(player_feature_definition)
        dealer_features = map_dealer(dealer_feature_definition)

        player_idx = [index for index,value in enumerate(player_features) if value == 1]
        dealer_idx = [index for index,value in enumerate(dealer_features) if value == 1]

        all_idx = []
        for idx in dealer_idx:
            offset = idx * 6
            for pi in player_idx:
                all_idx.append(pi + offset)

        all_idx = np.array(all_idx)
        all_idx += (action * 18)

        for i in all_idx:
            features[i] = 1
        return features

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
        while 1 <= state[1] < 17:
            state = tuple([state[0], state[1] + self.draw_card()])

        return state

    def player_action(self, state):
        return tuple([state[0] + self.draw_card(), state[1]])
