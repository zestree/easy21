import numpy as np


class Estimator:
    dealer_feature_definition = np.array([[1, 4], [4, 7], [7, 10]])
    player_feature_definition = np.array([[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]])

    def __init__(self, env):
        self.env = env
        self.weight = np.zeros(36)
        pass

    def featurize_state(self, state):
        player_points = state[0]
        dealer_points = state[1]

        map_player = np.vectorize(lambda v: 1 if v[0] <= player_points <= v[1] else 0, signature='(m)->()')
        map_dealer = np.vectorize(lambda v: 1 if v[0] <= dealer_points <= v[1] else 0, signature='(m)->()')

        player_features = map_player(self.player_feature_definition)
        dealer_features = map_dealer(self.dealer_feature_definition)

        player_idx = [index for index, value in enumerate(player_features) if value == 1]
        dealer_idx = [index for index, value in enumerate(dealer_features) if value == 1]

        all_idx = []
        for idx in dealer_idx:
            offset = idx * 6
            for pi in player_idx:
                all_idx.append(pi + offset)

        features = []
        for action in range(self.env.action_spaces):
            features.append(np.zeros(36))
            feature_idx = np.array(all_idx) + (action * 18)
            for i in feature_idx:
                features[action][i] = 1
        return features

    def featurize_sa(self, state, action):
        return self.featurize_state(state)[action]

    def predict(self, state):
        features = self.featurize_state(state)
        return np.dot(features, self.weight)
