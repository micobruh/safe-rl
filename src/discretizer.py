import numpy as np

class Discretizer:
    def __init__(self, features_ranges, bins_sizes, lambda_transform=None):
        assert len(features_ranges) == len(bins_sizes)

        self.num_features = len(features_ranges)
        self.feature_ranges = features_ranges
        self.bins_sizes = bins_sizes

        self.bins = [np.linspace(features_ranges[i][0], features_ranges[i][1], bins_sizes[i] + 1)[1: -1] for i in range(self.num_features)]

        self.lambda_transform = lambda_transform

    def discretize(self, features):
        if self.lambda_transform is None:
            return tuple(np.digitize(x=features[i], bins=self.bins[i]) for i in range(len(features)))
        else:
            features = self.lambda_transform(features)
            return tuple(np.digitize(x=features[i], bins=self.bins[i]) for i in range(len(features)))

    def get_empty_mat(self):
        return np.zeros(self.bins_sizes)

def create_discretizer(envs, env_id, num_bins=40):
    """
    Creates a Discretizer object for a given Safety Gymnasium environment.
    """
    if env_id == 'CartPole-v1':
        state_idxs = [2, 0]
    elif env_id == 'Pendulum-v1':
        state_idxs = [0, 2]    
    else:    
        state_idxs = [0, 1]

    # Get state space limits
    state_low = envs.single_observation_space.low[state_idxs]
    state_high = envs.single_observation_space.high[state_idxs]

    # Ensure valid finite values (replace -inf/inf with reasonable bounds)
    state_low = np.where(state_low == -np.inf, -10, state_low)
    state_high = np.where(state_high == np.inf, 10, state_high)

    # Feature ranges
    features_ranges = list(zip(state_low, state_high))

    # Define bins (same for all features if num_bins is an int)
    if isinstance(num_bins, int):
        bins_sizes = [num_bins] * len(features_ranges)
    else:
        bins_sizes = num_bins  # Custom bin sizes per feature
    transform_fn = lambda s: [s[state_idxs[0]], s[state_idxs[1]]]           

    # Create Discretizer
    discretizer = Discretizer(features_ranges, bins_sizes, lambda_transform=transform_fn)

    return discretizer
