import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import safety_gymnasium
from tqdm import tqdm, trange
import hashlib
from sklearn.neighbors import NearestNeighbors
from src.policy import PolicyNetwork, train_supervised
from src.q_val import QNetwork

def compute_v_value(policy, q_network, state, num_samples = 100):
    """
    Estimate V(s) by averaging Q(s, a) over actions sampled from the policy.
    """
    action_samples = [policy(state) for _ in range(num_samples)]
    q_values = torch.stack([q_network(state, a) for a in action_samples])
    return q_values.mean(dim = 0)

def compute_advantage(q_network, policy, state, action):
    """
    Compute A(s, a), i.e. difference between Q(s, a) and V(s)
    """
    q_value = q_network(state, action)
    v_value = compute_v_value(policy, q_network, state)
    return q_value - v_value

def collect_particles(env, policy, episode_nr, step_nr, state_dim, action_dim):
    """
    Collects samples by running policy in the env.
    """
    # State's dimension on step nr is one more to account for next state
    states = np.zeros((episode_nr, step_nr + 1, state_dim), dtype = np.float32)
    actions = np.zeros((episode_nr, step_nr, action_dim), dtype = np.float32)
    costs = np.zeros((episode_nr, step_nr, action_dim), dtype = np.float32)

    # Storing information in "D"
    for episode in range(episode_nr):
        s = env.reset()
        for t in range(step_nr):
            states[episode, t] = s
            a = policy.predict(s).numpy()
            actions[episode, t] = a
            ns, _, c, _, _, _ = env.step(a)
            costs[episode, t] = c
            s = ns
        # Indicating the final (next) state
        states[episode, t + 1] = s

    next_states = None
    for n_traj in range(episode_nr):
        traj_real_len = real_traj_lengths[n_traj].item()
        traj_next_states = states[n_traj, 1: traj_real_len + 1, : ].reshape(-1, env.num_features)
        if next_states is None:
            next_states = traj_next_states
        else:
            next_states = np.concatenate([next_states, traj_next_states], axis=0)

    return states, actions, costs, next_states


def compute_importance_weights(behavioral_policy, target_policy, states, actions,
                               num_traj, real_traj_lengths):
    # Initialize to None for the first concat
    importance_weights = None

    # Compute the importance weights
    # build iw vector incrementally from trajectory particles
    for n_traj in range(num_traj):
        traj_length = real_traj_lengths[n_traj][0].item()

        traj_states = states[n_traj, :traj_length]
        traj_actions = actions[n_traj, :traj_length]

        traj_target_log_p = target_policy.get_log_p(traj_states, traj_actions)
        traj_behavior_log_p = behavioral_policy.get_log_p(traj_states, traj_actions)

        traj_particle_iw = torch.exp(torch.cumsum(traj_target_log_p - traj_behavior_log_p, dim=0))

        if importance_weights is None:
            importance_weights = traj_particle_iw
        else:
            importance_weights = torch.cat([importance_weights, traj_particle_iw], dim=0)

    # Normalize the weights
    importance_weights /= torch.sum(importance_weights)
    return importance_weights


def compute_entropy(behavioral_policy, target_policy, states, actions,
                    num_traj, real_traj_lengths, distances, indices, k, G, B, ns, eps):
    importance_weights = compute_importance_weights(behavioral_policy, target_policy, states, actions,
                                                    num_traj, real_traj_lengths)
    # Compute objective function
    # compute weights sum for each particle
    weights_sum = torch.sum(importance_weights[indices[:, :-1]], dim=1)
    # compute volume for each particle
    volumes = (torch.pow(distances[:, k], ns) * torch.pow(torch.tensor(np.pi), ns/2)) / G
    # compute entropy
    entropy = - torch.sum((weights_sum / k) * torch.log((weights_sum / (volumes + eps)) + eps)) + B

    return entropy


def compute_kl(behavioral_policy, target_policy, states, actions,
               num_traj, real_traj_lengths, distances, indices, k, eps):
    importance_weights = compute_importance_weights(behavioral_policy, target_policy, states, actions,
                                                    num_traj, real_traj_lengths)

    weights_sum = torch.sum(importance_weights[indices[:, :-1]], dim=1)

    # Compute KL divergence between behavioral and target policy
    N = importance_weights.shape[0]
    kl = (1 / N) * torch.sum(torch.log(k / (N * weights_sum) + eps))

    numeric_error = torch.isinf(kl) or torch.isnan(kl)

    # Minimum KL is zero
    # NOTE: do not remove epsilon factor
    kl = torch.max(torch.tensor(0.0), kl)

    return kl, numeric_error


def collect_particles_and_compute_knn(env, behavioral_policy, num_traj, traj_len,
                                      state_filter, k, num_workers):
    assert num_traj % num_workers == 0, "Please provide a number of trajectories " \
                                        "that can be equally split among workers"

    # Collect particles using behavioral policy
    res = Parallel(n_jobs=num_workers)(
        delayed(collect_particles)(env, behavioral_policy, int(num_traj/num_workers), traj_len, state_filter)
        for _ in range(num_workers)
    )
    states, actions, real_traj_lengths, next_states = [np.vstack(x) for x in zip(*res)]

    # Fit knn for the batch of collected particles
    nbrs = NearestNeighbors(n_neighbors = k + 1, metric = 'euclidean', algorithm = 'auto', n_jobs = num_workers)
    nbrs.fit(next_states)
    distances, indices = nbrs.kneighbors(next_states)

    # Return tensors so that downstream computations can be moved to any target device (#todo)
    states = torch.tensor(states, dtype = torch.float64)
    actions = torch.tensor(actions, dtype = torch.float64)
    next_states = torch.tensor(next_states, dtype = torch.float64)
    real_traj_lengths = torch.tensor(real_traj_lengths, dtype = torch.int64)
    distances = torch.tensor(distances, dtype = torch.float64)
    indices = torch.tensor(indices, dtype = torch.int64)

    return states, actions, real_traj_lengths, next_states, distances, indices

def hash_numpy_array(arr):
    """Generate a fixed-length hash that preserves uniqueness."""
    return hashlib.md5(arr.tobytes()).hexdigest()  # Hashes content, including shape

def update_D(current_state, action, cost, next_state, D, state_dct, action_dct):
    # Convert all numpy arrays into hashable keys
    current_state_key = hash_numpy_array(current_state)
    action_key = hash_numpy_array(action)
    next_state_key = hash_numpy_array(next_state)
    # Put them into a tuple to let it hashable
    key = (current_state_key, action_key, cost, next_state_key)
    # Assign the corresponding value to be cost
    D[key] = cost
    # Only add key to dictionary if it doesn't exist before
    if current_state_key not in state_dct:
        state_dct[current_state_key] = current_state
    if next_state_key not in state_dct:
        state_dct[next_state_key] = next_state
    if action_key not in action_dct:
        action_dct[action_key] = action
    return D, state_dct, action_dct                

def update(state: tuple[int, int], reward: float, action: int):
    """Any code that processes a reward given the state and updates the agent.

    Args:
        state: The updated position of the agent.
        reward: The value which is returned by the environment as a
            reward.
        action: The action which was taken by the agent.
    """
    # Q-Learning update
    Q = np.zeros((grid.shape[0] * grid.shape[1], directions))
    current_state = state_to_nr(state)
    Q[old_state, action] += alpha * (reward + gamma * 
                                                    np.max(Q[current_state, : ]) - 
                                                    Q[old_state, action])  

def CEM(env_id, epoch_nr = 500, step_nr = 1000, reward_free = False, 
         delta = 0.1, lambda_omega = 1e-2, lambda_policy = 1e-5, 
         k = 4, d = 25, gamma = 0.99, episode_nr = 20):
    """
    T: Number of trajectories
    N: Number of time steps
    delta: Trust-region threshold (Maximum KL Divergence between two avg state density distributions)
    omega (see below): Safety weight/Lagrange multiplier (0 or larger)
    lambda: Learning rate
    k: Number of neighbors
    d: Safety threshold
    gamma: Discount factor of cost over time
    """
    D = {}
    omega = 0.1
    state_dct = {}
    action_dct = {}

    env = safety_gymnasium.make(env_id, render_mode = 'human', max_episode_steps = step_nr)
    # To ensure mujoco is showing the same number of steps as reality
    env.task.sim_conf.frameskip_binom_n = 1

    # Initialize policy neural network (theta)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    theta_nn = PolicyNetwork(state_dim, action_dim)
    theta_optimizer = torch.optim.Adam(theta_nn.parameters(), lr = lambda_policy)
    theta_nn = train_supervised(env, theta_nn, theta_optimizer)
    
    # Create a copy of theta (theta')
    theta_prime_nn = PolicyNetwork(state_dim, action_dim)
    theta_prime_nn.load_state_dict(theta_nn.state_dict()) 

    # Initialize Q value neural network
    q_nn = QNetwork(state_dim, action_dim)
    q_optimizer = torch.optim.Adam(q_nn.parameters(), lr = lambda_policy)
    
    for epoch in trange(epoch_nr):
        # Initialize the episode cost every time at the beginning of the episode
        ep_ret, ep_cost = 0, 0
        
        for _ in trange(episode_nr):
            state, info = env.reset()
            # Set seeds
            # obs, _ = env.reset(seed=0)        
            terminated, truncated = False, False
            assert env.observation_space.contains(state)
            for _ in range(step_nr):         
                # TODO: Sample using neural network instead
                action = theta_nn(state)
                # action = env.action_space.sample()
                assert env.action_space.contains(action)
                new_state, reward, cost, terminated, truncated, info = env.step(action)
                D, state_dct, action_dct = update_D(state, action, cost, new_state, D, state_dct, action_dct)
                ep_cost += cost
                # Only add the reward during deployment i.e. not reward-free
                if not reward_free:
                    ep_ret += reward
                    if terminated or truncated:
                        break
                state = new_state

        # Note that ep_cost is originally negative, but cost should be positive
        total_episode_cost = -ep_cost / episode_nr - d
        # Update omega
        omega = max(0, omega + lambda_omega * total_episode_cost)

        iter = 0
        while iter < 30:
            iter += 1
        
        # Empty the dynamics before the next epoch starts
        D = {}
        # Update theta' using the newest values of theta
        theta_prime_nn.load_state_dict(theta_nn.state_dict())

    env.close()
    return ep_ret, ep_cost    