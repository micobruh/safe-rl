import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import safety_gymnasium
from tqdm import tqdm, trange
import time
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
import scipy
import scipy.special
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
    next_states = np.zeros((episode_nr, step_nr, state_dim), dtype = np.float32)

    # Storing information in "D"
    for episode in range(episode_nr):
        s = env.reset()
        for t in range(step_nr):
            states[episode, t] = s
            a = policy.predict(s).numpy()
            actions[episode, t] = a
            ns, _, c, _, _, _ = env.step(a)
            costs[episode, t] = c
            next_states[episode, t] = ns
            s = ns
        # Indicating the final (next) state
        states[episode, t + 1] = s

    return states, actions, costs, next_states


def compute_importance_weights(behavioral_policy, target_policy, states, actions, episode_nr):
    # Initialize to None for the first concat
    importance_weights = None

    # Compute the importance weights
    # build iw vector incrementally from trajectory particles
    for episode in range(episode_nr):
        traj_states = states[episode]
        traj_actions = actions[episode]

        traj_target_log_p = target_policy.get_log_p(traj_states, traj_actions)
        traj_behavior_log_p = behavioral_policy.get_log_p(traj_states, traj_actions)

        traj_particle_iw = torch.exp(torch.cumsum(traj_target_log_p - traj_behavior_log_p, dim = 0))

        if importance_weights is None:
            importance_weights = traj_particle_iw
        else:
            importance_weights = torch.cat([importance_weights, traj_particle_iw], dim=0)

    # Normalize the weights
    importance_weights /= torch.sum(importance_weights)
    return importance_weights


def compute_entropy(behavioral_policy, target_policy, states, actions,
                    episode_nr, distances, indices, k, G, B, ns, eps):
    importance_weights = compute_importance_weights(behavioral_policy, target_policy, states, actions, episode_nr)
    # Compute objective function
    # compute weights sum for each particle
    weights_sum = torch.sum(importance_weights[indices[:, : -1]], dim=1)
    # compute volume for each particle
    volumes = (torch.pow(distances[:, k], ns) * torch.pow(torch.tensor(np.pi), ns / 2)) / G
    # compute entropy
    entropy = -torch.sum((weights_sum / k) * torch.log((weights_sum / (volumes + eps)) + eps)) + B

    return entropy


def compute_kl(behavioral_policy, target_policy, states, actions,
               episode_nr, indices, k, eps):
    importance_weights = compute_importance_weights(behavioral_policy, target_policy, states, actions, episode_nr)

    weights_sum = torch.sum(importance_weights[indices[:, : -1]], dim = 1)

    # Compute KL divergence between behavioral and target policy
    N = importance_weights.shape[0]
    kl = (1 / N) * torch.sum(torch.log(k / (N * weights_sum) + eps))

    numeric_error = torch.isinf(kl) or torch.isnan(kl)

    # Minimum KL is zero
    # NOTE: do not remove epsilon factor
    kl = torch.max(torch.tensor(0.0), kl)

    return kl, numeric_error


def collect_particles_and_compute_knn(env, behavioral_policy, episode_nr, step_nr,
                                      state_dim, action_dim, k, num_workers):
    assert episode_nr % num_workers == 0, "Please provide a number of trajectories " \
                                        "that can be equally split among workers"

    # Collect particles using behavioral policy
    res = Parallel(n_jobs=num_workers)(
        delayed(collect_particles)(env, behavioral_policy, int(episode_nr / num_workers), step_nr, state_dim, action_dim)
        for _ in range(num_workers)
    )
    states, actions, costs, next_states = [np.vstack(x) for x in zip(*res)]

    # Fit knn for the batch of collected particles
    nbrs = NearestNeighbors(n_neighbors = k + 1, metric = 'euclidean', algorithm = 'auto', n_jobs = num_workers)
    nbrs.fit(next_states)
    distances, indices = nbrs.kneighbors(next_states)

    # Return tensors so that downstream computations can be moved to any target device (#todo)
    states = torch.tensor(states, dtype = torch.float64)
    actions = torch.tensor(actions, dtype = torch.float64)
    costs = torch.tensor(costs, dtype = torch.float64)
    next_states = torch.tensor(next_states, dtype = torch.float64)
    distances = torch.tensor(distances, dtype = torch.float64)
    indices = torch.tensor(indices, dtype = torch.int64)

    return states, actions, costs, next_states, distances, indices

def policy_update(optimizer, behavioral_policy, target_policy, states, actions,
                  episode_nr, distances, indices, k, G, B, ns, eps):
    optimizer.zero_grad()

    # Maximize entropy
    # TODO: Update this formula to include one extra term
    loss = -compute_entropy(behavioral_policy, target_policy, states, actions,
                            episode_nr, distances, indices, k, G, B, ns, eps)

    numeric_error = torch.isinf(loss) or torch.isnan(loss)

    loss.backward()
    optimizer.step()

    return loss, numeric_error

# def mepol(env, env_name, state_filter, create_policy, k, kl_threshold, max_off_iters,
#           use_backtracking, backtrack_coeff, max_backtrack_try, eps,
#           learning_rate, num_traj, traj_len, num_epochs, optimizer,
#           full_entropy_traj_scale, full_entropy_k,
#           heatmap_every, heatmap_discretizer, heatmap_episodes, heatmap_num_steps,
#           heatmap_cmap, heatmap_labels, heatmap_interp,
#           seed, out_path, num_workers):

# def CEM(env_id, epoch_nr = 500, step_nr = 1000, reward_free = False, 
#          delta = 0.1, lambda_omega = 1e-2, lambda_policy = 1e-5, 
#          k = 4, d = 25, gamma = 0.99, episode_nr = 20):
#     """
#     T: Number of trajectories
#     N: Number of time steps
#     delta: Trust-region threshold (Maximum KL Divergence between two avg state density distributions)
#     omega (see below): Safety weight/Lagrange multiplier (0 or larger)
#     lambda: Learning rate
#     k: Number of neighbors
#     d: Safety threshold
#     gamma: Discount factor of cost over time
#     """
#     D = {}
#     omega = 0.1
#     state_dct = {}
#     action_dct = {}

def CEM(env_id, create_policy, k, delta, max_off_iters,
          use_backtracking, backtrack_coeff, max_backtrack_try, eps,
          lambda_omega, lambda_policy, T, N, epoch_nr, optimizer,
          heatmap_every, heatmap_discretizer, heatmap_episodes, heatmap_num_steps,
          heatmap_cmap, heatmap_labels, heatmap_interp,
          seed, out_path, num_workers):    
    """
    T: Number of trajectories/episodes
    N: Number of time steps
    delta: Trust-region threshold (Maximum KL Divergence between two avg state density distributions)
    omega (see below): Safety weight/Lagrange multiplier (0 or larger)
    lambda: Learning rate
    k: Number of neighbors
    d: Safety threshold
    gamma: Discount factor of cost over time
    """
    omega = 0.1    

    env = safety_gymnasium.make(env_id, render_mode = 'human', max_episode_steps = N)
    # Seed everything
    if seed is not None:
        # Seed everything
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)

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

    # Create a behavioral, a target policy and a tmp policy used to save valid target policies
    # (those with kl <= kl_threshold) during off policy opt
    behavioral_policy = create_policy(is_behavioral=True)
    target_policy = create_policy()
    last_valid_target_policy = create_policy()
    target_policy.load_state_dict(behavioral_policy.state_dict())
    last_valid_target_policy.load_state_dict(behavioral_policy.state_dict())

    # Set optimizer
    policy_optimizer = torch.optim.Adam(target_policy.parameters(), lr = lambda_policy)

    # Fixed constants
    ns = state_dim
    B = np.log(k) - scipy.special.digamma(k)
    G = scipy.special.gamma(ns / 2 + 1)

    # At epoch 0 do not optimize, just log stuff for the initial policy
    epoch = 0
    t0 = time.time()

    # Full entropy
    states, actions, costs, next_states, distances, indices = \
        collect_particles_and_compute_knn(env, behavioral_policy, T, N, 
                                          state_dim, action_dim, k, num_workers)

    with torch.no_grad():
        full_entropy = compute_entropy(behavioral_policy, behavioral_policy, states, actions,
                                       T, distances, indices, k, G, B, ns, eps)

    # Entropy
    states, actions, costs, next_states, distances, indices = \
        collect_particles_and_compute_knn(env, behavioral_policy, T, N, 
                                          state_dim, action_dim, k, num_workers)

    with torch.no_grad():
        entropy = compute_entropy(behavioral_policy, behavioral_policy, states, actions,
                                  T, distances, indices, k, G, B, ns, eps)        

    execution_time = time.time() - t0
    full_entropy = full_entropy.numpy()
    entropy = entropy.numpy()
    loss = -entropy

    # Main Loop
    global_num_off_iters = 0

    if use_backtracking:
        original_lr = lambda_policy

    while epoch < epoch_nr:
        t0 = time.time()

        # Off policy optimization
        kl_threshold_reached = False
        last_valid_target_policy.load_state_dict(behavioral_policy.state_dict())
        num_off_iters = 0

        # Collect particles to optimize off policy
        states, actions, costs, next_states, distances, indices = \
                collect_particles_and_compute_knn(env, behavioral_policy, T, N, 
                                          state_dim, action_dim, k, num_workers)

        if use_backtracking:
            lambda_policy = original_lr

            for param_group in optimizer.param_groups:
                param_group['lr'] = lambda_policy

            backtrack_iter = 1
        else:
            backtrack_iter = None

        while not kl_threshold_reached:
            # Optimize policy      
            loss, numeric_error = policy_update(optimizer, behavioral_policy, target_policy, states,
                                                actions, T, distances, indices, k, G, B, ns, eps)
            entropy = -loss.detach().numpy()

            with torch.no_grad():
                kl, kl_numeric_error = compute_kl(behavioral_policy, target_policy, states, actions,
                                                  T, indices, k, eps)

            kl = kl.numpy()

            if not numeric_error and not kl_numeric_error and kl <= delta:
                # Valid update
                last_valid_target_policy.load_state_dict(target_policy.state_dict())
                num_off_iters += 1
                global_num_off_iters += 1

            else:
                if use_backtracking:
                    # We are here either because we could not perform any update for this epoch
                    # or because we need to perform one last update
                    if not backtrack_iter == max_backtrack_try:
                        target_policy.load_state_dict(last_valid_target_policy.state_dict())

                        lambda_policy = original_lr / (backtrack_coeff ** backtrack_iter)

                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lambda_policy

                        backtrack_iter += 1
                        continue

                # Do not accept the update, set exit condition to end the epoch
                kl_threshold_reached = True

            if use_backtracking and backtrack_iter > 1:
                # Just perform at most 1 step using backtracking
                kl_threshold_reached = True

            if num_off_iters == max_off_iters:
                # Set exit condition also if the maximum number
                # of off policy opt iterations has been reached
                kl_threshold_reached = True

            if kl_threshold_reached:
                # Compute entropy of new policy
                with torch.no_grad():
                    full_entropy = compute_entropy(behavioral_policy, behavioral_policy, states, actions,
                                                   T, distances, indices, k, G, B, ns, eps)

                if torch.isnan(entropy) or torch.isinf(entropy):
                    print("Aborting because final entropy is nan or inf...")
                    print("There is most likely a problem in knn aliasing. Use a higher k.")
                    exit()
                else:
                    # End of epoch, prepare statistics to log
                    epoch += 1

                    # Update behavioral policy
                    behavioral_policy.load_state_dict(last_valid_target_policy.state_dict())
                    target_policy.load_state_dict(last_valid_target_policy.state_dict())

                    loss = -entropy.numpy()
                    entropy = entropy.numpy()
                    execution_time = time.time() - t0

                    if epoch % heatmap_every == 0:
                        # Full entropy
                        states, actions, costs, next_states, distances, indices = \
                                collect_particles_and_compute_knn(env, behavioral_policy, T, N, 
                                                        state_dim, action_dim, k, num_workers)

                        with torch.no_grad():
                            full_entropy = compute_entropy(behavioral_policy, behavioral_policy, states, actions,
                                                           T, distances, indices, k, G, B, ns, eps)

                        full_entropy = full_entropy.numpy()

    return behavioral_policy

# def update(state: tuple[int, int], reward: float, action: int):
#     """Any code that processes a reward given the state and updates the agent.

#     Args:
#         state: The updated position of the agent.
#         reward: The value which is returned by the environment as a
#             reward.
#         action: The action which was taken by the agent.
#     """
#     # Q-Learning update
#     Q = np.zeros((grid.shape[0] * grid.shape[1], directions))
#     current_state = state_to_nr(state)
#     Q[old_state, action] += alpha * (reward + gamma * 
#                                                     np.max(Q[current_state, : ]) - 
#                                                     Q[old_state, action])  

# def CEM(env_id, epoch_nr = 500, step_nr = 1000, reward_free = False, 
#          delta = 0.1, lambda_omega = 1e-2, lambda_policy = 1e-5, 
#          k = 4, d = 25, gamma = 0.99, episode_nr = 20):
#     """
#     T: Number of trajectories
#     N: Number of time steps
#     delta: Trust-region threshold (Maximum KL Divergence between two avg state density distributions)
#     omega (see below): Safety weight/Lagrange multiplier (0 or larger)
#     lambda: Learning rate
#     k: Number of neighbors
#     d: Safety threshold
#     gamma: Discount factor of cost over time
#     """
#     D = {}
#     omega = 0.1
#     state_dct = {}
#     action_dct = {}

#     env = safety_gymnasium.make(env_id, render_mode = 'human', max_episode_steps = step_nr)
#     # To ensure mujoco is showing the same number of steps as reality
#     env.task.sim_conf.frameskip_binom_n = 1

#     # Initialize policy neural network (theta)
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
#     theta_nn = PolicyNetwork(state_dim, action_dim)
#     theta_optimizer = torch.optim.Adam(theta_nn.parameters(), lr = lambda_policy)
#     theta_nn = train_supervised(env, theta_nn, theta_optimizer)
    
#     # Create a copy of theta (theta')
#     theta_prime_nn = PolicyNetwork(state_dim, action_dim)
#     theta_prime_nn.load_state_dict(theta_nn.state_dict()) 

#     # Initialize Q value neural network
#     q_nn = QNetwork(state_dim, action_dim)
#     q_optimizer = torch.optim.Adam(q_nn.parameters(), lr = lambda_policy)
    
#     for epoch in trange(epoch_nr):
#         # Initialize the episode cost every time at the beginning of the episode
#         ep_ret, ep_cost = 0, 0
        
#         for _ in trange(episode_nr):
#             state, info = env.reset()
#             # Set seeds
#             # obs, _ = env.reset(seed=0)        
#             terminated, truncated = False, False
#             assert env.observation_space.contains(state)
#             for _ in range(step_nr):         
#                 # TODO: Sample using neural network instead
#                 action = theta_nn(state)
#                 # action = env.action_space.sample()
#                 assert env.action_space.contains(action)
#                 new_state, reward, cost, terminated, truncated, info = env.step(action)
#                 D, state_dct, action_dct = update_D(state, action, cost, new_state, D, state_dct, action_dct)
#                 ep_cost += cost
#                 # Only add the reward during deployment i.e. not reward-free
#                 if not reward_free:
#                     ep_ret += reward
#                     if terminated or truncated:
#                         break
#                 state = new_state

#         # Note that ep_cost is originally negative, but cost should be positive
#         total_episode_cost = -ep_cost / episode_nr - d
#         # Update omega
#         omega = max(0, omega + lambda_omega * total_episode_cost)

#         iter = 0
#         while iter < 30:
#             iter += 1
        
#         # Empty the dynamics before the next epoch starts
#         D = {}
#         # Update theta' using the newest values of theta
#         theta_prime_nn.load_state_dict(theta_nn.state_dict())

#     env.close()
#     return ep_ret, ep_cost    