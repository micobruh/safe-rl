import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import safety_gymnasium
from tqdm import tqdm, trange
import time
from sklearn.neighbors import NearestNeighbors
import scipy
import scipy.special
from src.policy import PolicyNetwork, train_supervised
from tabulate import tabulate
import matplotlib
import matplotlib.pyplot as plt
from torch.utils import tensorboard
import os
from src.discretizer import Discretizer, create_discretizer

int_type = torch.int64
float_type = torch.float64
torch.set_default_dtype(float_type)

import torch

def create_policy(env, state_dim, action_dim, learning_rate, device, is_behavioral=False):

    policy = PolicyNetwork(state_dim, action_dim, device).to(device)

    if is_behavioral:
        policy = train_supervised(env, policy, learning_rate, device, train_steps=100, batch_size=5000)

    return policy

def get_heatmap(envs, policy, discretizer, num_steps,
                cmap, interp, labels, device):
    """
    Builds a log-probability state visitation heatmap by running
    the policy in env. The heatmap is built using the provided
    discretizer.
    """
    num_envs = envs.num_envs  # Number of parallel environments

    # Initialize state visitation count
    average_state_dist = discretizer.get_empty_mat()
    state_dist = np.zeros_like(average_state_dist)  # Track visits for this run

    print("\nGetting heatmap using vectorized environments...")

    # Reset all environments (batched)
    s, _ = envs.reset()

    for t in tqdm(range(num_steps)):
        # Convert states to tensor and predict actions
        with torch.inference_mode():
            a = policy.predict(s).cpu().numpy()  # Move actions to CPU for envs.step()

        # Step all environments at once
        s, _, _, _, _, _ = envs.step(a)

        # Discretize and count visited states for each environment
        for i in range(num_envs):
            state_dist[discretizer.discretize(s[i])] += 1

        # # Reset finished environments while keeping ongoing ones
        # if np.any(dones):
        #     s, _ = envs.reset_done()

    # Normalize state visitation
    state_dist /= num_steps
    average_state_dist += state_dist
    average_entropy = scipy.stats.entropy(state_dist.ravel())

    # Plot heatmap
    plt.close()
    image_fig = plt.figure()

    plt.xticks([])
    plt.yticks([])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    if len(average_state_dist.shape) == 2:
        log_p = np.ma.log(average_state_dist)
        log_p_ravel = log_p.ravel()
        min_log_p_ravel = np.min(log_p_ravel)
        second_min_log_p_ravel = np.min(log_p_ravel[log_p_ravel != min_log_p_ravel])
        log_p_ravel[np.argmin(log_p_ravel)] = second_min_log_p_ravel
        plt.imshow(log_p.filled(min_log_p_ravel), interpolation=interp, cmap=cmap)
    else:
        plt.bar([i for i in range(discretizer.bins_sizes[0])], average_state_dist)

    return average_state_dist, average_entropy, image_fig

def collect_particles(envs, policy, num_traj, traj_len, state_dim, action_dim, device):
    """
    Collects num_traj * traj_len samples in parallel across multiple environments.
    """
    states = np.zeros((num_traj, traj_len + 1, state_dim), dtype=np.float32)
    actions = np.zeros((num_traj, traj_len, action_dim), dtype=np.float32)
    costs = np.zeros((num_traj, traj_len, 1), dtype=np.float32)  # Cost is a scalar
    next_states = np.zeros((num_traj, traj_len, state_dim), dtype=np.float32)

    # Reset all environments
    s, _ = envs.reset()

    print("\nCollect particles")
    for t in tqdm(range(traj_len)):
        # Store state
        states[:, t] = s

        # Sample action from policy
        a = policy.predict(s).cpu().numpy()
        actions[:, t] = a

        # Apply action to all environments
        s, _, cost, _, _, _ = envs.step(a)

        # Store cost and next state
        costs[:, t] = cost.reshape(-1, 1)
        next_states[:, t] = s

    # Store final state
    states[:, traj_len] = s

    return states, actions, costs, next_states

# def collect_particles(env, policy, episode_nr, step_nr, state_dim, action_dim):
#     """
#     Collects samples by running policy in the env.
#     """
#     # State's dimension on step nr is one more to account for next state
#     states = np.zeros((episode_nr, step_nr + 1, state_dim), dtype = np.float32)
#     actions = np.zeros((episode_nr, step_nr, action_dim), dtype = np.float32)
#     costs = np.zeros((episode_nr, step_nr, action_dim), dtype = np.float32)
#     next_states = np.zeros((episode_nr, step_nr, state_dim), dtype = np.float32)

#     # Storing information in "D"
#     for episode in range(episode_nr):
#         s = env.reset()
#         for t in range(step_nr):
#             states[episode, t] = s
#             a = policy.predict(s).numpy()
#             actions[episode, t] = a
#             ns, _, c, _, _, _ = env.step(a)
#             costs[episode, t] = c
#             next_states[episode, t] = ns
#             s = ns
#         # Indicating the final (next) state
#         states[episode, t + 1] = s

#     return states, actions, costs, next_states

def compute_importance_weights(behavioral_policy, target_policy, states, actions, episode_nr):
    # Initialize to None for the first concat
    importance_weights = None

    # Compute the importance weights
    # build iw vector incrementally from trajectory particles
    print("\nCompute importance weights")
    for episode in tqdm(range(episode_nr)):
        # Last state (terminal) is discarded to match the action length
        traj_states = states[episode, : -1]
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

def compute_discounted_cost(costs, device, gamma=0.99):
    """
    Computes the discounted sum of costs over timesteps for each trajectory.
    """
    num_traj, traj_len, _ = costs.shape  # Extract shape

    # Create discount factors tensor of shape (traj_len, 1)
    discount_factors = (gamma ** torch.arange(traj_len, device=device)).view(1, traj_len, 1)

    # Compute discounted sum along timesteps (axis=1)
    discounted_costs = torch.sum(costs * discount_factors, dim=1)  # Shape (num_traj, 1)

    # Compute mean across trajectories & move result to CPU
    aggregated_cost = discounted_costs.mean().cpu()

    return aggregated_cost  # Final output is a scalar on CPU

def collect_particles_and_compute_knn(envs, behavioral_policy, episode_nr, step_nr,
                                      state_dim, action_dim, k, num_workers, device):
    # Run simulations
    states, actions, costs, next_states = collect_particles(envs, behavioral_policy, episode_nr, step_nr, state_dim, action_dim, device)

    print("\nCompute KNN starts")
    # Fit kNN for state density estimation
    reshaped_next_states = next_states.reshape(-1, state_dim)  # (num_samples, state_dim)
    nbrs = NearestNeighbors(n_neighbors = k + 1, metric = 'euclidean', algorithm = 'auto', n_jobs = num_workers)
    nbrs.fit(reshaped_next_states)
    distances, indices = nbrs.kneighbors(reshaped_next_states)
    print("\nCompute KNN finishes")

    # Convert to PyTorch tensors
    states = torch.tensor(states, dtype=float_type, device=device)
    actions = torch.tensor(actions, dtype=float_type, device=device)
    costs = torch.tensor(costs, dtype=float_type, device=device)
    next_states = torch.tensor(next_states, dtype=float_type, device=device)
    distances = torch.tensor(distances, dtype=float_type, device=device)
    indices = torch.tensor(indices, dtype=int_type, device=device)

    return states, actions, costs, next_states, distances, indices



# def collect_particles_and_compute_knn(env, behavioral_policy, episode_nr, step_nr,
#                                       state_dim, action_dim, k, num_workers):
#     assert episode_nr % num_workers == 0, "Please provide a number of trajectories " \
#                                         "that can be equally split among workers"

#     # Collect particles using behavioral policy
#     res = Parallel(n_jobs=num_workers)(
#         delayed(collect_particles)(env, behavioral_policy, int(episode_nr / num_workers), step_nr, state_dim, action_dim)
#         for _ in range(num_workers)
#     )
#     states, actions, costs, next_states = [np.vstack(x) for x in zip(*res)]

#     # Fit knn for the batch of collected particles
#     nbrs = NearestNeighbors(n_neighbors = k + 1, metric = 'euclidean', algorithm = 'auto', n_jobs = num_workers)
#     nbrs.fit(next_states)
#     distances, indices = nbrs.kneighbors(next_states)

#     # Return tensors so that downstream computations can be moved to any target device (#todo)
#     states = torch.tensor(states, dtype = float_type)
#     actions = torch.tensor(actions, dtype = float_type)
#     costs = torch.tensor(costs, dtype = float_type)
#     next_states = torch.tensor(next_states, dtype = float_type)
#     distances = torch.tensor(distances, dtype = float_type)
#     indices = torch.tensor(indices, dtype = int_type)

#     return states, actions, costs, next_states, distances, indices

def log_epoch_statistics(writer, log_file, csv_file_1, csv_file_2, epoch,
                         loss, entropy, mean_cost, num_off_iters, execution_time,
                         heatmap_image, heatmap_entropy, backtrack_iters, backtrack_lr):
    # Log to Tensorboard
    writer.add_scalar("Loss", loss, global_step=epoch)
    writer.add_scalar("Entropy", entropy, global_step=epoch)
    writer.add_scalar("Cost", mean_cost, global_step=epoch)
    writer.add_scalar("Execution time", execution_time, global_step=epoch)
    writer.add_scalar("Number off-policy iteration", num_off_iters, global_step=epoch)

    if heatmap_image is not None:
        writer.add_figure(f"Heatmap", heatmap_image, global_step=epoch)
        writer.add_scalar(f"Discrete entropy", heatmap_entropy, global_step=epoch)

    # Prepare tabulate table
    table = []
    fancy_float = lambda f : f"{f:.3f}"
    table.extend([
        ["Epoch", epoch],
        ["Execution time (s)", fancy_float(execution_time)],
        ["Entropy", fancy_float(entropy)],
        ["Cost", fancy_float(mean_cost)],
        ["Off-policy iters", num_off_iters]
    ])

    if heatmap_image is not None:
        table.extend([
            ["Heatmap entropy", fancy_float(heatmap_entropy)]
        ])

    if backtrack_iters is not None:
        table.extend([
            ["Backtrack iters", backtrack_iters],
        ])

    fancy_grid = tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign='right')

    # Log to csv file 1
    csv_file_1.write(f"{epoch},{loss},{entropy},{mean_cost},{num_off_iters},{execution_time}\n")
    csv_file_1.flush()

    # Log to csv file 2
    if heatmap_image is not None:
        csv_file_2.write(f"{epoch},{heatmap_entropy}\n")
        csv_file_2.flush()

    # Log to stdout and log file
    log_file.write(fancy_grid)
    log_file.write("\n\n")
    log_file.flush()
    print(fancy_grid)

def log_off_iter_statistics(writer, csv_file_3, epoch, global_off_iter,
                            num_off_iter, entropy, kl, mean_cost, lr):
    # Log to csv file 3
    csv_file_3.write(f"{epoch},{num_off_iter},{entropy},{kl},{mean_cost},{lr}\n")
    csv_file_3.flush()

    # Also log to tensorboard
    writer.add_scalar("Off policy iter Entropy", entropy, global_step=global_off_iter)
    writer.add_scalar("Off policy iter Cost", mean_cost, global_step=global_off_iter)
    writer.add_scalar("Off policy iter KL", kl, global_step=global_off_iter)

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

def MEPOL(env_id, k, delta, max_off_iters,
          use_backtracking, backtrack_coeff, max_backtrack_try, eps,
          lambda_policy, T, N, epoch_nr,
          heatmap_every, heatmap_cmap, heatmap_labels, heatmap_interp,
          seed, out_path):    
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
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("\nThere is GPU")
    else:
        dev = "cpu"
        print("\nThere is no GPU")
    device = torch.device(dev)    

    envs = safety_gymnasium.vector.make(env_id, max_episode_steps = N, num_envs = T)
    heatmap_discretizer = create_discretizer(envs)
    num_workers = min(os.cpu_count(), T)
    # # Seed everything
    # if seed is not None:
    #     # Seed everything
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     envs.seed(seed)

    # Initialize policy neural network (theta)
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    # theta_nn = PolicyNetwork(state_dim, action_dim)
    # theta_nn = train_supervised(envs, theta_nn, lambda_policy, device)
    
    # # Create a copy of theta (theta')
    # theta_prime_nn = PolicyNetwork(state_dim, action_dim)
    # theta_prime_nn.load_state_dict(theta_nn.state_dict()) 

    # theta_nn.to(device)
    # theta_prime_nn.to(device)

    # Create a behavioral, a target policy and a tmp policy used to save valid target policies
    # (those with kl <= kl_threshold) during off policy opt
    behavioral_policy = create_policy(envs, state_dim, action_dim, lambda_policy, device, is_behavioral=True)
    target_policy = create_policy(envs, state_dim, action_dim, lambda_policy, device)
    last_valid_target_policy = create_policy(envs, state_dim, action_dim, lambda_policy, device)
    target_policy.load_state_dict(behavioral_policy.state_dict())
    last_valid_target_policy.load_state_dict(behavioral_policy.state_dict())

    # Set optimizer for policy
    optimizer = torch.optim.Adam(target_policy.parameters(), lr = lambda_policy)

    # Create writer for tensorboard
    writer = tensorboard.SummaryWriter(out_path)

    # Create log files
    log_file = open(os.path.join((out_path), 'log_file.txt'), 'a', encoding="utf-8")

    csv_file_1 = open(os.path.join(out_path, f"{env_id}.csv"), 'w')
    csv_file_1.write(",".join(['epoch', 'loss', 'entropy', 'num_off_iters','execution_time']))
    csv_file_1.write("\n")

    if heatmap_discretizer is not None:
        csv_file_2 = open(os.path.join(out_path, f"{env_id}-heatmap.csv"), 'w')
        csv_file_2.write(",".join(['epoch', 'average_entropy']))
        csv_file_2.write("\n")
    else:
        csv_file_2 = None

    csv_file_3 = open(os.path.join(out_path, f"{env_id}_off_policy_iter.csv"), "w")
    csv_file_3.write(",".join(['epoch', 'off_policy_iter', 'entropy', 'kl', 'learning_rate']))
    csv_file_3.write("\n")

    # Fixed constants
    ns = state_dim
    B = np.log(k) - scipy.special.digamma(k)
    G = scipy.special.gamma(ns / 2 + 1)

    # At epoch 0 do not optimize, just log stuff for the initial policy
    epoch = 0
    print(f"\nInitial epoch starts")
    t0 = time.time()

    # Entropy
    states, actions, costs, next_states, distances, indices = \
        collect_particles_and_compute_knn(envs, behavioral_policy, T, N, 
                                  state_dim, action_dim, k, num_workers, device)

    with torch.inference_mode():
        entropy = compute_entropy(behavioral_policy, behavioral_policy, states, actions,
                                  T, distances, indices, k, G, B, ns, eps)        

    print("\nEntropy computed")

    execution_time = time.time() - t0
    entropy = entropy.cpu().numpy()
    loss = -entropy
    mean_cost = compute_discounted_cost(costs, device, gamma=0.99)

    # Heatmap
    if heatmap_discretizer is not None:
        _, heatmap_entropy, heatmap_image = \
            get_heatmap(envs, behavioral_policy, heatmap_discretizer, N,
                        heatmap_cmap, heatmap_interp, heatmap_labels, device)
    else:
        heatmap_entropy = None
        heatmap_image = None

    # Save initial policy
    torch.save(behavioral_policy.state_dict(), os.path.join(out_path, f"{epoch}-policy"))

    # Log statistics for the initial policy
    log_epoch_statistics(
            writer=writer, log_file=log_file, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
            epoch=epoch,
            loss=loss,
            entropy=entropy,
            mean_cost=mean_cost,
            execution_time=execution_time,
            num_off_iters=0,
            heatmap_image=heatmap_image,
            heatmap_entropy=heatmap_entropy,
            backtrack_iters=None,
            backtrack_lr=None
        )

    # Main Loop
    global_num_off_iters = 0

    if use_backtracking:
        original_lr = lambda_policy

    while epoch < epoch_nr:
        print(f"Epoch {epoch} starts")
        t0 = time.time()

        # Off policy optimization
        kl_threshold_reached = False
        last_valid_target_policy.load_state_dict(behavioral_policy.state_dict())
        num_off_iters = 0

        # Collect particles to optimize off policy
        states, actions, costs, next_states, distances, indices = \
            collect_particles_and_compute_knn(envs, behavioral_policy, T, N, 
                                  state_dim, action_dim, k, num_workers, device)

        if use_backtracking:
            lambda_policy = original_lr

            for param_group in optimizer.param_groups:
                param_group['lr'] = lambda_policy

            backtrack_iter = 1
        else:
            backtrack_iter = None

        while not kl_threshold_reached:
            print("\nOptimizing KL continues")
            # Optimize policy      
            loss, numeric_error = policy_update(optimizer, behavioral_policy, target_policy, states,
                                                actions, T, distances, indices, k, G, B, ns, eps)
            entropy = -loss.detach().cpu().numpy()

            with torch.inference_mode():
                kl, kl_numeric_error = compute_kl(behavioral_policy, target_policy, states, actions,
                                                  T, indices, k, eps)

            kl = kl.cpu().numpy()

            mean_cost = compute_discounted_cost(costs, device, gamma=0.99)

            if not numeric_error and not kl_numeric_error and kl <= delta:
                # Valid update
                last_valid_target_policy.load_state_dict(target_policy.state_dict())
                num_off_iters += 1
                global_num_off_iters += 1
                # Log statistics for this off policy iteration
                log_off_iter_statistics(writer, csv_file_3, epoch, global_num_off_iters,
                                        num_off_iters - 1, entropy, kl, mean_cost, lambda_policy)                

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
                with torch.inference_mode():
                    entropy = compute_entropy(behavioral_policy, behavioral_policy, states, actions,
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

                    loss = -entropy.cpu().numpy()
                    entropy = entropy.cpu().numpy()
                    mean_cost = compute_discounted_cost(costs, device, gamma=0.99)
                    execution_time = time.time() - t0

                    if epoch % heatmap_every == 0:
                        # Heatmap
                        if heatmap_discretizer is not None:
                            _, heatmap_entropy, heatmap_image = \
                                get_heatmap(envs, behavioral_policy, heatmap_discretizer, N,
                                            heatmap_cmap, heatmap_interp, heatmap_labels, device)
                        else:
                            heatmap_entropy = None
                            heatmap_image = None

                        # Full entropy
                        states, actions, costs, next_states, distances, indices = \
                            collect_particles_and_compute_knn(envs, behavioral_policy, T, N, 
                                                    state_dim, action_dim, k, num_workers, device)
                        
                        # Save policy
                        torch.save(behavioral_policy.state_dict(), os.path.join(out_path, f"{epoch}-policy"))  

                    else:
                        heatmap_entropy = None
                        heatmap_image = None

                    # Log statistics for this epoch
                    log_epoch_statistics(
                        writer=writer, log_file=log_file, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
                        epoch=epoch,
                        loss=loss,
                        entropy=entropy,
                        mean_cost=mean_cost,
                        execution_time=execution_time,
                        num_off_iters=num_off_iters,
                        heatmap_image=heatmap_image,
                        heatmap_entropy=heatmap_entropy,
                        backtrack_iters=backtrack_iter,
                        backtrack_lr=lambda_policy
                    )                                              

    return behavioral_policy