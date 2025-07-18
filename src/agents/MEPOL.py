import torch
import torch.nn as nn
import numpy as np
import gymnasium
from tqdm import tqdm
import time
from sklearn.neighbors import NearestNeighbors
import scipy
import scipy.special
from src.init_env import create_envs
from src.policy import PolicyNetwork, train_supervised
from tabulate import tabulate
import matplotlib.pyplot as plt
import os
from src.discretizer import create_discretizer

int_choice = torch.int64
float_choice = torch.float64
torch.set_default_dtype(float_choice)

class MEPOL:
    def __init__(self, env_id, episode_nr, step_nr, parallel_envs=8, int_type=int_choice, float_type=float_choice,
                 k=1, alpha=0, zeta=1, delta=1, max_off_iters=1, use_backtracking=True, backtrack_coeff=0, 
                 max_backtrack_try=0, eps=1e-5, lambda_policy=1e-3, epoch_nr=500, 
                 out_path="", use_behavioral=False, state_dependent_std=False):    
        """
        T: Number of trajectories/episodes
        N: Number of time steps
        delta: Trust-region threshold (Maximum KL Divergence between two avg state density distributions)
        omega (see below): Safety weight/Lagrange multiplier (0 or larger)
        lambda: Learning rate
        k: Number of neighbors
        gamma: Discount factor of cost over time
        """   
        # == Environment Config ==
        self.env_id = env_id
        self.parallel_envs = parallel_envs
        self.out_path = out_path
        self.int_type = int_type
        self.float_type = float_type

        # == Algorithm Hyperparameters ==
        self.k = k
        self.alpha = alpha
        self.zeta = zeta
        self.delta = delta
        self.max_off_iters = max_off_iters
        self.use_backtracking = use_backtracking
        self.backtrack_coeff = backtrack_coeff
        self.max_backtrack_try = max_backtrack_try
        self.eps = eps
        self.lambda_policy = lambda_policy
        self.episode_nr = episode_nr
        self.step_nr = step_nr
        self.epoch_nr = epoch_nr
        self.gamma = 0.99
        self.patience = 50
        self.use_behavioral = use_behavioral

        # == Environment State ==  
        self.envs = None  
        self.device = None
        self.state_dependent_std = state_dependent_std
        self.is_discrete = False
        self.state_dim = 0
        self.action_dim = 0
        self.num_workers = 1     
        self.B = 0
        self.G = 0               

        # == Neural Networks ==
        self.behavioral_policy = None
        self.target_policy = None
        self.policy_optimizer = None

        # == Heatmap Settings ==
        self.heatmap_cmap = 'Blues'
        self.heatmap_interp = 'spline16'
        self.heatmap_discretizer = None
        if self.env_id == "MountainCarContinuous-v0" or self.env_id == "MountainCar-v0":
            self.heatmap_labels = ('Position', 'Velocity')
        elif self.env_id == "CartPole-v1":
            self.heatmap_labels = ('Pole Angle', 'Cart Position')   
        elif self.env_id == "Pendulum-v1":   
            self.heatmap_labels = ('Cosine', 'Sine')              
        else:
            self.heatmap_labels = ('X', 'Y')          


    # Environment and Setup
    def create_policy(self, is_behavioral=False):
        policy = PolicyNetwork(self.state_dim, self.action_dim, self.state_dependent_std, self.is_discrete, self.device).to(self.device)

        # if is_behavioral and not self.is_discrete:
        #     policy = train_supervised(self.envs, policy, self.lambda_policy, self.device, train_steps=100)

        return policy

    def collect_particles(self, epoch, behavioral=True):
        """
        Collects num_loops * parallel_envs episodes (particles) of trajectory data.
        Each loop collects one batch of trajectories in parallel across multiple environments.
        """
        states = np.zeros((self.episode_nr, self.step_nr + 1, self.state_dim), dtype=np.float32)
        if self.is_discrete:
            actions = np.zeros((self.episode_nr, self.step_nr), dtype=np.int32)
        else:
            actions = np.zeros((self.episode_nr, self.step_nr, self.action_dim), dtype=np.float32)
        costs = np.zeros((self.episode_nr, self.step_nr, 1), dtype=np.float32)
        next_states = np.zeros((self.episode_nr, self.step_nr, self.state_dim), dtype=np.float32)

        print("\nCollect particles")
        num_loops = self.episode_nr // self.parallel_envs
        for loop_idx in range(num_loops):
            # Reset the environments for this batch
            s, _ = self.envs.reset(seed = epoch * self.episode_nr + loop_idx * self.parallel_envs)

            loop_offset = loop_idx * self.parallel_envs  # where to store this batch
            for t in tqdm(range(self.step_nr)):
                # Sample action from policy
                if behavioral:
                    a, _ = self.behavioral_policy.predict(s)
                else:
                    a, _ = self.target_policy.predict(s)
                a = a.cpu().numpy()    

                # if self.is_discrete:
                #     a = a.squeeze(-1)

                # Record
                states[loop_offset: loop_offset + self.parallel_envs, t] = s
                actions[loop_offset: loop_offset + self.parallel_envs, t] = a

                # Step through environments
                s, _, cost, _, _, _ = self.envs.step(a)
                costs[loop_offset: loop_offset + self.parallel_envs, t] = cost.reshape(-1, 1)
                next_states[loop_offset: loop_offset + self.parallel_envs, t] = s

            # Final state
            states[loop_offset: loop_offset + self.parallel_envs, self.step_nr] = s

        return states, actions, costs, next_states

    def collect_particles_and_compute_knn(self, epoch):
        # Run simulations
        states, actions, costs, next_states = self.collect_particles(epoch)

        print("\nCompute KNN starts")
        # Fit kNN for state density estimation
        reshaped_next_states = next_states.reshape(-1, self.state_dim)  # (num_samples, state_dim)
        nbrs = NearestNeighbors(n_neighbors = self.k + 1, metric = 'euclidean', algorithm = 'auto', n_jobs = self.num_workers)
        nbrs.fit(reshaped_next_states)
        distances, indices = nbrs.kneighbors(reshaped_next_states)
        print("\nCompute KNN finishes")

        # Convert to PyTorch tensors
        states = torch.tensor(states, dtype=self.float_type, device=self.device)
        if self.is_discrete:
            actions = torch.tensor(actions, dtype=self.int_type, device=self.device)
        else:
            actions = torch.tensor(actions, dtype=self.float_type, device=self.device)
        costs = torch.tensor(costs, dtype=self.float_type, device=self.device)
        next_states = torch.tensor(next_states, dtype=self.float_type, device=self.device)
        distances = torch.tensor(distances, dtype=self.float_type, device=self.device)
        indices = torch.tensor(indices, dtype=self.int_type, device=self.device)

        return states, actions, costs, next_states, distances, indices

    def get_heatmap(self, title = None):
        """
        Builds a log-probability state visitation heatmap by running
        the policy in env. The heatmap is built using the provided
        discretizer.
        """
        # Initialize state visitation count
        average_state_dist = self.heatmap_discretizer.get_empty_mat()
        state_dist = np.zeros_like(average_state_dist)  # Track visits for this run

        print("\nGetting heatmap using vectorized environments...")

        num_loops = self.episode_nr // self.parallel_envs
        for loop_nr in tqdm(range(num_loops)):
            # Reset all environments (batched)
            s, _ = self.envs.reset(seed = loop_nr * self.parallel_envs)            
            for t in tqdm(range(self.step_nr)):
                # Convert states to tensor and predict actions
                with torch.inference_mode():
                    a, _ = self.behavioral_policy.predict(s)
                    a = a.cpu().numpy()

                # Step all environments at once
                s, _, _, _, _, _ = self.envs.step(a)

                # Discretize and count visited states for each environment
                for i in range(self.parallel_envs):
                    state_dist[self.heatmap_discretizer.discretize(s[i])] += 1

        # Normalize state visitation
        state_dist /= self.step_nr
        average_state_dist += state_dist
        average_entropy = scipy.stats.entropy(state_dist.ravel())

        # Plot heatmap
        plt.close()
        image_fig = plt.figure()

        plt.xticks([])
        plt.yticks([])
        plt.xlabel(self.heatmap_labels[0])
        plt.ylabel(self.heatmap_labels[1])

        if len(average_state_dist.shape) == 2:
            log_p = np.ma.log(average_state_dist)
            log_p_ravel = log_p.ravel()
            min_log_p_ravel = np.min(log_p_ravel)
            second_min_log_p_ravel = np.min(log_p_ravel[log_p_ravel != min_log_p_ravel])
            log_p_ravel[np.argmin(log_p_ravel)] = second_min_log_p_ravel
            plt.imshow(log_p.filled(min_log_p_ravel), interpolation=self.heatmap_interp, cmap=self.heatmap_cmap)
        else:
            plt.bar([i for i in range(self.heatmap_discretizer.bins_sizes[0])], average_state_dist)
        
        # Safety constraint position in real world coordinates
        safety_position = self.envs.get_safety_threshold()

        if self.env_id == 'CartPole-v1':
            # Get y-axis bin edges
            y_bin_edges = self.heatmap_discretizer.bins[1] # Assuming second dimension corresponds to y

            # Convert to pixel bin indices
            unsafe_y_min_index = np.searchsorted(y_bin_edges, -safety_position)
            unsafe_y_max_index = np.searchsorted(y_bin_edges, safety_position)

            # Shade above and below the safe region
            plt.axhspan(plt.ylim()[0], unsafe_y_min_index, color='red', alpha=0.3, label="Unsafe Region")
            plt.axhspan(unsafe_y_max_index, plt.ylim()[1], color='red', alpha=0.3)
        else:    
            # Get x-axis bin edges
            x_bin_edges = self.heatmap_discretizer.bins[0]  # Assuming first dimension corresponds to x

            # Convert to pixel bin indices
            unsafe_x_index = np.searchsorted(x_bin_edges, safety_position)

            # Shade vertical safety constraint
            plt.axvspan(plt.xlim()[0], unsafe_x_index, color='red', alpha=0.3, label="Unsafe Region")
        
        plt.legend()

        if title != None:
            plt.title(title)

        return average_state_dist, average_entropy, image_fig
        

    # Optimization and Evaluation
    def compute_importance_weights(self, behavioral_policy, target_policy, states, actions):
        # Initialize to None for the first concat
        importance_weights = None

        # Compute the importance weights
        # build iw vector incrementally from trajectory particles
        print("\nCompute importance weights")
        for episode in tqdm(range(self.episode_nr)):
            # Last state (terminal) is discarded to match the action length
            traj_states = states[episode, : -1]
            traj_actions = actions[episode]

            if self.is_discrete:
                traj_actions = traj_actions.unsqueeze(-1)    

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

    def compute_entropy(self, behavioral_policy, target_policy, states, actions,
                        distances, indices):
        if self.use_behavioral:
            # Behavioral Entropy
            beta = np.exp((1 - self.alpha) * np.log(np.log(self.state_dim)))
            Rk = distances[:, self.k]
            log_term = torch.log(Rk + self.eps)
            entropy = Rk * torch.exp(-beta * (log_term ** self.alpha)) * (log_term ** self.alpha)
            entropy_per_episode = entropy.view(self.episode_nr, self.step_nr)            
        else:
            importance_weights = self.compute_importance_weights(behavioral_policy, target_policy, states, actions)
            weights_sum = torch.sum(importance_weights[indices[:, :-1]], dim=1)
            volumes = (torch.pow(distances[:, self.k], self.state_dim) *
                    torch.pow(torch.tensor(np.pi), self.state_dim / 2)) / self.G

            # Shannon Entropy
            if self.zeta == 1:
                entropy_terms = -(weights_sum / self.k) * torch.log((weights_sum / (volumes + self.eps)) + self.eps)
                entropy_terms_per_episode = entropy_terms.view(self.episode_nr, self.step_nr)
                entropy_per_episode = torch.sum(entropy_terms_per_episode, dim=1) + self.B
            # Renyi Entropy
            else:
                density_estimate = weights_sum / (volumes + self.eps)
                density_estimate = density_estimate.view(self.episode_nr, self.step_nr)
                entropy_per_episode = (1.0 / (1.0 - self.zeta)) * torch.log(
                    torch.sum(density_estimate ** self.zeta, dim=1) + self.eps
                )

        mean_entropy = torch.mean(entropy_per_episode)
        std_entropy = torch.std(entropy_per_episode, unbiased=False)

        if self.is_discrete:
            entropy_bonus_weight = 0.01
            mean_entropy += entropy_bonus_weight * mean_entropy

        return mean_entropy, std_entropy

    def compute_discounted_cost(self, costs):
        """
        Computes the discounted sum of costs over timesteps for each trajectory.
        """
        # Create discount factors tensor of shape (traj_len, 1)
        discount_factors = (self.gamma ** torch.arange(self.step_nr, device=self.device)).view(1, self.step_nr, 1)

        # Compute discounted sum along timesteps (axis=1)
        discounted_costs = torch.sum(costs * discount_factors, dim=1)  # Shape (num_traj, 1)

        # Compute mean and variance across trajectories & move result to CPU
        mean_cost = discounted_costs.mean().cpu()
        sd_cost = discounted_costs.std().cpu()

        return mean_cost, sd_cost

    def compute_kl(self, behavioral_policy, target_policy, states, actions, indices):
        importance_weights = self.compute_importance_weights(behavioral_policy, target_policy, states, actions)

        weights_sum = torch.sum(importance_weights[indices[:, : -1]], dim = 1)

        # Compute KL divergence between behavioral and target policy
        kl = (1 / self.episode_nr / self.step_nr) * torch.sum(torch.log(self.k / (self.episode_nr * self.step_nr * weights_sum) + self.eps))

        numeric_error = torch.isinf(kl) or torch.isnan(kl)

        # Minimum KL is zero
        # NOTE: do not remove epsilon factor
        kl = torch.max(torch.tensor(0.0), kl)

        return kl, numeric_error

    def policy_update(self, optimizer, behavioral_policy, target_policy, states, actions, distances, indices):
        optimizer.zero_grad()

        # Maximize entropy
        mean_entropy, std_entropy = self.compute_entropy(behavioral_policy, target_policy, states, actions, distances, indices)
        loss = -mean_entropy

        numeric_error = torch.isinf(loss) or torch.isnan(loss)

        loss.backward()
        optimizer.step()

        return loss, numeric_error, mean_entropy, std_entropy


    # Logging
    def log_epoch_statistics(self, log_file, csv_file_1, csv_file_2, epoch,
                             policy_loss, mean_entropy, std_entropy, mean_cost, std_cost, num_off_iters, execution_time,
                             heatmap_image, heatmap_entropy, backtrack_iters, backtrack_lr):
        # Prepare tabulate table
        table = []
        fancy_float = lambda f : f"{f:.3f}"
        table.extend([
            ["Epoch", epoch],
            ["Execution time (s)", fancy_float(execution_time)],
            ["Entropy", fancy_float(mean_entropy)],
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
        csv_file_1.write(f"{epoch},{policy_loss},{mean_entropy},{std_entropy},{mean_cost},{std_cost},{num_off_iters},{execution_time}\n")
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

    def log_off_iter_statistics(self, csv_file_3, epoch, num_off_iter, global_off_iter,
                                mean_entropy, std_entropy, kl, mean_cost, std_cost, lr):
        # Log to csv file 3
        csv_file_3.write(f"{epoch},{num_off_iter},{global_off_iter},{mean_entropy},{mean_entropy},{kl},{mean_cost},{std_cost},{lr}\n")
        csv_file_3.flush()

    # Main Training
    def _initialize_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("\nUsing GPU")
        else:
            self.device = torch.device("cpu")
            print("\nUsing CPU")

    def _initialize_envs(self):
        self.envs = create_envs(self.env_id, self.step_nr, self.parallel_envs)
        self.heatmap_discretizer = create_discretizer(self.envs, self.env_id)
        self.num_workers = min(os.cpu_count(), self.parallel_envs)
        
        self.is_discrete = isinstance(self.envs.single_action_space, gymnasium.spaces.Discrete)
        self.state_dim = self.envs.single_observation_space.shape[0]
        if self.is_discrete:
            self.action_dim = self.envs.single_action_space.n
        else:
            self.action_dim = self.envs.single_action_space.shape[0]        
        
        # Fix constants
        self.B = np.log(self.k) - scipy.special.digamma(self.k)
        self.G = scipy.special.gamma(self.state_dim / 2 + 1)  

    def _initialize_networks(self):
        # Initialize policy neural network
        # Create a behavioral, a target policy and a tmp policy used to save valid target policies
        # (those with kl <= kl_threshold) during off policy opt
        self.behavioral_policy = self.create_policy(is_behavioral=True)
        self.target_policy = self.create_policy()
        last_valid_target_policy = self.create_policy()
        self.target_policy.load_state_dict(self.behavioral_policy.state_dict())
        last_valid_target_policy.load_state_dict(self.behavioral_policy.state_dict())

        # Set optimizer for policy
        self.policy_optimizer = torch.optim.Adam(self.target_policy.parameters(), lr = self.lambda_policy)

        return last_valid_target_policy       

    def _initialize_logging(self):
        # Create log files
        log_file = open(os.path.join((self.out_path), 'log_file.txt'), 'a', encoding="utf-8")

        csv_file_1 = open(os.path.join(self.out_path, f"{self.env_id}.csv"), 'w')
        csv_file_1.write(",".join(['epoch', 'loss', 'mean_entropy', 'std_entropy', 'mean_cost', 'std_cost', 'num_off_iters','execution_time']))
        csv_file_1.write("\n")

        if self.heatmap_discretizer is not None:
            csv_file_2 = open(os.path.join(self.out_path, f"{self.env_id}-heatmap.csv"), 'w')
            csv_file_2.write(",".join(['epoch', 'average_entropy']))
            csv_file_2.write("\n")
        else:
            csv_file_2 = None

        csv_file_3 = open(os.path.join(self.out_path, f"{self.env_id}_off_policy_iter.csv"), "w")
        csv_file_3.write(",".join(['epoch', 'off_policy_iter', 'global_off_policy_iter', 'mean_entropy', 'std_entropy', 'kl', 'mean_cost', 'learning_rate']))
        csv_file_3.write("\n")

        return log_file, csv_file_1, csv_file_2, csv_file_3

    def _plot_heatmap(self, best_epoch):
        # Heatmap
        if self.heatmap_discretizer is not None:
            heatmap_ver = "initial"
            if best_epoch != 0:
                model_link = os.path.join(self.out_path, f"{best_epoch}-policy.pt")
                self.behavioral_policy.load_state_dict(torch.load(model_link))
                heatmap_ver = "final"
            _, heatmap_entropy, heatmap_image = \
                self.get_heatmap()
            heatmap_image.savefig(f"{self.out_path}/{heatmap_ver}_heatmap.png")
            plt.close(heatmap_image)    
        else:
            heatmap_entropy = None
            heatmap_image = None   
        return heatmap_entropy, heatmap_image

    def _run_initial_evaluation(self, log_file, csv_file_1, csv_file_2):
        # At epoch 0 do not optimize, just log stuff for the initial policy
        print(f"\nInitial epoch starts")
        t0 = time.time()

        # Entropy
        states, actions, costs, next_states, distances, indices = \
            self.collect_particles_and_compute_knn(0)

        with torch.inference_mode():
            mean_entropy, std_entropy = self.compute_entropy(self.behavioral_policy, self.behavioral_policy, states, 
                                           actions, distances, indices)   
            policy_loss = -mean_entropy             

        print("\nEntropy computed")

        execution_time = time.time() - t0
        mean_entropy = mean_entropy.cpu().numpy()
        std_entropy = std_entropy.cpu().numpy()
        mean_cost, std_cost = self.compute_discounted_cost(costs)
        policy_loss = policy_loss.cpu().numpy()

        # Heatmap
        heatmap_entropy, heatmap_image = self._plot_heatmap(0)

        # Save initial policy
        torch.save(self.behavioral_policy.state_dict(), os.path.join(self.out_path, "0-policy.pt"))

        # Log statistics for the initial policy
        self.log_epoch_statistics(
            log_file=log_file, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
            epoch=0,
            policy_loss=policy_loss,
            mean_entropy=mean_entropy,
            std_entropy=std_entropy,
            mean_cost=mean_cost,
            std_cost=std_cost,
            num_off_iters=0,
            execution_time=execution_time,
            heatmap_image=heatmap_image,
            heatmap_entropy=heatmap_entropy,
            backtrack_iters=None,
            backtrack_lr=None
        )

        return policy_loss, heatmap_entropy, heatmap_image

    def _optimize_kl(self, states, actions, costs, distances, indices, original_lr, 
                     last_valid_target_policy, backtrack_iter, csv_file_3, epoch, num_off_iters, 
                     global_num_off_iters, mean_behavorial_costs, std_behavorial_costs):
        
        kl_threshold_reached = False

        while not kl_threshold_reached:
            print("\nOptimizing KL continues")             
            # Update target policy network    
            loss, numeric_error, mean_entropy, std_entropy = self.policy_update(self.policy_optimizer, self.behavioral_policy, 
                                                                    self.target_policy, states, actions, distances, indices)
            mean_entropy = mean_entropy.detach().cpu().numpy()
            std_entropy = std_entropy.detach().cpu().numpy()
            loss = loss.detach().cpu().numpy()

            with torch.inference_mode():
                kl, kl_numeric_error = self.compute_kl(self.behavioral_policy, self.target_policy, states, actions, indices)

            kl = kl.cpu().numpy()

            if not numeric_error and not kl_numeric_error and kl <= self.delta:
                # Valid update
                last_valid_target_policy.load_state_dict(self.target_policy.state_dict())
                num_off_iters += 1
                global_num_off_iters += 1
                lr = self.lambda_policy
                # Log statistics for this off policy iteration
                self.log_off_iter_statistics(csv_file_3, epoch, num_off_iters - 1, global_num_off_iters - 1,
                                             mean_entropy, std_entropy, kl, mean_behavorial_costs, std_behavorial_costs, lr)                

            else:
                if self.use_backtracking:
                    # We are here either because we could not perform any update for this epoch
                    # or because we need to perform one last update
                    if not backtrack_iter == self.max_backtrack_try:
                        self.target_policy.load_state_dict(last_valid_target_policy.state_dict())

                        self.lambda_policy = original_lr / (self.backtrack_coeff ** backtrack_iter)

                        for param_group in self.policy_optimizer.param_groups:
                            param_group['lr'] = self.lambda_policy

                        backtrack_iter += 1
                        continue

                # Do not accept the update, set exit condition to end the epoch
                kl_threshold_reached = True

            if self.use_backtracking and backtrack_iter > 1:
                # Just perform at most 1 step using backtracking
                kl_threshold_reached = True

            if num_off_iters == self.max_off_iters:
                # Set exit condition also if the maximum number
                # of off policy opt iterations has been reached
                kl_threshold_reached = True 

        return backtrack_iter, num_off_iters, global_num_off_iters
    
    def _epoch_train(self, policy_loss, last_valid_target_policy, log_file, csv_file_1, csv_file_2, csv_file_3, heatmap_entropy, heatmap_image):
        if self.use_backtracking:
            original_lr = self.lambda_policy

        best_loss = policy_loss
        best_epoch = 0
        patience_counter = 0
        global_num_off_iters = 0

        for epoch in range(1, self.epoch_nr + 1):
            print(f"Epoch {epoch} starts")
            t0 = time.time()

            # Off policy optimization
            last_valid_target_policy.load_state_dict(self.behavioral_policy.state_dict())
            num_off_iters = 0

            # Collect particles to optimize off policy
            states, actions, costs, next_states, distances, indices = \
                self.collect_particles_and_compute_knn(epoch - 1)

            if self.use_backtracking:
                self.lambda_policy = original_lr

                for param_group in self.policy_optimizer.param_groups:
                    param_group['lr'] = self.lambda_policy

                backtrack_iter = 1
            else:
                backtrack_iter = None

            mean_behavorial_costs, std_behavorial_costs = self.compute_discounted_cost(costs)

            backtrack_iter, num_off_iters, global_num_off_iters = self._optimize_kl(states, actions, costs, distances, indices, original_lr, last_valid_target_policy, backtrack_iter, csv_file_3, epoch, num_off_iters, global_num_off_iters, mean_behavorial_costs, std_behavorial_costs)

            # Compute entropy of new policy
            with torch.inference_mode():                   
                mean_entropy, std_entropy = self.compute_entropy(self.behavioral_policy, self.behavioral_policy, states, 
                                                actions, distances, indices)

            if torch.isnan(mean_entropy) or torch.isinf(mean_entropy):
                print("Aborting because final entropy is nan or inf...")
                print("There is most likely a problem in knn aliasing. Use a higher k.")
                exit()
            else:
                # End of epoch, prepare statistics to log
                # Update behavioral policy
                self.behavioral_policy.load_state_dict(last_valid_target_policy.state_dict())
                self.target_policy.load_state_dict(last_valid_target_policy.state_dict())

                loss = -mean_entropy.cpu().numpy()
                mean_entropy = mean_entropy.cpu().numpy()
                std_entropy = std_entropy.cpu().numpy()
                mean_cost, std_cost = self.compute_discounted_cost(costs)
                execution_time = time.time() - t0

                # Log statistics for the initial policy
                self.log_epoch_statistics(
                    log_file=log_file, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
                    epoch=epoch,
                    policy_loss=loss,
                    mean_entropy=mean_entropy,
                    std_entropy=std_entropy,
                    mean_cost=mean_cost,
                    std_cost=std_cost,
                    num_off_iters=num_off_iters,
                    execution_time=execution_time,
                    heatmap_image=heatmap_image,
                    heatmap_entropy=heatmap_entropy,
                    backtrack_iters=None,
                    backtrack_lr=None
                )    

                if loss < best_loss:
                    # Save policy
                    torch.save(self.behavioral_policy.state_dict(), os.path.join(self.out_path, f"{epoch}-policy.pt"))
                    patience_counter = 0
                    best_epoch = epoch
                else:
                    patience_counter += 1
                    if patience_counter == self.patience:
                        break  

        return best_epoch

    def train(self):    
        self._initialize_device()
        self._initialize_envs()
        last_valid_target_policy = self._initialize_networks()
        log_file, csv_file_1, csv_file_2, csv_file_3 = self._initialize_logging()

        policy_loss, heatmap_entropy, heatmap_image = self._run_initial_evaluation(log_file, csv_file_1, csv_file_2)

        best_epoch = self._epoch_train(policy_loss, last_valid_target_policy, log_file, csv_file_1, csv_file_2, csv_file_3, heatmap_entropy, heatmap_image)                                                                
        heatmap_entropy, heatmap_image = self._plot_heatmap(best_epoch)

        if isinstance(self.envs, gymnasium.vector.VectorEnv):
            self.envs.close()


    def plot_heatmap(self):    
        """
        T: Number of trajectories/episodes
        N: Number of time steps
        """
        self._initialize_device()
        self._initialize_envs()

        model_lst = ["./results/MountainCarContinuous/MEPOL/0-policy.pt", "./results/MountainCarContinuous/MEPOL/299-policy.pt"]
        title_lst = ["Initial", "Final"]

        for model_link, title_txt in zip(model_lst, title_lst):
            # Create a behavioral, a target policy and a tmp policy used to save valid target policies
            # (those with kl <= kl_threshold) during off policy opt
            self.behavioral_policy = PolicyNetwork(self.state_dim, self.action_dim, self.state_dependent_std, self.is_discrete, self.device)  # Recreate the model architecture
            self.behavioral_policy.load_state_dict(torch.load(model_link))
            self.behavioral_policy.to(self.device)  # Move model to the correct device (CPU/GPU)
            self.behavioral_policy.eval()  # Set to evaluation mode (disables dropout, batch norm, etc.)
            title = f"Heatmap of {title_txt} Epoch State Exploration"

            # Heatmap
            _, average_entropy, heatmap_image = \
                self.get_heatmap(title)
            print(f"\nHeatmap Entropy at {title_txt} Epoch: {average_entropy}")
            
            heatmap_image.savefig(f"./{title_txt}_heatmap.png")
            plt.close(heatmap_image)

        if isinstance(self.envs, gymnasium.vector.VectorEnv):
            self.envs.close()