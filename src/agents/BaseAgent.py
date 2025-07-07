import torch
import torch.nn as nn
import numpy as np
import gymnasium
import safety_gymnasium
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import scipy
import scipy.special
from src.init_env import create_envs
from src.policy import PolicyNetwork, train_supervised
import matplotlib.pyplot as plt
import os
from src.discretizer import create_discretizer

int_choice = torch.int64
float_choice = torch.float64
torch.set_default_dtype(float_choice)

class BaseAgent:
    def __init__(self, env_id, episode_nr, step_nr, parallel_envs=8, int_type=int_choice, float_type=float_choice, omega=100,
                 k=1, alpha=0, zeta=1, delta=1, use_backtracking=True, eps=1e-5, d=1, lambda_policy=1e-3, lambda_cost_value=1e-3, 
                 lambda_omega=1e-3, epoch_nr=500, out_path="", use_behavioral=False, state_dependent_std=False):    
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
        self.use_backtracking = use_backtracking
        self.eps = eps
        self.d = d
        self.lambda_policy = lambda_policy
        self.lambda_cost_value = lambda_cost_value
        self.lambda_omega = lambda_omega
        self.episode_nr = episode_nr
        self.step_nr = step_nr        
        self.epoch_nr = epoch_nr
        self.gamma = 0.99
        self.omega = omega
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
        self.cost_value_nn = None
        self.policy_optimizer = None
        self.cost_value_optimizer = None

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
        first_layer_neuron = 400 if self.env_id == "SafetyPointGoal1-v0" else 300
        second_layer_neuron = 300
        policy = PolicyNetwork(self.state_dim, self.action_dim, first_layer_neuron, second_layer_neuron, self.state_dependent_std, self.is_discrete, self.device).to(self.device)

        # if is_behavioral and not self.is_discrete:
        #     policy = train_supervised(self.envs, policy, self.lambda_policy, self.device, train_steps=100)

        return policy

    def collect_particles(self, epoch, state_dist=None):
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
        stds_or_probs = np.zeros((self.episode_nr, self.step_nr, self.action_dim), dtype=np.float32)

        print("\nCollect particles")
        num_loops = self.episode_nr // self.parallel_envs
        for loop_idx in range(num_loops):
            # Reset the environments for this batch
            s, _ = self.envs.reset(seed = epoch * self.episode_nr + loop_idx * self.parallel_envs)

            loop_offset = loop_idx * self.parallel_envs  # where to store this batch
            for t in tqdm(range(self.step_nr)):
                # Sample action from policy
                a, std_or_prob = self.behavioral_policy.predict(s)
                a = a.cpu().numpy()
                std_or_prob = std_or_prob.cpu().numpy()    

                # Record
                states[loop_offset: loop_offset + self.parallel_envs, t] = s
                actions[loop_offset: loop_offset + self.parallel_envs, t] = a
                stds_or_probs[loop_offset: loop_offset + self.parallel_envs, t] = std_or_prob

                # Step through environments
                s, _, cost, _, _, _ = self.envs.step(a)
                costs[loop_offset: loop_offset + self.parallel_envs, t] = cost.reshape(-1, 1)
                next_states[loop_offset: loop_offset + self.parallel_envs, t] = s

                # Discretize and count visited states for each environment
                if state_dist is not None:
                    for i in range(self.parallel_envs):
                        state_dist[self.heatmap_discretizer.discretize(s[i])] += 1                

            # Final state
            states[loop_offset: loop_offset + self.parallel_envs, self.step_nr] = s

        if state_dist is not None:
            state_dist /= (self.step_nr * self.parallel_envs)
            return states, actions, costs, next_states, state_dist
        
        return states, actions, costs, next_states, stds_or_probs
    
    def collect_particles_and_compute_knn(self, epoch, state_dist=None):
        # Run simulations
        if state_dist is not None:
            states, actions, costs, next_states, state_dist = self.collect_particles(epoch, state_dist)
        else:
            states, actions, costs, next_states, _ = self.collect_particles(epoch)  

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

        if state_dist is not None:
            return states, actions, costs, next_states, distances, indices, state_dist
        
        return states, actions, costs, next_states, distances, indices
    
    def get_heatmap(self, title=None):
        """
        Builds a log-probability state visitation heatmap by running
        the policy in env. The heatmap is built using the provided
        discretizer.
        """
        print("\nGetting heatmap using vectorized environments...")
        
        # Initialize state visitation count
        state_dist = self.heatmap_discretizer.get_empty_mat()
        states, actions, costs, next_states, distances, indices, state_dist = self.collect_particles_and_compute_knn(0, state_dist)
        with torch.inference_mode():
            importance_weights = self.compute_importance_weights(self.behavioral_policy, self.behavioral_policy, states, actions, True)
            mean_entropy, std_entropy = self.compute_entropy(distances, indices, self.use_behavioral, self.zeta, importance_weights)   
        mean_entropy = mean_entropy.cpu().numpy()
        std_entropy = std_entropy.cpu().numpy()
        mean_cost, std_cost = self.compute_discounted_cost(costs)

        # Plot heatmap
        plt.close()
        image_fig = plt.figure()

        plt.xticks([])
        plt.yticks([])
        plt.xlabel(self.heatmap_labels[0])
        plt.ylabel(self.heatmap_labels[1])

        if len(state_dist.shape) == 2:
            log_p = np.ma.log(state_dist)
            log_p_ravel = log_p.ravel()
            min_log_p_ravel = np.min(log_p_ravel)
            second_min_log_p_ravel = np.min(log_p_ravel[log_p_ravel != min_log_p_ravel])
            log_p_ravel[np.argmin(log_p_ravel)] = second_min_log_p_ravel
            plt.imshow(log_p.filled(min_log_p_ravel), interpolation=self.heatmap_interp, cmap=self.heatmap_cmap)
        else:
            plt.bar([i for i in range(self.heatmap_discretizer.bins_sizes[0])], state_dist)
        
        # Safety constraint position in real world coordinates
        safety_position = self.envs.get_safety_threshold()

        if self.env_id == 'CartPole-v1':
            # Get y-axis bin edges
            y_bin_edges = self.heatmap_discretizer.bins[1] # Assuming second dimension corresponds to y

            # Convert to pixel bin indices
            unsafe_y_min_index = np.searchsorted(y_bin_edges, -safety_position)
            unsafe_y_max_index = np.searchsorted(y_bin_edges, safety_position)

            # Shade above and below the safe region
            plt.axhspan(unsafe_y_min_index, plt.ylim()[1], color='red', alpha=0.3, label="Unsafe Region")
            plt.axhspan(plt.ylim()[0], unsafe_y_max_index, color='red', alpha=0.3)
        else:    
            # Get x-axis bin edges
            x_bin_edges = self.heatmap_discretizer.bins[0]  # Assuming first dimension corresponds to x

            # Convert to pixel bin indices
            unsafe_x_index = np.searchsorted(x_bin_edges, safety_position)

            # Shade vertical safety constraint
            plt.axvspan(plt.xlim()[0], unsafe_x_index, color='red', alpha=0.3, label="Unsafe Region")
        
        plt.legend()

        if title is not None:
            plt.title(title)

        return state_dist, mean_entropy, mean_cost, image_fig
    

    # Optimization and Evaluation
    def compute_importance_weights(self, behavioral_policy, target_policy, states, actions, same_policy=False):
        # Skip computation if the same policies are used
        if same_policy:
            num_samples = self.episode_nr * self.step_nr
            importance_weights = torch.ones(num_samples, dtype=self.float_type, device=self.device)
            importance_weights /= num_samples
            return importance_weights
        
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

    def compute_entropy(self, distances, indices, use_behavioral, zeta, importance_weights):
        if use_behavioral:
            # Behavioral Entropy
            beta = np.exp((1 - self.alpha) * np.log(np.log(self.state_dim)))
            Rk = distances[:, self.k]
            log_term = torch.log(Rk + self.eps)
            entropy = Rk * torch.exp(-beta * (log_term ** self.alpha)) * (log_term ** self.alpha)
            entropy_per_episode = entropy.view(self.episode_nr, self.step_nr)            
        else:
            weights_sum = torch.sum(importance_weights[indices[:, : -1]], dim=1)
            volumes = (torch.pow(distances[:, self.k], self.state_dim) *
                    torch.pow(torch.tensor(np.pi), self.state_dim / 2)) / self.G

            # Shannon Entropy
            if zeta == 1:
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
        std_cost = discounted_costs.std().cpu()

        return mean_cost, std_cost
    
    def compute_cost_advantage(self, states, costs, update = False):
        V = self.cost_value_nn(states)

        old_V = V[:, : -1, :]
        new_V = V[:, 1:, :]

        if update:
            target = costs + self.gamma * new_V.detach()
            # Compute value loss for value network
            # MSE between predicted V'(s) and V(s), i.e. E(A(s, a) ** 2)
            value_loss = nn.functional.mse_loss(old_V, target)
            return value_loss

        # Compute advantage loss (A part of policy network loss)
        # A(s_t, a_t) = c_t + γ * V(s_{t+1}) - V(s_t)
        # TD(0) is used here
        advantage = costs + self.gamma * new_V - old_V
        return advantage


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

    def _plot_heatmap(self, best_epoch):
        # Heatmap
        if self.heatmap_discretizer is not None:
            heatmap_ver = "initial"
            if best_epoch != 0:
                model_link = os.path.join(self.out_path, f"{best_epoch}-policy.pt")
                self.behavioral_policy.load_state_dict(torch.load(model_link))
                heatmap_ver = "final"
            _, heatmap_entropy, heatmap_cost, heatmap_image = \
                self.get_heatmap()
            heatmap_image.savefig(f"{self.out_path}/{heatmap_ver}_heatmap.png")
            plt.close(heatmap_image)    
        else:
            heatmap_entropy = None
            heatmap_cost = None
            heatmap_image = None   
        return heatmap_entropy, heatmap_cost, heatmap_image


    # Utilities
    def compare_heatmap(self, model_init, model_final):    
        """
        T: Number of trajectories/episodes
        N: Number of time steps
        """
        self._initialize_device()
        self._initialize_envs()

        model_lst = [model_init, model_final]
        title_lst = ["Initial", "Final"]

        for model_link, title_txt in zip(model_lst, title_lst):
            # Create a behavioral, a target policy and a tmp policy used to save valid target policies
            # (those with kl <= kl_threshold) during off policy opt
            first_layer_neuron = 400 if self.env_id == "SafetyPointGoal1-v0" else 300
            second_layer_neuron = 300            
            self.behavioral_policy = PolicyNetwork(self.state_dim, self.action_dim, first_layer_neuron, second_layer_neuron, self.state_dependent_std, self.is_discrete, self.device)  # Recreate the model architecture
            self.behavioral_policy.load_state_dict(torch.load(model_link))
            self.behavioral_policy.to(self.device)  # Move model to the correct device (CPU/GPU)
            self.behavioral_policy.eval()  # Set to evaluation mode (disables dropout, batch norm, etc.)
            title = f"Heatmap of {title_txt} Epoch State Exploration"

            # Heatmap
            _, heatmap_entropy, heatmap_cost, heatmap_image = \
                self.get_heatmap(title)
            print(f"\nHeatmap Entropy: {np.round(heatmap_entropy, 3)}, Heatmap Cost: {np.round(heatmap_cost, 3)}")
            
            heatmap_image.savefig(f"./{title_txt}_heatmap.png")
            plt.close(heatmap_image)

        if isinstance(self.envs, gymnasium.vector.VectorEnv) or isinstance(self.envs, safety_gymnasium.vector.VectorEnv):
            self.envs.close()