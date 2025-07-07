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
from src.value import ValueNetwork
from src.vae import VAE, Encoder, Decoder, Prior

int_choice = torch.int64
float_choice = torch.float64
torch.set_default_dtype(float_choice)

class RENYI:
    def __init__(self, env_id, episode_nr, step_nr, heatmap_cmap, heatmap_interp,
                 parallel_envs=8, int_type=int_choice, float_type=float_choice, omega=3000,
                 k=1, alpha=1, zeta=0, delta=1, max_off_iters=1, use_backtracking=True, backtrack_coeff=0, max_backtrack_try=0, 
                 eps=1e-5, d=1, eta=1e-4, epsilon=0.2, lambda_vae=1e-3, lambda_entropy_value=1e-3, 
                 lambda_cost_value=1e-3, lambda_policy=1e-3, lambda_omega=1e-3, epoch_nr=500, 
                 out_path="", use_behavioral=False, state_dependent_std=False):    
        """
        T: Number of trajectories/episodes
        N: Number of time steps
        delta: Trust-region threshold (Maximum KL Divergence between two avg state density distributions)
        omega (see below): Safety weight/Lagrange multiplier (0 or larger)
        lambda: Learning rate
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
        self.max_off_iters = max_off_iters
        self.use_backtracking = use_backtracking
        self.backtrack_coeff = backtrack_coeff
        self.max_backtrack_try = max_backtrack_try
        self.eps = eps
        self.d = d
        self.lambda_vae = lambda_vae
        self.lambda_entropy_value = lambda_entropy_value
        self.lambda_cost_value = lambda_cost_value
        self.lambda_policy = lambda_policy
        self.lambda_omega = lambda_omega
        self.episode_nr = episode_nr
        self.step_nr = step_nr        
        self.epoch_nr = epoch_nr
        self.gamma = 0.99
        self.omega = omega
        self.eta = eta
        self.epsilon = epsilon
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

        # == Neural Networks ==
        self.behavioral_policy = None
        self.target_policy = None
        self.entropy_value_nn = None
        self.cost_value_nn = None
        self.vae = None
        self.policy_optimizer = None
        self.entropy_value_optimizer = None
        self.cost_value_optimizer = None
        self.vae_optimizer = None

        # == Heatmap Settings ==
        self.heatmap_cmap = heatmap_cmap
        self.heatmap_interp = heatmap_interp
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
        stds_or_probs = np.zeros((self.episode_nr, self.step_nr, self.action_dim), dtype=np.float32)

        print("\nCollect particles")
        num_loops = self.episode_nr // self.parallel_envs
        for loop_idx in range(num_loops):
            # Reset the environments for this batch
            s, _ = self.envs.reset(seed = epoch * self.episode_nr + loop_idx * self.parallel_envs)

            loop_offset = loop_idx * self.parallel_envs  # where to store this batch
            for t in tqdm(range(self.step_nr)):
                # Sample action from policy
                if behavioral:
                    a, std_or_prob = self.behavioral_policy.predict(s)
                else:
                    a, std_or_prob = self.target_policy.predict(s)
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

            # Final state
            states[loop_offset: loop_offset + self.parallel_envs, self.step_nr] = s

        # Convert to PyTorch tensors
        states = torch.tensor(states, dtype=self.float_type, device=self.device)
        if self.is_discrete:
            actions = torch.tensor(actions, dtype=self.int_type, device=self.device)
        else:
            actions = torch.tensor(actions, dtype=self.float_type, device=self.device)
        costs = torch.tensor(costs, dtype=self.float_type, device=self.device)
        next_states = torch.tensor(next_states, dtype=self.float_type, device=self.device)
        stds_or_probs = torch.tensor(stds_or_probs, dtype=self.float_type, device=self.device)

        return states, actions, costs, next_states, stds_or_probs
    
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

        if title is not None:
            plt.title(title)

        return average_state_dist, average_entropy, image_fig
    

    def _one_hot(self, a, scaling=5):
        oh = nn.functional.one_hot(a, num_classes=self.action_dim).to(self.float_type)
        return scaling * oh

    # Optimization and Evaluation
    def train_vae(self, states, actions, train=True):
        # One-hot encode the actions if they are discrete
        if self.is_discrete:
            act_embed = self._one_hot(actions)
        else:
            act_embed = actions
        
        # Stack state and action
        sa = torch.cat([states[:, : -1, :], act_embed], dim=-1)        
        sa_flat = sa.reshape(-1, self.state_dim + self.action_dim)

        # VAE training
        batch_size = 256
        dataset = torch.utils.data.TensorDataset(sa_flat)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print("\nTraining VAE...")
        vae_loss = 0
        for batch in dataloader:
            x_batch = batch[0]
            loss = self.vae(x_batch)

            if train:
                self.vae_optimizer.zero_grad()
                loss.backward()
                self.vae_optimizer.step()

            vae_loss += loss.item()

        return vae_loss
        
    def compute_importance_weights(self, behavioral_policy, target_policy, states, actions):
        if self.is_discrete:
            actions = actions.unsqueeze(-1)

        # Log probabilities under both policies
        logp = target_policy.get_log_p(states[:, : -1], actions)
        old_logp = behavioral_policy.get_log_p(states[:, : -1], actions)

        # Compute importance ratio
        importance_weights = torch.exp(logp - old_logp)

        if self.is_discrete:
            importance_weights = importance_weights.view(-1)        
        
        return importance_weights

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

    def compute_entropy_reward(self, states, actions):
        print("\nCompute entropy using VAE")
        
        # One-hot encode the actions if they are discrete
        if self.is_discrete:
            act_embed = self._one_hot(actions)
        else:
            act_embed = actions
        
        # Stack state and action
        sa = torch.cat([states[:, : -1, :], act_embed], dim=-1)        
        sa_flat = sa.reshape(-1, self.state_dim + self.action_dim)

        # Compute log-likelihood under VAE decoder
        with torch.inference_mode():
            z = self.vae.encoder.sample(sa_flat)
            log_probs = self.vae.decoder.log_prob(x=sa_flat, z=z)  # shape: (B*T,)

            # Behavioral Entropy
            if self.use_behavioral:
                beta = np.exp((1 - self.alpha) * np.log(np.log(self.state_dim)))
                entropy_reward = beta * torch.exp(-beta * ((-log_probs) ** self.alpha)) * ((-log_probs) ** self.alpha)
                # # Behavioral entropy reward using distances between latent codes
                # dists = torch.cdist(z, z, p=2)  # pairwise distances (B*T, B*T)
                # sorted_dists, _ = torch.sort(dists, dim=1)
                # Rk = sorted_dists[:, self.k]  # distance to k-th nearest neighbor
                # beta = np.exp((1 - self.alpha) * np.log(np.log(z.shape[1])))
                # log_term = torch.log(Rk + self.eps)
                # entropy_reward = Rk * torch.exp(-beta * (log_term ** self.alpha)) * (log_term ** self.alpha)
                # entropy_reward = entropy_reward.view(self.episode_nr, self.step_nr, 1)
            else:
                # Convert to negative log-prob (entropy reward)
                if self.zeta == 1:
                    entropy_reward = -log_probs
                else:
                    # Convert to estimated density: p(x) ≈ exp(log p(x | z))
                    entropy_reward = torch.pow(torch.exp(log_probs) + self.eps, self.zeta - 1)
            entropy_reward = entropy_reward.view(self.episode_nr, self.step_nr, 1)
        
        return entropy_reward

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

    def compute_entropy_advantage(self, states, rewards, update = False):
        # TD(lambda) is used here
        lam = 0.95
        V = self.entropy_value_nn(states)  # shape: (batch_size, T+1, 1)

        old_V = V[:, : -1, :]
        new_V = V[:, 1:, :]

        # TD residuals (deltas)
        deltas = rewards + self.gamma * new_V - old_V  # shape: (batch_size, T)
        deltas = deltas.squeeze(-1)

        # GAE advantage computation via backward recursion
        advantage = torch.zeros_like(deltas)
        gae = torch.zeros(deltas.shape[0], device=self.device)

        for t in reversed(range(deltas.shape[1])):
            gae = deltas[:, t] + self.gamma * lam * gae
            advantage[:, t] = gae

        if update:
            # Create value targets: A_t + V(s_t)
            target = advantage.unsqueeze(-1).detach() + old_V  # stop gradient through advantage
            value_loss = nn.functional.mse_loss(old_V, target)
            return value_loss

        # Otherwise, return advantage for policy update
        # Flatten for expected shape: (batch_size * T,)
        return advantage.view(-1)

    def compute_policy_entropy(self, stds_or_probs):    
        if self.is_discrete:
            # Probs are used here
            entropy = -(stds_or_probs * torch.log(stds_or_probs + 1e-8)).sum(dim=-1)
        else:
            # Stds are used here
            entropy = 0.5 * (1.0 + np.log(2 * np.pi)) * self.action_dim \
                    + torch.log(stds_or_probs + 1e-8).sum(dim=-1)
        return entropy.mean()


    # Logging
    def log_epoch_statistics(self, log_file, csv_file_1, csv_file_2, epoch, vae_loss, entropy_value_loss, 
                             cost_value_loss, policy_loss, safety_weight, value_lr, entropy_advantage,
                             cost_advantage, policy_entropy, mean_cost, std_cost, num_off_iters, execution_time,
                             heatmap_image, heatmap_entropy, backtrack_iters, backtrack_lr):
        # Prepare tabulate table
        table = []
        fancy_float = lambda f : f"{f:.3f}"
        table.extend([
            ["Epoch", epoch],
            ["Execution time (s)", fancy_float(execution_time)],
            ["VAE loss", fancy_float(vae_loss)],
            ["Entropy Value loss", fancy_float(entropy_value_loss)],
            ["Cost Value loss", fancy_float(cost_value_loss)],
            ["Policy loss", fancy_float(policy_loss)],
            ["Entropy Advantage", fancy_float(entropy_advantage)],
            ["Cost Advantage", fancy_float(cost_advantage)],
            ["Policy Entropy", fancy_float(policy_entropy)],
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
        csv_file_1.write(f"{epoch},{vae_loss},{entropy_value_loss},{cost_value_loss},{policy_loss},{safety_weight},{value_lr},{entropy_advantage},{cost_advantage},{policy_entropy},{mean_cost},{std_cost},{num_off_iters},{execution_time}\n")
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

    def _initialize_networks(self):
        # Initialize policy neural network
        # Create a behavioral policy and a target policy
        # (those with kl <= kl_threshold) during off policy opt
        self.behavioral_policy = self.create_policy(is_behavioral=True)
        self.target_policy = self.create_policy()
        self.target_policy.load_state_dict(self.behavioral_policy.state_dict())

        # Initialize entropy value neural network
        self.entropy_value_nn = ValueNetwork(self.state_dim, self.device).to(self.device)
        self.entropy_value_optimizer = torch.optim.Adam(self.entropy_value_nn.parameters(), 
                                                        lr = self.lambda_entropy_value) 

        # Initialize cost value neural network
        self.cost_value_nn = ValueNetwork(self.state_dim, self.device).to(self.device)
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value_nn.parameters(), lr = self.lambda_cost_value)       

        # Set optimizer for policy
        self.policy_optimizer = torch.optim.Adam(self.target_policy.parameters(), lr = self.lambda_policy)

        # Initialize VAE neural network
        D = self.state_dim + self.action_dim
        L = 32  # latent dimension
        M = 256  # hidden dimension size
        num_components = 16  # for Mixture prior
        encoder_net = nn.Sequential(
            nn.Linear(D, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, 2 * L)
        )
        encoder = Encoder(encoder_net)
        decoder_net = nn.Sequential(
            nn.Linear(L, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, 2 * D)
        )
        decoder = Decoder(decoder_net)
        prior = Prior(L=L, num_components=num_components)
        self.vae = VAE(encoder, decoder, prior, L=L)
        self.vae.encoder.to(self.device)
        self.vae.decoder.to(self.device)
        self.vae.prior.to(self.device)        
        self.vae.to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.lambda_vae)           

    def _initialize_logging(self):
        # Create log files
        log_file = open(os.path.join((self.out_path), 'log_file.txt'), 'a', encoding="utf-8")

        csv_file_1 = open(os.path.join(self.out_path, f"{self.env_id}.csv"), 'w')
        csv_file_1.write(",".join(['epoch', 'vae_loss', 'entropy_value_loss', 'cost_value_loss', 'policy_loss', 'safety_weight', 'value_learning_rate', 'entropy_advantage', 'cost_advantage', 'policy_entropy', 'mean_cost', 'std_cost', 'num_off_iters','execution_time']))
        csv_file_1.write("\n")

        if self.heatmap_discretizer is not None:
            csv_file_2 = open(os.path.join(self.out_path, f"{self.env_id}-heatmap.csv"), 'w')
            csv_file_2.write(",".join(['epoch', 'average_entropy']))
            csv_file_2.write("\n")
        else:
            csv_file_2 = None

        return log_file, csv_file_1, csv_file_2

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
        states, actions, costs, next_states, stds_or_probs = self.collect_particles(0)

        with torch.inference_mode():
            vae_loss = self.train_vae(states, actions, False)
            entropy_reward = self.compute_entropy_reward(states, actions)  
            entropy_value_loss = self.compute_entropy_advantage(states, entropy_reward, True)
            cost_value_loss = self.compute_cost_advantage(states, costs, True)
            
            importance_weights = self.compute_importance_weights(self.behavioral_policy, self.behavioral_policy, states, actions)
            entropy_advantage = self.compute_entropy_advantage(states, entropy_reward)
            # Clip the advantage as per PPO
            clipped_adv = torch.where(
                entropy_advantage >= 0,
                (1 + self.epsilon) * entropy_advantage,
                (1 - self.epsilon) * entropy_advantage
            )
            # PPO loss: negative of min (r * A, clipped A)
            entropy_advantage_loss = torch.mean(torch.min(importance_weights * entropy_advantage, clipped_adv))

            cost_advantage = self.compute_cost_advantage(states, costs)
            cost_advantage_loss = torch.mean(cost_advantage.view(-1) * importance_weights)

            policy_entropy = self.compute_policy_entropy(stds_or_probs)

            policy_loss = -entropy_advantage_loss + self.omega * cost_advantage_loss - self.eta * policy_entropy

        print("\nEntropy computed")

        execution_time = time.time() - t0
        policy_entropy = policy_entropy.detach().cpu().numpy()
        entropy_value_loss = entropy_value_loss.cpu().numpy()
        entropy_advantage_loss = entropy_advantage_loss.cpu().numpy()
        mean_cost, std_cost = self.compute_discounted_cost(costs)
        cost_value_loss = cost_value_loss.cpu().numpy()
        cost_advantage_loss = cost_advantage_loss.cpu().numpy()
        policy_loss = policy_loss.cpu().numpy()

        # Heatmap
        heatmap_entropy, heatmap_image = self._plot_heatmap(0)

        # Save initial policy
        torch.save(self.vae.state_dict(), os.path.join(self.out_path, "0-vae.pt"))
        torch.save(self.behavioral_policy.state_dict(), os.path.join(self.out_path, "0-policy.pt"))
        torch.save(self.entropy_value_nn.state_dict(), os.path.join(self.out_path, "0-entropy-value.pt"))
        torch.save(self.cost_value_nn.state_dict(), os.path.join(self.out_path, "0-cost-value.pt"))

        safety_weight = self.omega
        value_lr = self.lambda_cost_value

        # Log statistics for the initial policy
        self.log_epoch_statistics(
            log_file=log_file, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
            epoch=0,
            vae_loss=vae_loss,
            entropy_value_loss=entropy_value_loss,
            cost_value_loss=cost_value_loss,
            policy_loss=policy_loss,
            safety_weight=safety_weight,
            value_lr=value_lr,
            entropy_advantage=entropy_advantage_loss,
            cost_advantage=cost_advantage_loss,
            policy_entropy=policy_entropy,
            mean_cost=mean_cost,
            std_cost=std_cost,
            num_off_iters=0,
            execution_time=execution_time,
            heatmap_image=heatmap_image,
            heatmap_entropy=heatmap_entropy,
            backtrack_iters=None,
            backtrack_lr=None
        )

        return vae_loss, entropy_value_loss, cost_value_loss, policy_loss, heatmap_entropy, heatmap_image
    
    def _epoch_train(self, vae_loss, entropy_value_loss, cost_value_loss, policy_loss, log_file, csv_file_1, csv_file_2, heatmap_entropy, heatmap_image):
        if self.use_backtracking:
            original_lr = self.lambda_policy

        best_vae_loss = vae_loss
        best_cost_value_loss = cost_value_loss
        best_entropy_value_loss = entropy_value_loss
        best_loss = policy_loss
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, self.epoch_nr + 1):
            print(f"Epoch {epoch} starts")
            t0 = time.time()

            # Off policy optimization
            num_off_iters = 0

            # Collect particles to optimize off policy
            states, actions, costs, next_states, stds_or_probs = self.collect_particles(epoch)

            if self.use_backtracking:
                self.lambda_policy = original_lr

                for param_group in self.policy_optimizer.param_groups:
                    param_group['lr'] = self.lambda_policy

                backtrack_iter = 1
            else:
                backtrack_iter = None

            # Update VAE
            vae_loss = self.train_vae(states, actions)
            if vae_loss < best_vae_loss:
                torch.save(self.vae.state_dict(), os.path.join(self.out_path, f"{epoch}-vae.pt"))               

            # Update entropy value network
            self.entropy_value_optimizer.zero_grad()
            entropy_reward = self.compute_entropy_reward(states, actions)  
            entropy_value_loss = self.compute_entropy_advantage(states, entropy_reward, True)
            entropy_value_loss.backward()
            self.entropy_value_optimizer.step()
            entropy_value_loss = entropy_value_loss.detach().cpu().numpy()
            if entropy_value_loss < best_entropy_value_loss:
                torch.save(self.entropy_value_nn.state_dict(), os.path.join(self.out_path, f"{epoch}-entropy-value.pt"))            

            # Update safety weight (omega)
            self.omega = max(0, self.omega + self.lambda_omega * (torch.sum(costs) / self.episode_nr - self.d))
            # mean_behavorial_costs, std_behavorial_costs = self.compute_discounted_cost(costs)

            # Update cost value network
            self.cost_value_optimizer.zero_grad()
            cost_value_loss = self.compute_cost_advantage(states, costs, True)            
            cost_value_loss.backward()
            self.cost_value_optimizer.step()
            cost_value_loss = cost_value_loss.detach().cpu().numpy()
            if cost_value_loss < best_cost_value_loss:
                torch.save(self.cost_value_nn.state_dict(), os.path.join(self.out_path, f"{epoch}-cost-value.pt"))

            importance_weights = self.compute_importance_weights(self.behavioral_policy, self.target_policy, states, actions)
            entropy_advantage = self.compute_entropy_advantage(states, entropy_reward)
            # Clip the advantage as per PPO
            clipped_adv = torch.where(
                entropy_advantage >= 0,
                (1 + self.epsilon) * entropy_advantage,
                (1 - self.epsilon) * entropy_advantage
            )
            # PPO loss: negative of min (r * A, clipped A)
            entropy_advantage_loss = torch.mean(torch.min(importance_weights * entropy_advantage, clipped_adv))

            cost_advantage = self.compute_cost_advantage(states, costs)
            cost_advantage_loss = torch.mean(cost_advantage.view(-1) * importance_weights)

            policy_entropy = self.compute_policy_entropy(stds_or_probs)

            # Update policy network
            self.policy_optimizer.zero_grad()
            loss = -entropy_advantage_loss + self.omega * cost_advantage_loss - self.eta * policy_entropy
            loss.backward()
            self.policy_optimizer.step()
            loss = loss.detach().cpu().numpy()
            policy_entropy = policy_entropy.detach().cpu().numpy()
            mean_cost, std_cost = self.compute_discounted_cost(costs)
            execution_time = time.time() - t0

            backtrack_lr = self.lambda_policy
            safety_weight = self.omega
            value_lr = self.lambda_cost_value
            # Log statistics for the initial policy
            self.log_epoch_statistics(
                log_file=log_file, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
                epoch=epoch,
                vae_loss=vae_loss,
                entropy_value_loss=entropy_value_loss,
                cost_value_loss=cost_value_loss,
                policy_loss=policy_loss,
                safety_weight=safety_weight,
                value_lr=value_lr,
                entropy_advantage=entropy_advantage_loss,
                cost_advantage=cost_advantage_loss,
                policy_entropy=policy_entropy,
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

            if epoch // 5 == 0:
                self.behavioral_policy.load_state_dict(self.target_policy.state_dict())

        return best_epoch        

    def train(self):
        self._initialize_device()
        self._initialize_envs()
        self._initialize_networks()
        log_file, csv_file_1, csv_file_2 = self._initialize_logging()

        vae_loss, entropy_value_loss, cost_value_loss, policy_loss, heatmap_entropy, heatmap_image = self._run_initial_evaluation(log_file, csv_file_1, csv_file_2)

        best_epoch = self._epoch_train(vae_loss, entropy_value_loss, cost_value_loss, policy_loss, log_file, csv_file_1, csv_file_2, heatmap_entropy, heatmap_image)                                                                
        heatmap_entropy, heatmap_image = self._plot_heatmap(best_epoch)

        if isinstance(self.envs, gymnasium.vector.VectorEnv):
            self.envs.close()


    # Utilities
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