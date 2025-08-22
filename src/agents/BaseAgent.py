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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import os
from src.discretizer import create_discretizer
from sklearn.manifold import TSNE
import matplotlib.lines as mlines
import pandas as pd
import re
from pathlib import Path

int_choice = torch.int64
float_choice = torch.float64
torch.set_default_dtype(float_choice)
np.random.seed(0)
torch.manual_seed(0)

class BaseAgent:
    def __init__(self, env_id, safety_weight=6000, alpha=0, zeta=1, epoch_nr=500, use_behavioral=False, state_dependent_std=False, out_path=""):    
        """
        k: Number of neighbors
        alpha: Parameter used in Behavioral entropy
        zeta: Parameter used in Rényi entropy
        gamma: Discount factor of cost/entropy reward value over time
        """   
        # == Environment Config ==
        self.env_id = env_id
        self.parallel_envs = 8
        self.out_path = out_path
        self.int_type = int_choice
        self.float_type = float_choice

        # == Algorithm Hyperparameters ==
        self.k = 4
        self.alpha = alpha
        self.zeta = zeta
        self.use_backtracking = True
        self.safety_weight = safety_weight     
        self.cost_value_lr = 1e-3
        self.safety_weight_lr = 1e-2
        if self.env_id == "MountainCarContinuous-v0" or self.env_id == "MountainCar-v0":
            self.step_nr = 400
            self.policy_lr = 1e-4
            self.safety_constraint = 0.5
        elif self.env_id == "CartPole-v1" or self.env_id == "Pendulum-v1":
            self.step_nr = 300
            self.policy_lr = 1e-4
            self.safety_constraint = 5
        else:
            self.step_nr = 500
            self.policy_lr = 1e-5
            self.safety_constraint = 25
        self.epoch_nr = epoch_nr
        self.episode_nr = 24
        self.gamma = 0.99
        self.eps = 1e-8        
        self.patience = 50
        self.use_behavioral = use_behavioral
        self.alg_name = ""
        if use_behavioral:
            self.entropy_name = fr"Behavioral Entropy ($\alpha={self.alpha}$)"
        elif self.zeta == 1:
            self.entropy_name = f"Shannon Entropy"
        else:
            self.entropy_name = fr"Rényi Entropy ($\zeta={self.zeta}$)"
        self.max_off_iters = 30
        self.backtrack_coeff = 2
        self.max_backtrack_try = 10
        if self.env_id == "MountainCarContinuous-v0" or self.env_id == "MountainCar-v0" or "CartPole-v1" or "Pendulum-v1":
            self.trust_region_threshold = 0.5
        else:
            self.trust_region_threshold = 0.1            
            
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
        self.policy_scheduler = None
        self.cost_value_optimizer = None
        self.cost_value_scheduler = None

        # == Heatmap Settings ==
        self.heatmap_cmap = 'Blues'
        self.heatmap_interp = 'spline16'
        self.heatmap_discretizer = None
        if self.env_id == "MountainCarContinuous-v0" or self.env_id == "MountainCar-v0":
            self.heatmap_labels = ('Position', 'Velocity')
        elif self.env_id == "CartPole-v1":
            self.heatmap_labels = ('Pole Angle', 'Cart Position') 
        elif self.env_id == "Pendulum-v1":   
            self.heatmap_labels = ('Cosine Angle', 'Angular Velocity')     
        else:
            self.heatmap_labels = ('X', 'Y')       


    # Environment and Setup
    def create_policy(self):
        first_layer_neuron = 400 if self.env_id == "SafetyPointGoal1-v0" else 300
        second_layer_neuron = 300
        policy = PolicyNetwork(self.state_dim, self.action_dim, first_layer_neuron, second_layer_neuron, self.state_dependent_std, self.is_discrete, self.device).to(self.device)
        return policy
    
    def normalize_states(self, states):
        is_tensor = isinstance(states, torch.Tensor)
        states_np = states.cpu().numpy() if is_tensor else states

        obs_low = self.envs.single_observation_space.low
        obs_high = self.envs.single_observation_space.high

        # Robust check for boundedness
        finite_low = np.isfinite(obs_low)
        finite_high = np.isfinite(obs_high)

        bounded = finite_low & finite_high
        lower_bounded = finite_low & ~finite_high
        upper_bounded = ~finite_low & finite_high
        unbounded = ~finite_low & ~finite_high

        norm_states = np.zeros_like(states_np)

        # Fully bounded: Min-Max scaling to [-1, 1]
        norm_states[..., bounded] = 2 * (states_np[..., bounded] - obs_low[bounded]) / \
            np.maximum(obs_high[bounded] - obs_low[bounded], self.eps) - 1

        # Lower bounded only
        norm_states[..., lower_bounded] = states_np[..., lower_bounded] - obs_low[lower_bounded]

        # Upper bounded only
        norm_states[..., upper_bounded] = obs_high[upper_bounded] - states_np[..., upper_bounded]

        # Unbounded: Standardize
        if unbounded.any():
            mean = np.mean(states_np[..., unbounded], axis=0, keepdims=True)
            std = np.std(states_np[..., unbounded], axis=0, keepdims=True)
            std[std < self.eps] = 1.0  # Avoid division by very small std
            norm_states[..., unbounded] = (states_np[..., unbounded] - mean) / std

        # Handle any remaining NaNs (e.g. from infs or divisions)
        norm_states = np.nan_to_num(norm_states, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(norm_states, dtype=states.dtype, device=states.device) if is_tensor else norm_states

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
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, metric='euclidean', algorithm='auto', n_jobs=self.num_workers)
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
    

    # Optimization and Evaluation
    def compute_importance_weights(self, behavioral_policy, target_policy, states, actions, same_policy=False):
        # Skip computation if the same policies are used
        if same_policy:
            num_samples = states.shape[0] * self.step_nr
            importance_weights = torch.ones(num_samples, dtype=self.float_type, device=self.device)
            importance_weights /= num_samples
            return importance_weights
        
        # Initialize to None for the first concat
        importance_weights = None

        # Compute the importance weights
        # build iw vector incrementally from trajectory particles
        print("\nCompute importance weights")
        for episode in tqdm(range(states.shape[0])):
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
            Rk = distances[:, self.k]
            log_Rk = torch.log(Rk + self.eps)
            term = torch.abs(self.state_dim * log_Rk)

            beta = torch.exp((1 - self.alpha) * torch.log(torch.log(torch.tensor(float(self.state_dim)))))

            entropy = torch.pow(Rk, self.state_dim) * \
                    torch.exp(-beta * torch.pow(term, self.alpha)) * \
                    torch.pow(term, self.alpha)

            importance_weights = 1.0 / (Rk ** self.state_dim + self.eps)
            weighted_entropy = importance_weights * entropy
            entropy_per_episode = weighted_entropy.view(self.episode_nr, self.step_nr)            
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
                log_terms = (self.zeta - 1) * torch.log(density_estimate + self.eps)
                entropy_per_episode = (1 / (1 - self.zeta)) * torch.logsumexp(log_terms, dim=1)

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
        mean_cost = discounted_costs.mean().item()
        std_cost = discounted_costs.std().item()

        return mean_cost, std_cost
    
    def compute_actual_cost(self, costs):  # costs: [N_episodes, T, 1] with ct in {0,1}
        ep_costs = costs.sum(dim=1).squeeze(-1)     # [N_episodes]
        return ep_costs.mean().item(), ep_costs.std(unbiased=False).item()    

    def compute_cost_advantage(self, states, costs):
        """
        TD(0) critic target and advantage for the cost critic.
        Returns:
            adv_flat  : [B*T] tensor (detached, normalized)
            value_loss: scalar MSE loss for the cost value network
        """
        V = self.cost_value_nn(states)
        V_det = V.detach()

        # TD(0) advantage: A_t = c_t + γ V(s_{t+1}) - V(s_t)
        adv = costs + self.gamma * V_det[: , 1: , : ] - V_det[: , : -1, : ]

        # Critic target and loss: y_t = c_t + γ V(s_{t+1})   (no λ)
        target = (costs + self.gamma * V_det[:, 1: , :]).detach()
        pred = V[: , : -1, : ]
        value_loss = torch.nn.functional.mse_loss(pred, target)

        # Flatten & normalize for the policy loss
        adv_flat = adv.view(-1).detach()
        # adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + self.eps)

        return adv_flat, value_loss
    

    # Visualization
    def rarity_from_counts(self, ref_counts):
        ref_counts = np.asarray(ref_counts, dtype=float)
        ref_probs = (ref_counts + 1) / (ref_counts.sum() + ref_counts.size)
        rarity_ref = -np.log(ref_probs)
        return rarity_ref, ref_probs

    def weighted_quantile(self, values, weights, q):
        v = values.ravel(); w = weights.ravel().astype(float)
        m = w > 0; v, w = v[m], w[m]
        order = np.argsort(v); v, w = v[order], w[order]
        cumw = np.cumsum(w); cumw /= cumw[-1]
        q = np.atleast_1d(q)
        return np.interp(q, cumw, v)

    def tail_coverage_curve(self, rarity_ref, ref_probs, meth_counts, q_list=np.linspace(0.05, 0.5, 10)):
        meth_counts = np.asarray(meth_counts, dtype=float)
        meth_probs = meth_counts / (meth_counts.sum() + meth_counts.size)
        # thresholds so that rarest q% under reference have rarity >= t_q
        t = self.weighted_quantile(rarity_ref, ref_probs, 1 - q_list)
        coverage = [(meth_probs[rarity_ref >= tq]).sum() for tq in t]
        return q_list, np.array(coverage)

    def get_simulations(self, best_epoch):
        self._initialize_device()
        self._initialize_envs()   
        _ = self._initialize_networks()        
        self.behavioral_policy.load_state_dict(torch.load(f"{self.out_path}/{best_epoch}-policy.pt"))
        state_dist = self.heatmap_discretizer.get_empty_mat()
        np.random.seed(0)
        torch.manual_seed(0)        
        states, actions, costs, next_states, distances, indices, state_dist = self.collect_particles_and_compute_knn(0, state_dist)
        with torch.inference_mode():
            importance_weights = self.compute_importance_weights(self.behavioral_policy, self.behavioral_policy, states, actions, True)
            mean_entropy, _ = self.compute_entropy(distances, indices, False, 1, importance_weights)   
        mean_entropy = mean_entropy.cpu().numpy()
        mean_cost, _ = self.compute_discounted_cost(costs)
        return states, costs, mean_entropy, mean_cost, state_dist

    def visualize_policy_comparison(self, best_epoch, num_states=1000):
        self._initialize_device()
        self._initialize_envs()   
        before_model = self.create_policy()
        after_model = self.create_policy()
        before_model.load_state_dict(torch.load(f"{self.out_path}/0-policy.pt"))
        after_model.load_state_dict(torch.load(f"{self.out_path}/{best_epoch}-policy.pt"))

        before_model.eval()
        after_model.eval()

        # Sample random states from state space
        low_bounds = self.envs.single_observation_space.low
        high_bounds = self.envs.single_observation_space.high

        np.random.seed(0)
        states = np.random.uniform(low_bounds, high_bounds, size=(num_states, self.state_dim))
        states = torch.tensor(states, dtype=torch.float64, device=self.device)

        with torch.inference_mode():
            _, before_actions = before_model(states, deterministic=True)
            _, after_actions = after_model(states, deterministic=True)

        # Convert to numpy
        before_actions = before_actions.cpu().numpy()
        after_actions = after_actions.cpu().numpy()

        # Plot histograms of action samples
        plt.hist(before_actions, bins=100, alpha=0.6, label='Before Training', density=True)
        plt.hist(after_actions, bins=100, alpha=0.6, label='After Training', density=True)
        plt.title(f"Action Distribution of {self.env_id[: -3]} (Before vs After)")
        plt.xlabel("Action")
        plt.ylabel("Density")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.out_path}/action_comparison.png")

    def visualize_policy_heatmap(self, best_epoch, use_mean=True):
        self._initialize_device()
        self._initialize_envs()        
        before_model = self.create_policy()
        after_model = self.create_policy()
        before_model.load_state_dict(torch.load(f"{self.out_path}/0-policy.pt"))
        after_model.load_state_dict(torch.load(f"{self.out_path}/{best_epoch}-policy.pt"))

        before_model.eval()
        after_model.eval()

        # Sample random states from state space
        low_bounds = self.envs.single_observation_space.low
        high_bounds = self.envs.single_observation_space.high

        if self.env_id == 'CartPole-v1':
            state_idxs = [2, 0] 
        elif self.env_id == 'Pendulum-v1':
            state_idxs = [0, 2]    
        else:    
            state_idxs = [0, 1]

        # Find some discrete states from the state space
        x_lin = np.linspace(low_bounds[state_idxs[0]], high_bounds[state_idxs[0]], 100)
        y_lin = np.linspace(low_bounds[state_idxs[1]], high_bounds[state_idxs[1]], 100)
        grid_x, grid_y = np.meshgrid(x_lin, y_lin)

        np.random.seed(0)
        states = np.random.uniform(low_bounds, high_bounds, size=(grid_x.size, self.state_dim))
     
        # Overwrite the two visualised dims with grid
        states[:, state_idxs[0]] = grid_x.ravel()
        states[:, state_idxs[1]] = grid_y.ravel()
        states = torch.tensor(states, dtype=torch.float64, device=self.device)

        with torch.no_grad():
            before_std, before_action = before_model(states, deterministic=True)
            after_std, after_action = after_model(states, deterministic=True) 

        if self.env_id == "CartPole-v1":
            before_action = before_action.argmax(dim=1, keepdim=True)
            after_action = after_action.argmax(dim=1, keepdim=True)

        if use_mean:
            before = before_action.cpu().numpy().reshape(grid_x.shape)
            after = after_action.cpu().numpy().reshape(grid_x.shape)
            main_txt = "Action Mean"
        else:
            if self.state_dependent_std:
                before = before_std.cpu().numpy().reshape(grid_x.shape)
                after = after_std.cpu().numpy().reshape(grid_x.shape)
            else:
                before = np.full(grid_x.shape, before_std.cpu().numpy())
                after = np.full(grid_x.shape, after_std.cpu().numpy())                  
            main_txt = "Action Standard Deviaiton"

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        titles = ["Before Training", "After Training"]
        vmin, vmax = np.min(before), np.max(after)
        datas = [before, after]  

        for ax, data, title in zip(axes, datas, titles):
            im = ax.imshow(
                data,
                origin="lower",
                extent=[x_lin[0], x_lin[-1], y_lin[0], y_lin[-1]],
                aspect="auto",
                vmin=vmin, vmax=vmax,
                cmap="viridis"
            )
            ax.set_title(f"{main_txt} ({title})")
            ax.set_xlabel(self.heatmap_labels[0])
            ax.set_ylabel(self.heatmap_labels[1])       
                 
            fig.colorbar(im, ax=ax, shrink=0.8)

            if self.env_id == "Pendulum-v1":
                safety_threshold_1, safety_threshold_2 = self.envs.get_safety_threshold()
                ax.axvline(x=safety_threshold_1, color='red', linestyle='--', linewidth=10)
                ax.axvline(y=-safety_threshold_2, color='red', linestyle='--', linewidth=10)  
                ax.axvline(y=safety_threshold_2, color='red', linestyle='--', linewidth=10)
            elif self.env_id == "MountainCarContinuous-v0":
                safety_threshold = self.envs.get_safety_threshold()
                ax.axvline(x=safety_threshold, color='red', linestyle='--', linewidth=10)   
            elif self.env_id == "CartPole-v1":
                safety_threshold = self.envs.get_safety_threshold()
                ax.axhline(y=-safety_threshold, color='red', linestyle='--', linewidth=10)
                ax.axhline(y=safety_threshold, color='red', linestyle='--', linewidth=10)                                

        plt.tight_layout()
        plt.savefig(f"{self.out_path}/action_{'mean' if use_mean else 'std'}_heatmap.png")

    def best_epoch_vis(self):    
        df_heatmap = pd.read_csv(f"{self.out_path}/best_epochs.csv")
        fig, ax = plt.subplots(figsize=(7,4))
        palette = {"Yes (Dangerous)": "#d62728", "No (Safe)": "#1f77b4"}      
        sns.scatterplot(data=df_heatmap, x="best_epoch", y="heatmap_entropy", hue="exceed_threshold", hue_order=["Yes (Dangerous)", "No (Safe)"], palette=palette, ax=ax)
        ax.axhline(df_heatmap.loc[df_heatmap["best_epoch"] == 0, "heatmap_entropy"].iloc[0], ls="--", color="grey", label="Initial Heatmap Entropy")
        ax.set_xlabel("Best Epoch")
        ax.set_ylabel("Heatmap Entropy")
        ax.legend(title="Exceed Safety Threshold")
        ax.set_title(f"{self.env_id[: -3]}: Change in Heatmap Entropy\nover Best Epochs with New Minimum Policy Loss\nUsing {self.alg_name} Algorithm and {self.entropy_name}")
        fig.tight_layout()
        entropy_file_name = (self.entropy_name
                                .lower()
                                .replace("(", "")
                                .replace(")", "")
                                .replace("$", "")
                                .replace("\\", "")
                                .replace("", "")
                                .replace("=", " ")
                                .replace(" entropy", "")
                                .replace(" ", "_"))           
        fig.savefig(f"{self.out_path}/epoch_heatmap_entropy_{self.alg_name.lower()}_{entropy_file_name}_{self.env_id}.png")
        plt.close('all') 
        true_best_epoch = df_heatmap.loc[df_heatmap["heatmap_entropy"].idxmax()]["best_epoch"]
        if self.alg_name == "CEM":
            self.cem_vis(true_best_epoch)
        elif self.alg_name == "RENYI":
            self.renyi_vis(true_best_epoch)

    def compute_best_epochs(self):    
        best_epochs = sorted(
            {int(m.group(1))
            for p in Path(self.out_path).iterdir()
            if (m := re.match(r'^(\d+)-policy\.pt$', p.name))}
        ) 
        init_states, init_costs, init_entropy, init_mean_cost, init_state_dict = self.get_simulations(0)
        entropy_lst = []
        cost_lst = []

        for best_epoch in best_epochs:
            final_states, final_costs, final_entropy, final_mean_cost, final_state_dict = self.get_simulations(best_epoch)  
            self._plot_rarity(best_epoch, init_state_dict, final_state_dict)
            self._plot_heatmap(final_entropy, final_mean_cost, final_state_dict, best_epoch)
            self._plot_tsne(best_epoch, init_states, init_costs, init_entropy, init_mean_cost, final_states, final_costs, final_entropy, final_mean_cost) 

            if best_epoch == 0:
                entropy_lst.append(init_entropy)
                cost_lst.append(init_mean_cost)        
            else:
                entropy_lst.append(final_entropy)
                cost_lst.append(final_mean_cost)   

            if hasattr(self, "envs") and self.envs is not None:
                try:
                    self.envs.close()
                except Exception as e:
                    print(f"Failed to close envs: {e}")
                self.envs = None

        df_heatmap = pd.DataFrame({"best_epoch": best_epochs, "heatmap_entropy": entropy_lst, "heatmap_cost": cost_lst})  
        df_heatmap["exceed_threshold"] = np.where(df_heatmap["heatmap_cost"] > self.safety_constraint, "Yes (Dangerous)", "No (Safe)")  
        df_heatmap.to_csv(f"{self.out_path}/best_epochs.csv", index=False) 

        self.best_epoch_vis()            


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

    def _plot_heatmap(self, mean_entropy, mean_cost, state_dist, best_epoch=0):
        # Heatmap
        if self.heatmap_discretizer is not None:
            heatmap_ver = "initial" if best_epoch == 0 else "final"
            title = f"{heatmap_ver.title()} States Exploration Heatmap of {self.env_id[: -3]} \nUsing {self.alg_name} Algorithm and {self.entropy_name}"
            
            # Plot heatmap
            plt.close()
            fig, ax = plt.subplots()

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(self.heatmap_labels[0])
            ax.set_ylabel(self.heatmap_labels[1])

            if len(state_dist.shape) == 2:
                log_p = np.ma.log(state_dist)
                flat = log_p.ravel()
                m1 = flat.min()
                flat[flat == m1] = flat[flat != m1].min()
                im = ax.imshow(log_p.filled(m1).T,
                               origin='lower',
                               interpolation=self.heatmap_interp,
                               cmap=self.heatmap_cmap)
            else:
                ax.bar(range(self.heatmap_discretizer.bins_sizes[0]), state_dist)

            if self.env_id == 'CartPole-v1':
                safety = self.envs.get_safety_threshold()
                y_edges = self.heatmap_discretizer.bins[1]        
                y_lo = np.searchsorted(y_edges, -safety)
                y_hi = np.searchsorted(y_edges, safety)
                y0, y1 = ax.get_ylim()
                ax.axhspan(y_lo, y0, color='red', alpha=0.3, label='Unsafe Region')
                ax.axhspan(y1, y_hi, color='red', alpha=0.3)
            elif self.env_id == 'Pendulum-v1':
                safety_1, safety_2 = self.envs.get_safety_threshold()
                x_edges = self.heatmap_discretizer.bins[0]
                y_edges = self.heatmap_discretizer.bins[1]
                x_idx = np.searchsorted(x_edges, safety_1)
                y_lo = np.searchsorted(y_edges, -safety_2)
                y_hi = np.searchsorted(y_edges, safety_2)
                x0, _= ax.get_xlim()
                y0, y1 = ax.get_ylim()     
                ax.add_patch(Rectangle((x0, y_lo), x_idx - x0, y1 - y_lo,
                                    facecolor='red', alpha=0.3, label='Unsafe Region'))
                ax.add_patch(Rectangle((x0, y0), x_idx - x0, y_hi - y0,
                                    facecolor='red', alpha=0.3))
            else:
                safety = self.envs.get_safety_threshold()
                x_edges = self.heatmap_discretizer.bins[0]
                x_idx = np.searchsorted(x_edges, safety)
                x0, _ = ax.get_xlim()
                ax.axvspan(x0, x_idx, color='red', alpha=0.3, label='Unsafe Region')

            ax.legend(frameon=False, loc='best')
            ax.set_title(f"{title}\nCoverage Entropy: {mean_entropy:.3f}, Expected Cost: {mean_cost:.3f}")
            plt.tight_layout()
                      
            entropy_file_name = (self.entropy_name
                                 .lower()
                                 .replace("(", "")
                                 .replace(")", "")
                                 .replace("$", "")
                                 .replace("\\", "")
                                 .replace("", "")
                                 .replace("=", " ")
                                 .replace(" entropy", "")
                                 .replace(" ", "_"))
            fig.savefig(f"{self.out_path}/{best_epoch}_heatmap_{self.alg_name.lower()}_{entropy_file_name}_{self.env_id}.png")
            plt.close(fig)    

    def _plot_tsne(self, best_epoch, init_states, init_costs, init_entropy, init_mean_cost, final_states, final_costs, final_entropy, final_mean_cost):
        norm_states = torch.cat((init_states, final_states), 0)
        norm_states = norm_states.reshape(-1, norm_states.shape[2])
        zero_col = torch.zeros(self.episode_nr, 1, 1, device=init_states.device, dtype=init_states.dtype)
        init_costs = torch.cat((zero_col, init_costs), dim=1).reshape(-1)
        final_costs = torch.cat((zero_col, final_costs), dim=1).reshape(-1)
        init_unsafe = (init_costs.cpu().numpy() > 0)
        final_unsafe = (final_costs.cpu().numpy() > 0)
        init_safe = ~init_unsafe
        final_safe = ~final_unsafe     
        norm_states = norm_states.cpu().numpy()

        if self.env_id == "CartPole-v1":
            tsne_states = norm_states[:, [2, 0]]
        elif self.env_id == "MountainCarContinuous-v0":
            tsne_states = norm_states   
        else:
            tsne = TSNE(n_components=2, perplexity=min(30, len(norm_states) // 3), random_state=42)
            tsne_states = tsne.fit_transform(norm_states)

        # Plot heatmap
        plt.close()
        fig, ax = plt.subplots()

        ax.set_xticks([])
        ax.set_yticks([])
        if self.env_id == "CartPole-v1":
            ax.set_xlabel("Pole Angle")
            ax.set_ylabel("Cart Position")
        elif self.env_id == "MountainCarContinuous-v0":
            ax.set_xlabel("Position")
            ax.set_ylabel("Velocity")
        else:      
            ax.set_xlabel("TSNE Feature 1")
            ax.set_ylabel("TSNE Feature 2")                          

        init_points = tsne_states[: tsne_states.shape[0] // 2]
        final_points = tsne_states[tsne_states.shape[0] // 2: ]

        start_idx = [i * init_states.shape[1] for i in range(init_states.shape[0] * 2)]
        start_pts = tsne_states[start_idx]
     
        if self.env_id == "SafetyPointGoal1-v0":
            alpha1 = 0.02
            alpha2 = 0.4
        else:
            alpha1 = 0.1  
            alpha2 = 0.8

        ax.scatter(init_points[init_safe, 0], init_points[init_safe, 1], 
                s=5, c='red', alpha=alpha1, marker='o', label='_nolegend_')
        ax.scatter(init_points[init_unsafe, 0], init_points[init_unsafe, 1], 
                s=5, c='red', alpha=alpha2, marker='^', label='_nolegend_')
        ax.scatter(final_points[final_safe, 0], final_points[final_safe, 1], 
                s=5, c='blue', alpha=alpha1, marker='o', label='_nolegend_')
        ax.scatter(final_points[final_unsafe, 0], final_points[final_unsafe, 1], 
                s=5, c='blue', alpha=alpha2, marker='^', label='_nolegend_')              
        ax.scatter(start_pts[: , 0], start_pts[: , 1], s=10, color='black', alpha=0.5, marker='x', label='_nolegend_')
       
        legend_init_safe = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                    markersize=5, alpha=0.25, label='Initial Model (Safe States)')
        legend_init_unsafe = mlines.Line2D([], [], color='red', marker='^', linestyle='None',
                                    markersize=5, alpha=1, label='Initial Model (Unsafe States)')
        legend_final_safe = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                                    markersize=5, alpha=0.25, label='Final Model (Safe States)')
        legend_final_unsafe = mlines.Line2D([], [], color='blue', marker='^', linestyle='None',
                                    markersize=5, alpha=1, label='Final Model (Unsafe States)')        
        legend_start = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                                    markersize=5, alpha=1, label='Starting States')

        # Add legend with custom handles
        ax.legend(handles=[legend_init_safe, legend_init_unsafe, legend_final_safe, legend_final_unsafe, legend_start], loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)        

        if self.env_id == "SafetyPointGoal1-v0":
            title = f"States Exploration TSNE Plot of {self.env_id[: -3]} \nUsing {self.alg_name} Algorithm and {self.entropy_name}\nInitial Average Entropy: {np.round(init_entropy, 3)}, Initial Average Total Cost: {np.round(init_mean_cost, 3)}\nFinal Average Entropy: {np.round(final_entropy, 3)}, Final Average Total Cost: {np.round(final_mean_cost, 3)}"
        else:
            title = f"States Exploration Plot of {self.env_id[: -3]} \nUsing {self.alg_name} Algorithm and {self.entropy_name}\nInitial Average Entropy: {np.round(init_entropy, 3)}, Initial Average Total Cost: {np.round(init_mean_cost, 3)}\nFinal Average Entropy: {np.round(final_entropy, 3)}, Final Average Total Cost: {np.round(final_mean_cost, 3)}"

        ax.set_title(title)
        plt.tight_layout()

        entropy_file_name = (self.entropy_name
                                .lower()
                                .replace("(", "")
                                .replace(")", "")
                                .replace("$", "")
                                .replace("\\", "")
                                .replace("", "")
                                .replace("=", " ")
                                .replace(" entropy", "")
                                .replace(" ", "_"))
        fig.savefig(f"{self.out_path}/{best_epoch}_tsne_{self.alg_name.lower()}_{entropy_file_name}_{self.env_id}.png", bbox_inches='tight')
        plt.close(fig)

    def _plot_rarity(self, best_epoch, ref_counts, meth_counts):
        rarity_ref, ref_probs = self.rarity_from_counts(ref_counts)
        q, coverage = self.tail_coverage_curve(rarity_ref, ref_probs, meth_counts, q_list=np.r_[0.0, np.linspace(0.01, 0.5, 25), np.linspace(0.6, 0.99, 20), 1.0])
        auc = np.trapz(coverage, q)

        plt.close()      
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))

        im = ax[0].imshow(rarity_ref.T, origin='lower', cmap='magma')
        ax[0].set_title("Rarity Map ($−\log p_{ref}$)")
        ax[0].set_xlabel("Bin $x$"); ax[0].set_ylabel("Bin $y$")
        cbar = plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
        cbar.set_label("Rarity (Higher = Rarer)")
        handles, labels = ax[0].get_legend_handles_labels()

        for alt_q, per_q, color, linestyle, in [(0.1, 10, 'blue', '--')]:
            tq = self.weighted_quantile(rarity_ref, ref_probs, 1 - alt_q).item()
            cs = ax[0].contour((rarity_ref >= tq).T.astype(int), levels=[0.5], colors=color, linewidths=1.5, linestyles=linestyle)
            h = mlines.Line2D([0],[0], color=color, lw=1.6, ls=linestyle, label=f"Rarest {per_q}%")          
            handles += [h]
            labels += [h.get_label()]            
        ax[0].legend(handles, labels, loc='upper left', frameon=True, facecolor='white')

        # (B) Tail-coverage curve for the method
        ax[1].plot(q * 100, coverage, marker='o')
        ax[1].set_xlabel("Rarest q% of States (w.r.t. Baseline)")
        ax[1].set_ylabel("Method Mass in Tail (↑)")
        ax[1].set_title(f"Tail Coverage of {self.env_id[: -3]} \nUsing {self.alg_name} Algorithm and {self.entropy_name}\nAUC: {np.round(auc * 100, 3)}%")
        ax[1].grid(alpha=0.3)

        plt.tight_layout()

        entropy_file_name = (self.entropy_name
                                .lower()
                                .replace("(", "")
                                .replace(")", "")
                                .replace("$", "")
                                .replace("\\", "")
                                .replace("", "")
                                .replace("=", " ")
                                .replace(" entropy", "")
                                .replace(" ", "_"))
        fig.savefig(f"{self.out_path}/{best_epoch}_rarity_{self.alg_name.lower()}_{entropy_file_name}_{self.env_id}.png", bbox_inches='tight')
        plt.close(fig)           