import torch
import torch.nn as nn
import numpy as np
import gymnasium
import safety_gymnasium
import time
import os
from src.value import ValueNetwork
from src.vae import VAE, Encoder, Decoder, Prior, log_normal_diag
from src.agents.BaseAgent import BaseAgent
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

int_choice = torch.int64
float_choice = torch.float64
torch.set_default_dtype(float_choice)
np.random.seed(0)
torch.manual_seed(0)

class RENYI(BaseAgent):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs) 

        # == Algorithm Hyperparameters ==
        if self.env_id == "MountainCarContinuous-v0" or self.env_id == "MountainCar-v0" or self.env_id == "CartPole-v1" or self.env_id == "Pendulum-v1":
            self.vae_lr = 1e-3
        else:
            self.vae_lr = 1e-4    
        self.entropy_value_lr = 1e-3
        self.eta = 1
        self.epsilon = 0.2
        self.alg_name = "RENYI"

        # == Neural Networks ==
        self.vae = None
        self.entropy_value_nn = None
        self.vae_optimizer = None
        self.vae_scheduler = None
        self.entropy_value_optimizer = None
        self.entropy_value_scheduler = None        
    

    # Optimization and Evaluation
    # def form_states_action_pairs(self, states, actions):
    #     # One-hot encode the actions if they are discrete
    #     if self.is_discrete:
    #         act_embed = nn.functional.one_hot(actions, num_classes=self.action_dim).to(self.float_type)
    #     else:
    #         act_embed = actions
    #         # Pendulum actions are in range (-2, 2) and need min-max normalization
    #         # if self.env_id == "Pendulum-v0":
    #         #     act_embed /= 2

    #     # Stack state and action
    #     # states_embed = self.normalize_states(states[:, : -1, :])
    #     states_embed = states[:, : -1, :]
    #     sa = torch.cat([states_embed, act_embed], dim=-1)        
    #     sa_flat = sa.reshape(-1, self.state_dim + self.action_dim)

    #     return sa_flat            

    def train_vae(self, next_states, train=True):
        reshaped_next_states = next_states.reshape(-1, self.state_dim)

        # VAE training
        print("\nTraining VAE...")
        loss, RE, KL = self.vae(reshaped_next_states)
        
        mu_e, log_var_e = self.vae.encoder.encode(reshaped_next_states)
        z = self.vae.encoder.sample(x=reshaped_next_states, mu_e=mu_e, log_var_e=log_var_e)

        # ELBO
        if train:
            self.vae_optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=5.0) 
            self.vae_optimizer.step()

        vae_loss = loss.item()
        vae_RE = RE.item()
        vae_KL = KL.item()

        return vae_loss, vae_RE, vae_KL

    def compute_entropy_reward(self, next_states):
        print("\nCompute entropy using VAE")
        reshaped_next_states = next_states.reshape(-1, self.state_dim)

        # Compute log-likelihood under VAE decoder
        with torch.inference_mode():
            mu_e, log_var_e = self.vae.encoder.encode(reshaped_next_states)
            z = self.vae.encoder.reparameterization(mu_e, log_var_e)
            log_px_z = self.vae.decoder.log_prob(reshaped_next_states, z)              # log p(x|z)
            log_pz = torch.sum(self.vae.prior.log_prob(z), dim=-1)  # log p(z)
            log_qz_x = torch.sum(log_normal_diag(z, mu_e, log_var_e), dim=-1)  # log q(z|x)
            log_p_x = log_px_z + log_pz - log_qz_x

            # z = self.vae.encoder.sample(reshaped_next_states)
            # log_probs = self.vae.decoder.log_prob(x=reshaped_next_states, z=z)  # shape: (B*T,)

            # Behavioral Entropy
            if self.use_behavioral:
                x = torch.clamp(-log_p_x, min=1e-8)
                beta = np.exp((1 - self.alpha) * np.log(np.log(self.state_dim)))
                entropy_reward = beta * torch.exp(-beta * (x ** self.alpha)) * (x ** self.alpha)
            else:
                # Convert to negative log-prob (entropy reward)
                if self.zeta == 1:
                    entropy_reward = -log_p_x
                else:
                    # Convert to estimated density: p(x) ≈ exp(log p(x | z))
                    entropy_reward = torch.pow(torch.exp(log_p_x) + self.eps, self.zeta - 1)
            # print("Entropy Reward: ", entropy_reward.mean().item(), entropy_reward.std().item(), entropy_reward.max().item(), entropy_reward.min().item())
            # Normalize reward before usage
            entropy_reward = (entropy_reward - entropy_reward.mean()) / (entropy_reward.std() + self.eps)
            entropy_reward = entropy_reward.view(next_states.shape[0], self.step_nr, 1)
        
        entropy_reward = entropy_reward.clone()
        return entropy_reward

    # def compute_entropy_advantage(self, states, rewards, lam=0.95, update=False):
    #     # TD(lambda) is used here
    #     V = self.entropy_value_nn(states)
    #     target = torch.zeros_like(rewards)

    #     old_V = V[: , : -1, : ]
    #     new_V = V[: , 1: , : ]   

    #     if update:
    #         target = rewards + self.gamma * new_V.detach()
    #         # Compute value loss for value network
    #         # MSE between predicted V'(s) and V(s), i.e. E(A(s, a) ** 2)
    #         value_loss = nn.functional.mse_loss(old_V, target)
    #         return value_loss

    #     # Compute advantage loss (A part of policy network loss)
    #     # A(s_t, a_t) = c_t + γ * V(s_{t+1}) - V(s_t)
    #     # TD(0) is used here
    #     advantage = rewards + self.gamma * new_V - old_V
    #     return advantage.view(-1)

    def compute_entropy_advantage(self, states, rewards, lam=0.95):
        """
        Truncated TD(λ) value/advantage computation as described in Sutton & Barto (2018).
        """
        B, T, _ = rewards.shape
        device = rewards.device

        # One forward with grad for the critic
        V = self.entropy_value_nn(states)          # [B, T+1, 1]
        V_det = V.detach()

        dones = torch.zeros((B, T, 1), device=device, dtype=V_det.dtype)
        not_done = 1.0 - dones

        # TD residuals using the *detached* V to avoid leaking gradients into advantages
        deltas = rewards + self.gamma * V_det[: , 1: , : ] * not_done - V_det[: , : -1, : ]   # [B, T, 1]

        # Backward GAE(λ)
        adv = torch.zeros_like(deltas)
        gae = torch.zeros((B, 1, 1), device=device, dtype=V_det.dtype)
        for t in reversed(range(T)):
            gae = deltas[: , t: t + 1, : ] + self.gamma * lam * not_done[: , t: t + 1, : ] * gae
            adv[: , t: t + 1, : ] = gae

        # Critic target and loss
        V_target = (adv + V_det[:, : -1, :]).detach()           # [B, T, 1]
        value_pred = V[: , : -1, :]                              # [B, T, 1] (with grad)
        value_loss = torch.nn.functional.mse_loss(value_pred, V_target)

        # Flatten advantage for PPO and (usually) normalize
        adv_flat = adv.view(-1).detach()
        # adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + self.eps)

        return adv_flat, value_loss

        # V = self.entropy_value_nn(states)  # shape: (B, T+1, 1)
        # B, T, _ = rewards.shape
        # device = rewards.device
        # lam = 0.95
        # P = 15

        # n_step_returns = torch.zeros((P, B, T), device=device)

        # # Compute V^{nstep}_p for p = 1 to P
        # for p in tqdm(range(1, P + 1)):
        #     for t in range(T):
        #         # horizon h = min(T, t + p)
        #         h = min(t + p, T)
        #         # Sum rewards r_t + ... + r_{h-1}
        #         discounts = torch.tensor([self.gamma ** i for i in range(h - t)], device=device).view(1, -1, 1)
        #         r_slice = rewards[:, t: h, :]  # shape: (B, h - t, 1)
        #         discounted_sum = torch.sum(discounts * r_slice, dim=1)  # (B, 1)

        #         # Add bootstrapped value V(s_h)
        #         v_boot = (self.gamma ** (h - t)) * V[: , h, : ]  # shape: (B, 1)
        #         n_step_returns[p - 1, : , t] = (discounted_sum + v_boot).squeeze(-1)

        # # TD(λ) target: weighted sum of n-step returns
        # weights = [(1 - lam) * lam ** (p - 1) for p in range(1, P)]
        # weights.append(lam ** (P - 1))  # Last term uses λ^{P-1}
        # weights = torch.tensor(weights, device=device).view(P, 1, 1)  # shape: (P, 1, 1)

        # V_lambda = torch.sum(weights * n_step_returns, dim=0).unsqueeze(-1)  # shape: (B, T, 1)

        # if update:
        #     V_old = V[: , : T, : ]  # shape: (B, T, 1)
        #     value_loss = nn.functional.mse_loss(V_old, V_lambda.detach())
        #     return value_loss

        # # Compute advantage A(s, a) = r + γV(s') - V(s)
        # advantage = V_lambda - V[: , : T, : ]
        # return advantage.view(-1)

    def compute_importance_weights_alt(self, behavioral_policy, target_policy, states, actions, same_policy=False):
        """
        states:  [B, T+1, ...]
        actions: [B, T]   (discrete: int indices; continuous: action vectors)
        returns: r.view(-1) where r_t = pi_new/pi_old per timestep (no normalization)
        """
        B, T = states.size(0), states.size(1) - 1

        if same_policy:
            # PPO expects r_t = 1 for all steps when new==old
            return torch.ones(B * T, dtype=self.float_type, device=self.device)

        # Slice to align with actions
        # Flatten time & batch
        st_flat = states[: , : -1].reshape(-1, *states.shape[2: ])          # [B*T, ...]
        if self.is_discrete:
            # actions comes in as [B, T] (ints); make it [B*T, 1] long on the right device
            ac_flat = actions.reshape(-1).unsqueeze(-1) # [B*T, 1]
        else:
            # continuous: keep as vectors [B*T, act_dim]
            ac_shape = actions.shape[2: ] if actions.ndim == 3 else ()
            ac_flat = actions.reshape(-1, *ac_shape)

        # Log-probs per step under new/old
        logp_new = target_policy.get_log_p(st_flat, ac_flat).squeeze(-1)   # expect [B*T] or [B*T,1]
        logp_old = behavioral_policy.get_log_p(st_flat, ac_flat).squeeze(-1).detach()

        # Per-timestep ratio r_t
        r = torch.exp(logp_new - logp_old)  # [B*T]
        return r.to(self.float_type)

    def compute_kl(self, behavioral_policy, target_policy, states, actions):
        st_flat = states[:, : -1].reshape(-1, *states.shape[2: ])
        if self.is_discrete:
            # actions comes in as [B, T] (ints); make it [B*T, 1] long on the right device
            ac_flat = actions.reshape(-1).unsqueeze(-1) # [B*T, 1]
        else:
            # continuous: keep as vectors [B*T, act_dim]
            ac_shape = actions.shape[2: ] if actions.ndim == 3 else ()
            ac_flat = actions.reshape(-1, *ac_shape)
        logp_new = target_policy.get_log_p(st_flat, ac_flat).squeeze(-1)
        logp_old = behavioral_policy.get_log_p(st_flat, ac_flat).squeeze(-1).detach()
        kl = torch.mean(logp_old - logp_new)
        kl = torch.clamp(kl, min=0.0)
        numeric_error = ~torch.isfinite(kl)
        return kl, numeric_error

    # def compute_kl(self, behavioral_policy, target_policy, states, actions):
    #     importance_weights = self.compute_importance_weights_alt(behavioral_policy, target_policy, states, actions)

    #     # Compute KL divergence between behavioral and target policy
    #     # if self.zeta == 1:
    #     #     kl = (1 / self.episode_nr / self.step_nr) * torch.sum(torch.log(self.k / (self.episode_nr * self.step_nr * weights_sum) + self.eps))
    #     # else:
    #     #     kl = (1 / (self.zeta - 1)) * torch.log(
    #     #         (1 / self.episode_nr / self.step_nr) * torch.sum(torch.pow(1 / (weights_sum + self.eps), self.zeta - 1)) + self.eps
    #     #     )

    #     kl = torch.mean(importance_weights * torch.log(importance_weights + self.eps))  
    #     numeric_error = torch.isinf(kl) or torch.isnan(kl)

    #     # Minimum KL is zero
    #     # NOTE: do not remove epsilon factor
    #     kl = torch.max(torch.tensor(0.0), kl)

    #     return kl, numeric_error

    def policy_update(self, entropy_advantage_loss, cost_advantage_loss):
        self.policy_optimizer.zero_grad()

        # Maximize entropy
        policy_loss = -entropy_advantage_loss + self.safety_weight * cost_advantage_loss
        numeric_error = torch.isinf(policy_loss) or torch.isnan(policy_loss)
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.target_policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        return policy_loss, numeric_error
        

    # Logging
    def log_epoch_statistics(self, csv_file_1, epoch, vae_RE, vae_KL, vae_loss, entropy_value_loss, 
                             cost_value_loss, policy_loss, safety_weight, value_lr, entropy_advantage,
                             cost_advantage, mean_cost, std_cost, num_off_iters, execution_time):
        # Log to csv file 1
        csv_file_1.write(f"{epoch},{vae_RE},{vae_KL},{vae_loss},{entropy_value_loss},{cost_value_loss},{policy_loss},{safety_weight},{value_lr},{entropy_advantage},{cost_advantage},{mean_cost},{std_cost},{num_off_iters},{execution_time}\n")
        csv_file_1.flush()

    def log_off_iter_statistics(self, csv_file_2, epoch, num_off_iter, global_off_iter, policy_loss,
                                entropy_advantage, cost_advantage, kl, mean_cost, std_cost, lr):
        # Log to csv file 2
        csv_file_2.write(f"{epoch},{num_off_iter},{global_off_iter},{policy_loss},{entropy_advantage},{cost_advantage},{kl},{mean_cost},{std_cost},{lr}\n")
        csv_file_2.flush()

    def renyi_vis(self, best_epoch, only_half=False):
        df = pd.read_csv(f"{self.out_path}/{self.env_id}.csv", index_col=False)
        if only_half:
            df = df.query("epoch <= 150")        
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))

        sns.lineplot(data=df, x="epoch", y="entropy_value_loss", ax=ax[0, 0])
        ax[0, 0].set_xlabel("Epoch Number")
        ax[0, 0].set_ylabel("Entropy Value Loss")
        ax[0, 0].axvline(x=best_epoch, color='black', linestyle='--', linewidth=1)
        ax[0, 0].set_title(f"{self.env_id[: -3]}: \nChange in Entropy Value Loss over Epochs")

        sns.lineplot(data = df, x="epoch", y="mean_cost", ax=ax[0, 1])
        ax[0, 1].set_xlabel("Epoch Number")
        ax[0, 1].set_ylabel("Total Cost")
        ax[0, 1].fill_between(
            df["epoch"],
            df["mean_cost"] - df["std_cost"],
            df["mean_cost"] + df["std_cost"],
            color="gray",
            alpha=0.3,
        )
        ax[0, 1].axhline(y=self.safety_constraint, color="red")
        ax[0, 1].axvline(x=best_epoch, color='black', linestyle='--', linewidth=1)
        ax[0, 1].set_title(f"{self.env_id[: -3]}: \nChange in Total Cost over Epochs")

        sns.lineplot(data=df, x="epoch", y="cost_value_loss", ax=ax[1, 0])
        ax[1, 0].set_xlabel("Epoch Number")
        ax[1, 0].set_ylabel("Cost Value Loss")
        ax[1, 0].axvline(x=best_epoch, color='black', linestyle='--', linewidth=1)
        ax[1, 0].set_title(f"{self.env_id[: -3]}: \nChange in Cost Value Loss over Epochs")

        sns.lineplot(data=df, x="epoch", y="safety_weight", ax=ax[1, 1])
        ax[1, 1].set_xlabel("Epoch Number")
        ax[1, 1].set_ylabel("Safety Weight")
        ax[1, 1].axvline(x=best_epoch, color='black', linestyle='--', linewidth=1)
        ax[1, 1].set_title(f"{self.env_id[: -3]}: \nChange in Safety Weight over Epochs")

        sns.lineplot(data=df, x="epoch", y="vae_RE", ax=ax[2, 0])
        ax[2, 0].set_xlabel("Epoch Number")
        ax[2, 0].set_ylabel("VAE RE")
        ax[2, 0].axvline(x=best_epoch, color='black', linestyle='--', linewidth=1)
        ax[2, 0].set_title(f"{self.env_id[: -3]}: \nChange in VAE Reconstruction Error over Epochs")
        
        sns.lineplot(data=df, x="epoch", y="vae_KL", ax=ax[2, 1])
        ax[2, 1].set_xlabel("Epoch Number")
        ax[2, 1].set_ylabel("VAE KL")
        ax[2, 1].axvline(x=best_epoch, color='black', linestyle='--', linewidth=1)
        ax[2, 1].set_title(f"{self.env_id[: -3]}: \nChange in VAE KL Divergence over Epochs")    

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
        fig.savefig(f"{self.out_path}/stats_visualization_{self.alg_name.lower()}_{entropy_file_name}_{self.env_id}.png") 
        plt.close('all')  


    # Main Training
    def _initialize_networks(self):
        # Initialize policy neural network
        # Create a behavioral policy and a target policy
        # (those with kl <= kl_threshold) during off policy opt
        self.behavioral_policy = self.create_policy()
        self.target_policy = self.create_policy()
        last_valid_target_policy = self.create_policy()
        self.behavioral_policy.load_state_dict(torch.load(f"results/{self.env_id[: -3]}/0-policy.pt"))        
        self.target_policy.load_state_dict(self.behavioral_policy.state_dict())
        last_valid_target_policy.load_state_dict(self.behavioral_policy.state_dict())

        # Initialize entropy value neural network
        self.entropy_value_nn = ValueNetwork(self.state_dim, self.device).to(self.device)
        self.entropy_value_optimizer = torch.optim.Adam(self.entropy_value_nn.parameters(), lr=self.entropy_value_lr, weight_decay=1e-4) 
        self.entropy_value_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.entropy_value_optimizer,
            mode='min',
            factor=0.5,
            patience=50,
            threshold=1e-4,
            min_lr=1e-6)         

        # Initialize cost value neural network
        self.cost_value_nn = ValueNetwork(self.state_dim, self.device).to(self.device)
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value_nn.parameters(), lr=self.cost_value_lr, weight_decay=1e-4) 
        self.cost_value_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.cost_value_optimizer,
            mode='min',
            factor=0.5,
            patience=50,
            threshold=1e-4,
            min_lr=1e-6)            

        # Set optimizer for policy
        self.policy_optimizer = torch.optim.Adam(self.target_policy.parameters(), lr=self.policy_lr)

        # Initialize VAE neural network
        D = self.state_dim
        L = 32  # latent dimension
        M = 64  # hidden dimension size
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
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.vae_lr, weight_decay=1e-4)   
        self.vae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.vae_optimizer,
            mode='min',
            factor=0.5,
            patience=50,
            threshold=1e-4,
            min_lr=1e-6)  
        
        return last_valid_target_policy  
    
    def _initialize_logging(self):
        # Create log files
        csv_file_1 = open(os.path.join(self.out_path, f"{self.env_id}.csv"), 'w')
        csv_file_1.write(",".join(['epoch', 'vae_RE', 'vae_KL', 'vae_loss', 'entropy_value_loss', 'cost_value_loss', 'policy_loss', 'safety_weight', 'value_learning_rate', 'entropy_advantage', 'cost_advantage', 'mean_cost', 'std_cost', 'num_off_iters','execution_time']))
        csv_file_1.write("\n")

        csv_file_2 = open(os.path.join(self.out_path, f"{self.env_id}_off_policy_iter.csv"), "w")
        csv_file_2.write(",".join(['epoch', 'off_policy_iter', 'global_off_policy_iter', 'policy_loss', 'entropy_advantage', 'cost_advantage', 'kl', 'mean_cost', 'std_cost', 'policy_learning_rate']))
        csv_file_2.write("\n")

        return csv_file_1, csv_file_2

    def _run_initial_evaluation(self, csv_file_1):
        # At epoch 0 do not optimize, just log stuff for the initial policy
        print(f"\nInitial epoch starts")
        t0 = time.time()

        # Entropy
        states, actions, costs, next_states, stds_or_probs = self.collect_particles(0)
        # Convert to PyTorch tensors
        states = torch.tensor(states, dtype=self.float_type, device=self.device)
        if self.is_discrete:
            actions = torch.tensor(actions, dtype=self.int_type, device=self.device)
        else:
            actions = torch.tensor(actions, dtype=self.float_type, device=self.device)
        costs = torch.tensor(costs, dtype=self.float_type, device=self.device)
        next_states = torch.tensor(next_states, dtype=self.float_type, device=self.device)
        stds_or_probs = torch.tensor(stds_or_probs, dtype=self.float_type, device=self.device)        

        with torch.inference_mode():
            vae_loss, vae_RE, vae_KL = self.train_vae(next_states, False)
            entropy_reward = self.compute_entropy_reward(next_states)  
            entropy_advantage, entropy_value_loss = self.compute_entropy_advantage(states, entropy_reward)
            cost_advantage, cost_value_loss = self.compute_cost_advantage(states, costs)
            
            importance_weights = self.compute_importance_weights_alt(self.behavioral_policy, self.behavioral_policy, states, actions, True)
            
            # Clip the advantage as per PPO
            clipped_adv = torch.where(
                entropy_advantage >= 0,
                (1 + self.epsilon) * entropy_advantage,
                (1 - self.epsilon) * entropy_advantage
            )
            # PPO loss: negative of min (r * A, clipped A)
            entropy_advantage_loss = torch.mean(torch.min(importance_weights * entropy_advantage, clipped_adv))
            cost_advantage_loss = torch.mean(cost_advantage * importance_weights)

            policy_loss = -entropy_advantage_loss + self.safety_weight * cost_advantage_loss

        print("\nEntropy computed")

        execution_time = time.time() - t0
        entropy_value_loss = entropy_value_loss.cpu().numpy()
        entropy_advantage_loss = entropy_advantage_loss.cpu().numpy()
        mean_cost, std_cost = self.compute_discounted_cost(costs)
        # mean_cost, std_cost = self.compute_actual_cost(costs)
        cost_value_loss = cost_value_loss.cpu().numpy()
        cost_advantage_loss = cost_advantage_loss.cpu().numpy()
        policy_loss = policy_loss.cpu().numpy()

        # Save initial policy
        torch.save(self.behavioral_policy.state_dict(), os.path.join(self.out_path, "0-policy.pt"))

        # Heatmap
        if self.env_id != "SafetyPointGoal1-v0":
            _, _, heatmap_entropy, heatmap_cost, state_dist = self.get_simulations(0)
            self._plot_heatmap(heatmap_entropy, heatmap_cost, state_dist, 0)

        # Log statistics for the initial policy
        self.log_epoch_statistics(
            csv_file_1=csv_file_1,
            epoch=0,
            vae_RE=vae_RE,
            vae_KL=vae_KL,
            vae_loss=vae_loss,
            entropy_value_loss=entropy_value_loss,
            cost_value_loss=cost_value_loss,
            policy_loss=policy_loss,
            safety_weight=self.safety_weight,
            value_lr=self.cost_value_lr,
            entropy_advantage=entropy_advantage_loss,
            cost_advantage=cost_advantage_loss,
            mean_cost=mean_cost,
            std_cost=std_cost,
            num_off_iters=0,
            execution_time=execution_time,
        )

        return vae_loss, entropy_value_loss, cost_value_loss, policy_loss

    def _optimize_kl(self, states, actions, cost_advantage, original_lr, entropy_advantage,
                     last_valid_target_policy, backtrack_iter, csv_file_2, epoch, num_off_iters, 
                     global_num_off_iters, mean_behavorial_costs, std_behavorial_costs):
                
        kl_threshold_reached = False

        while not kl_threshold_reached:
            print("\nOptimizing KL continues")
            importance_weights = self.compute_importance_weights_alt(self.behavioral_policy, self.target_policy, states, actions)
            # Clip the advantage as per PPO
            clipped_adv = torch.where(
                entropy_advantage >= 0,
                (1 + self.epsilon) * entropy_advantage,
                (1 - self.epsilon) * entropy_advantage
            )
            # PPO loss: negative of min (r * A, clipped A)
            entropy_advantage_loss = torch.mean(torch.min(importance_weights * entropy_advantage, clipped_adv))    
            cost_advantage_loss = torch.mean(cost_advantage * importance_weights)

            # Update policy network
            policy_loss, numeric_error = self.policy_update(entropy_advantage_loss, cost_advantage_loss)
            policy_loss = policy_loss.detach().cpu().numpy()

            with torch.inference_mode():
                kl, kl_numeric_error = self.compute_kl(self.behavioral_policy, self.target_policy, states, actions)
                kl = kl.cpu().numpy()                    

            if not numeric_error and not kl_numeric_error and kl <= self.trust_region_threshold:
                # Valid update
                last_valid_target_policy.load_state_dict(self.target_policy.state_dict())
                num_off_iters += 1
                global_num_off_iters += 1
                lr = self.policy_lr
                # Log statistics for this off policy iteration
                self.log_off_iter_statistics(csv_file_2, epoch, num_off_iters - 1, global_num_off_iters - 1,
                                             policy_loss, entropy_advantage_loss, cost_advantage_loss, kl, 
                                             mean_behavorial_costs, std_behavorial_costs, lr)            

            else:
                if self.use_backtracking:
                    # We are here either because we could not perform any update for this epoch
                    # or because we need to perform one last update
                    if not backtrack_iter == self.max_backtrack_try:
                        self.target_policy.load_state_dict(last_valid_target_policy.state_dict())

                        self.policy_lr = original_lr / (self.backtrack_coeff ** backtrack_iter)

                        for param_group in self.policy_optimizer.param_groups:
                            param_group['lr'] = self.policy_lr

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

        return last_valid_target_policy, backtrack_iter, num_off_iters, global_num_off_iters

    def _epoch_train(self, vae_loss, entropy_value_loss, cost_value_loss, policy_loss, last_valid_target_policy, csv_file_1, csv_file_2):
        if self.use_backtracking:
            original_lr = self.policy_lr

        best_vae_loss = vae_loss
        best_cost_value_loss = cost_value_loss
        best_entropy_value_loss = entropy_value_loss
        best_loss = policy_loss
        best_epoch = 0
        global_num_off_iters = 0

        for epoch in range(1, self.epoch_nr + 1):
            print(f"Epoch {epoch} starts")
            t0 = time.time()

            # Off policy optimization
            num_off_iters = 0

            # Collect particles to optimize off policy
            states, actions, costs, next_states, stds_or_probs = self.collect_particles(epoch)
            # Convert to PyTorch tensors
            states = torch.tensor(states, dtype=self.float_type, device=self.device)
            if self.is_discrete:
                actions = torch.tensor(actions, dtype=self.int_type, device=self.device)
            else:
                actions = torch.tensor(actions, dtype=self.float_type, device=self.device)
            costs = torch.tensor(costs, dtype=self.float_type, device=self.device)
            next_states = torch.tensor(next_states, dtype=self.float_type, device=self.device)
            stds_or_probs = torch.tensor(stds_or_probs, dtype=self.float_type, device=self.device)  

            if self.use_backtracking:
                self.policy_lr = original_lr

                for param_group in self.policy_optimizer.param_groups:
                    param_group['lr'] = self.policy_lr

                backtrack_iter = 1
            else:
                backtrack_iter = None            

            # Update VAE
            vae_loss, vae_RE, vae_KL = self.train_vae(next_states)                 
            if vae_loss < best_vae_loss:
                best_vae_loss = vae_loss

            # Update entropy value network
            self.entropy_value_optimizer.zero_grad()
            entropy_reward = self.compute_entropy_reward(next_states)              
            
            # entropy_value_loss = self.compute_entropy_advantage(states, entropy_reward, update=True)
            entropy_advantage, entropy_value_loss = self.compute_entropy_advantage(states, entropy_reward)
            entropy_value_loss.backward()
            self.entropy_value_optimizer.step()
            entropy_value_loss = entropy_value_loss.detach().cpu().numpy()
            self.entropy_value_scheduler.step(entropy_value_loss)
            
            # Save best entropy value model
            if entropy_value_loss < best_entropy_value_loss:
                best_entropy_value_loss = entropy_value_loss          

            # Update safety weight
            self.safety_weight = max(0, self.safety_weight + self.safety_weight_lr * 
                                     (torch.sum(costs) / costs.shape[0] - self.safety_constraint))
            mean_behavorial_costs, std_behavorial_costs = self.compute_discounted_cost(costs)

            # Update cost value network
            self.cost_value_optimizer.zero_grad()
            cost_advantage, cost_value_loss = self.compute_cost_advantage(states, costs)        
            cost_value_loss.backward()
            self.cost_value_optimizer.step()
            cost_value_loss = cost_value_loss.detach().cpu().numpy()
            self.cost_value_scheduler.step(cost_value_loss)
            
            # Save best entropy value model
            if cost_value_loss < best_cost_value_loss:
                best_cost_value_loss = cost_value_loss
            
            last_valid_target_policy, backtrack_iter, num_off_iters, global_num_off_iters = self._optimize_kl(states, actions, costs, original_lr, entropy_advantage, last_valid_target_policy, backtrack_iter, csv_file_2, epoch, num_off_iters, global_num_off_iters, mean_behavorial_costs, std_behavorial_costs)

            # Compute entropy of new policy
            with torch.inference_mode():
                importance_weights = self.compute_importance_weights_alt(last_valid_target_policy, last_valid_target_policy, states, actions, True)

                # Clip the advantage as per PPO
                clipped_adv = torch.where(
                    entropy_advantage >= 0,
                    (1 + self.epsilon) * entropy_advantage,
                    (1 - self.epsilon) * entropy_advantage
                )
                # PPO loss: negative of min (r * A, clipped A)
                entropy_advantage_loss = torch.mean(torch.min(importance_weights * entropy_advantage, clipped_adv))
                cost_advantage_loss = torch.mean(cost_advantage * importance_weights)

            # Update policy network
            self.behavioral_policy.load_state_dict(last_valid_target_policy.state_dict())
            self.target_policy.load_state_dict(last_valid_target_policy.state_dict())
            policy_loss = (-entropy_advantage_loss + self.safety_weight * cost_advantage_loss).cpu().numpy()
            mean_cost, std_cost = self.compute_discounted_cost(costs)
            # mean_cost, std_cost = self.compute_actual_cost(costs)
            execution_time = time.time() - t0

            if policy_loss < best_loss:
                best_loss = policy_loss
                # Save policy
                torch.save(self.behavioral_policy.state_dict(), os.path.join(self.out_path, f"{epoch}-policy.pt"))
                best_epoch = epoch

            # Log statistics for the initial policy
            self.log_epoch_statistics(
                csv_file_1=csv_file_1,
                epoch=epoch,
                vae_RE=vae_RE,
                vae_KL=vae_KL,
                vae_loss=vae_loss,
                entropy_value_loss=entropy_value_loss,
                cost_value_loss=cost_value_loss,
                policy_loss=policy_loss,
                safety_weight=self.safety_weight,
                value_lr=self.cost_value_lr,
                entropy_advantage=entropy_advantage_loss,
                cost_advantage=cost_advantage_loss,
                mean_cost=mean_cost,
                std_cost=std_cost,
                num_off_iters=num_off_iters,
                execution_time=execution_time,
            )                           

        return best_epoch        

    def train(self):
        self._initialize_device()
        self._initialize_envs()
        last_valid_target_policy = self._initialize_networks()
        csv_file_1, csv_file_2 = self._initialize_logging()

        vae_loss, entropy_value_loss, cost_value_loss, policy_loss = self._run_initial_evaluation(csv_file_1)

        self._epoch_train(vae_loss, entropy_value_loss, cost_value_loss, policy_loss, last_valid_target_policy, csv_file_1, csv_file_2)                                                                

        if self.env_id == "Pendulum-v1" or self.env_id == "MountainCarContinuous-v0":
            self.visualize_policy_comparison()   
            self.visualize_policy_heatmap(True)  

        self.compute_best_epochs()        

        if isinstance(self.envs, gymnasium.vector.VectorEnv) or isinstance(self.envs, safety_gymnasium.vector.VectorEnv):
            self.envs.close()