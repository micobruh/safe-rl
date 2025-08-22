import torch
import numpy as np
import gymnasium
import safety_gymnasium
import time
import os
from src.value import ValueNetwork
from src.agents.BaseAgent import BaseAgent
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

int_choice = torch.int64
float_choice = torch.float64
torch.set_default_dtype(float_choice)
np.random.seed(0)
torch.manual_seed(0)

class CEM(BaseAgent):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)

        # == Algorithm Hyperparameters ==
        self.alg_name = "CEM"    
  
    # Optimization and Evaluation
    def compute_kl(self, behavioral_policy, target_policy, states, actions, indices):
        importance_weights = self.compute_importance_weights(behavioral_policy, target_policy, states, actions)

        weights_sum = torch.sum(importance_weights[indices[: , : -1]], dim = 1)

        # Compute KL divergence between behavioral and target policy
        kl = (1 / self.episode_nr / self.step_nr) * torch.sum(torch.log(self.k / (self.episode_nr * self.step_nr * weights_sum) + self.eps))    
        numeric_error = torch.isinf(kl) or torch.isnan(kl)

        # Minimum KL is zero
        # NOTE: do not remove epsilon factor
        kl = torch.max(torch.tensor(0.0), kl)

        return kl, numeric_error
    
    def compute_behavioral_entropy_diff(self, old_mean_entropy, mean_entropy):
        if old_mean_entropy is None:
            return 0, False
        entropy_diff = np.linalg.norm(mean_entropy - old_mean_entropy) / mean_entropy.size
        numeric_error = not np.isfinite(entropy_diff)
        return entropy_diff, numeric_error

    def policy_update(self, cost_advantage_loss, distances, indices, importance_weights):
        self.policy_optimizer.zero_grad()

        # Maximize entropy
        mean_entropy, std_entropy = self.compute_entropy(distances, indices, self.use_behavioral, self.zeta, importance_weights)
        policy_loss = -mean_entropy / self.step_nr + self.safety_weight * cost_advantage_loss

        numeric_error = torch.isinf(policy_loss) or torch.isnan(policy_loss)

        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.target_policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        return policy_loss, numeric_error, mean_entropy, std_entropy


    # Logging
    def log_epoch_statistics(self, csv_file_1, epoch,
                            cost_value_loss, policy_loss, safety_weight, value_lr, 
                            cost_advantage, mean_entropy, std_entropy, mean_cost, std_cost, num_off_iters, execution_time):
        # Log to csv file 1
        csv_file_1.write(f"{epoch},{cost_value_loss},{policy_loss},{safety_weight},{value_lr},{cost_advantage},{mean_entropy},{std_entropy},{mean_cost},{std_cost},{num_off_iters},{execution_time}\n")
        csv_file_1.flush()

    def log_off_iter_statistics(self, csv_file_2, epoch, num_off_iter, global_off_iter,
                                policy_loss, cost_advantage, mean_entropy, std_entropy, kl, mean_cost, std_cost, lr):
        # Log to csv file 2
        csv_file_2.write(f"{epoch},{num_off_iter},{global_off_iter},{policy_loss},{cost_advantage},{mean_entropy},{std_entropy},{kl},{mean_cost},{std_cost},{lr}\n")
        csv_file_2.flush()

    def cem_vis(self, best_epoch, only_half=False):
        df = pd.read_csv(f"{self.out_path}/{self.env_id}.csv", index_col=False)
        if only_half:
            df = df.query("epoch <= 150")
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

        sns.lineplot(data=df, x="epoch", y="mean_entropy", ax=ax[0, 0])
        ax[0, 0].set_xlabel("Epoch Number")
        ax[0, 0].set_ylabel("State Entropy")
        ax[0, 0].fill_between(
            df["epoch"],
            df["mean_entropy"] - df["std_entropy"],
            df["mean_entropy"] + df["std_entropy"],
            color="gray",
            alpha=0.3,
        )
        ax[0, 0].axvline(x=best_epoch, color='black', linestyle='--', linewidth=1)
        ax[0, 0].set_title(f"{self.env_id[: -3]}: \nChange in State Entropy over Epochs")

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
        # Create a behavioral, a target policy and a tmp policy used to save valid target policies
        # (those with kl <= kl_threshold) during off policy opt
        self.behavioral_policy = self.create_policy()
        self.target_policy = self.create_policy()
        last_valid_target_policy = self.create_policy()
        self.behavioral_policy.load_state_dict(torch.load(f"results/{self.env_id[: -3]}/0-policy.pt"))
        self.target_policy.load_state_dict(self.behavioral_policy.state_dict())
        last_valid_target_policy.load_state_dict(self.behavioral_policy.state_dict())

        # Set optimizer for policy
        self.policy_optimizer = torch.optim.Adam(self.target_policy.parameters(), lr=self.policy_lr)

        # Initialize value neural network
        self.cost_value_nn = ValueNetwork(self.state_dim, self.device).to(self.device)
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value_nn.parameters(), lr=self.cost_value_lr, weight_decay=1e-4)
        self.cost_value_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.cost_value_optimizer,
            mode='min',
            factor=0.5,
            patience=50,
            threshold=1e-4,
            min_lr=1e-6)        

        return last_valid_target_policy       

    def _initialize_logging(self):
        # Create log files
        csv_file_1 = open(os.path.join(self.out_path, f"{self.env_id}.csv"), 'w')
        csv_file_1.write(",".join(['epoch', 'cost_value_loss', 'policy_loss', 'safety_weight', 'value_learning_rate', 'cost_advantage', 'mean_entropy', 'std_entropy', 'mean_cost', 'std_cost', 'num_off_iters','execution_time']))
        csv_file_1.write("\n")

        csv_file_2 = open(os.path.join(self.out_path, f"{self.env_id}_off_policy_iter.csv"), "w")
        csv_file_2.write(",".join(['epoch', 'off_policy_iter', 'global_off_policy_iter', 'policy_loss', 'cost_advantage', 'mean_entropy', 'std_entropy', 'kl', 'mean_cost', 'std_cost', 'policy_learning_rate']))
        csv_file_2.write("\n")

        return csv_file_1, csv_file_2

    def _run_initial_evaluation(self, csv_file_1):
        # At epoch 0 do not optimize, just log stuff for the initial policy
        print(f"\nInitial epoch starts")
        t0 = time.time()

        # Entropy
        states, actions, costs, next_states, distances, indices = \
            self.collect_particles_and_compute_knn(0)

        with torch.inference_mode():
            importance_weights = self.compute_importance_weights(self.behavioral_policy, self.behavioral_policy, states, actions, True)
            mean_entropy, std_entropy = self.compute_entropy(distances, indices, self.use_behavioral, self.zeta, importance_weights)   
            cost_advantage, cost_value_loss = self.compute_cost_advantage(states, costs)
            cost_advantage_loss = torch.mean(cost_advantage * importance_weights)
            policy_loss = -mean_entropy / self.step_nr + self.safety_weight * cost_advantage_loss             

        print("\nEntropy computed")

        execution_time = time.time() - t0
        mean_entropy = mean_entropy.cpu().numpy()
        std_entropy = std_entropy.cpu().numpy()
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
            cost_value_loss=cost_value_loss,
            policy_loss=policy_loss,
            safety_weight=self.safety_weight,
            value_lr=self.cost_value_lr,
            cost_advantage=cost_advantage_loss,
            mean_entropy=mean_entropy,
            std_entropy=std_entropy,
            mean_cost=mean_cost,
            std_cost=std_cost,
            num_off_iters=0,
            execution_time=execution_time,
        )

        return cost_value_loss, policy_loss
    
    def _optimize_kl(self, states, actions, cost_advantage, distances, indices, original_lr, 
                     last_valid_target_policy, backtrack_iter, csv_file_2, epoch, num_off_iters, 
                     global_num_off_iters, mean_behavorial_costs, std_behavorial_costs):
                
        kl_threshold_reached = False

        while not kl_threshold_reached:
            print("\nOptimizing KL continues")
            importance_weights = self.compute_importance_weights(self.behavioral_policy, self.target_policy, states, actions)
            # Use the newly computed advantage loss
            cost_advantage_loss = torch.mean(cost_advantage * importance_weights)      
            # Update target policy network    
            policy_loss, numeric_error, mean_entropy, std_entropy = self.policy_update(cost_advantage_loss, distances, indices, importance_weights)
            mean_entropy = mean_entropy.detach().cpu().numpy()
            std_entropy = std_entropy.detach().cpu().numpy()
            policy_loss = policy_loss.detach().cpu().numpy()

            with torch.inference_mode():
                kl, kl_numeric_error = self.compute_kl(self.behavioral_policy, self.target_policy, states, actions, indices)
                kl = kl.cpu().numpy()                    

            if not numeric_error and not kl_numeric_error and kl <= self.trust_region_threshold:
                # Valid update
                last_valid_target_policy.load_state_dict(self.target_policy.state_dict())
                num_off_iters += 1
                global_num_off_iters += 1
                lr = self.policy_lr
                # Log statistics for this off policy iteration
                self.log_off_iter_statistics(csv_file_2, epoch, num_off_iters - 1, global_num_off_iters - 1,
                                             policy_loss, cost_advantage_loss, mean_entropy, std_entropy, kl, 
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
    
    def _epoch_train(self, cost_value_loss, policy_loss, last_valid_target_policy, csv_file_1, csv_file_2):
        if self.use_backtracking:
            original_lr = self.policy_lr

        best_cost_value_loss = cost_value_loss
        best_loss = policy_loss
        best_epoch = 0
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
                self.policy_lr = original_lr

                for param_group in self.policy_optimizer.param_groups:
                    param_group['lr'] = self.policy_lr

                backtrack_iter = 1
            else:
                backtrack_iter = None

            # Update safety weight
            self.safety_weight = max(0, self.safety_weight + self.safety_weight_lr * 
                                     (torch.sum(costs) / self.episode_nr - self.safety_constraint))
            mean_behavorial_costs, std_behavorial_costs = self.compute_discounted_cost(costs)
            # mean_behavorial_costs, std_behavorial_costs = self.compute_actual_cost(costs)

            # Update cost value network
            self.cost_value_optimizer.zero_grad()
            cost_advantage, cost_value_loss = self.compute_cost_advantage(states, costs)
            cost_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cost_value_nn.parameters(), max_norm=1.0) 
            self.cost_value_optimizer.step()
            cost_value_loss = cost_value_loss.detach().cpu().item()
            self.cost_value_scheduler.step(cost_value_loss)

            # Save best cost value model
            if cost_value_loss < best_cost_value_loss:
                best_cost_value_loss = cost_value_loss

            last_valid_target_policy, backtrack_iter, num_off_iters, global_num_off_iters = self._optimize_kl(states, actions, costs, distances, indices, original_lr, last_valid_target_policy, backtrack_iter, csv_file_2, epoch, num_off_iters, global_num_off_iters, mean_behavorial_costs, std_behavorial_costs)

            # Compute entropy of new policy
            with torch.inference_mode():
                importance_weights = self.compute_importance_weights(last_valid_target_policy, last_valid_target_policy, states, actions, True)
                cost_advantage_loss = torch.mean(cost_advantage * importance_weights)
                mean_entropy, std_entropy = self.compute_entropy(distances, indices, self.use_behavioral, self.zeta, importance_weights)

            if torch.isnan(mean_entropy):
                print("Aborting because final entropy is nan...")
                print("There is most likely a problem in knn aliasing. Use a higher k.")
                exit()
            elif torch.isinf(mean_entropy):
                print("Aborting because final entropy is inf...")
                print("There is most likely a problem in knn aliasing. Use a higher k.")
                exit()                
            else:
                # End of epoch, prepare statistics to log
                # Update behavioral policy
                self.behavioral_policy.load_state_dict(last_valid_target_policy.state_dict())
                self.target_policy.load_state_dict(last_valid_target_policy.state_dict())
                policy_loss = (-mean_entropy / self.step_nr + self.safety_weight * cost_advantage_loss).cpu().numpy()
                mean_entropy = mean_entropy.cpu().numpy()
                std_entropy = std_entropy.cpu().numpy()
                mean_cost, std_cost = self.compute_discounted_cost(costs)
                # mean_cost, std_cost = self.compute_actual_cost(costs)
                execution_time = time.time() - t0

                if policy_loss < best_loss:
                    # Save policy
                    best_loss = policy_loss
                    torch.save(self.behavioral_policy.state_dict(), os.path.join(self.out_path, f"{epoch}-policy.pt"))
                    best_epoch = epoch

                # Log statistics for the initial policy
                self.log_epoch_statistics(
                    csv_file_1=csv_file_1, 
                    epoch=epoch,
                    cost_value_loss=cost_value_loss,
                    policy_loss=policy_loss,
                    safety_weight=self.safety_weight,
                    value_lr=self.cost_value_lr,
                    cost_advantage=cost_advantage_loss,
                    mean_entropy=mean_entropy,
                    std_entropy=std_entropy,
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

        cost_value_loss, policy_loss = self._run_initial_evaluation(csv_file_1)

        self._epoch_train(cost_value_loss, policy_loss, last_valid_target_policy, csv_file_1, csv_file_2)                                                                

        if self.env_id == "Pendulum-v1" or self.env_id == "MountainCarContinuous-v0":
            self.visualize_policy_comparison()    
            self.visualize_policy_heatmap(True)   

        self.compute_best_epochs()      

        if isinstance(self.envs, gymnasium.vector.VectorEnv) or isinstance(self.envs, safety_gymnasium.vector.VectorEnv):
            self.envs.close()