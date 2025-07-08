import torch
import gymnasium
import safety_gymnasium
import time
from tabulate import tabulate
import os
from src.value import ValueNetwork
from src.agents.BaseAgent import BaseAgent

int_choice = torch.int64
float_choice = torch.float64
torch.set_default_dtype(float_choice)

class CEM(BaseAgent):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)

        # == Algorithm Hyperparameters ==
        self.max_off_iters = 30
        self.backtrack_coeff = 2
        self.max_backtrack_try = 10
        if self.env_id == "MountainCarContinuous-v0" or self.env_id == "MountainCar-v0" or "CartPole-v1" or "Pendulum-v1":
            self.trust_region_threshold = 0.5
        else:
            self.trust_region_threshold = 0.1
        self.alg_name = "CEM"    
  
    # Optimization and Evaluation
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

    def policy_update(self, optimizer, cost_advantage_loss, distances, indices, importance_weights):
        optimizer.zero_grad()

        # Maximize entropy
        mean_entropy, std_entropy = self.compute_entropy(distances, indices, self.use_behavioral, self.zeta, importance_weights)
        loss = -mean_entropy + self.safety_weight * cost_advantage_loss

        numeric_error = torch.isinf(loss) or torch.isnan(loss)

        loss.backward()
        optimizer.step()

        return loss, numeric_error, mean_entropy, std_entropy


    # Logging
    def log_epoch_statistics(self, log_file, csv_file_1, csv_file_2, epoch,
                            cost_value_loss, policy_loss, safety_weight, value_lr, 
                            cost_advantage, mean_entropy, std_entropy, mean_cost, std_cost, num_off_iters, execution_time,
                            heatmap_image, heatmap_entropy, heatmap_cost, backtrack_iters, backtrack_lr):
        # Prepare tabulate table
        table = []
        fancy_float = lambda f : f"{f:.3f}"
        table.extend([
            ["Epoch", epoch],
            ["Execution time (s)", fancy_float(execution_time)],
            ["Cost Value loss", fancy_float(cost_value_loss)],
            ["Policy loss", fancy_float(policy_loss)],
            ["Cost Advantage", fancy_float(cost_advantage)],
            ["Entropy", fancy_float(mean_entropy)],
            ["Cost", fancy_float(mean_cost)],
            ["Off-policy iters", num_off_iters]
        ])

        if heatmap_image is not None:
            table.extend([
                ["Heatmap entropy", fancy_float(heatmap_entropy)],
                ["Heatmap cost", fancy_float(heatmap_cost)],
            ])

        if backtrack_iters is not None:
            table.extend([
                ["Backtrack iters", backtrack_iters],
            ])

        fancy_grid = tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign='right')

        # Log to csv file 1
        csv_file_1.write(f"{epoch},{cost_value_loss},{policy_loss},{safety_weight},{value_lr},{cost_advantage},{mean_entropy},{std_entropy},{mean_cost},{std_cost},{num_off_iters},{execution_time}\n")
        csv_file_1.flush()

        # Log to csv file 2
        if heatmap_image is not None:
            csv_file_2.write(f"{epoch},{heatmap_entropy},{heatmap_cost}\n")
            csv_file_2.flush()

        # Log to stdout and log file
        log_file.write(fancy_grid)
        log_file.write("\n\n")
        log_file.flush()
        print(fancy_grid)

    def log_off_iter_statistics(self, csv_file_3, epoch, num_off_iter, global_off_iter,
                                loss, cost_advantage, mean_entropy, std_entropy, kl, mean_cost, std_cost, lr):
        # Log to csv file 3
        csv_file_3.write(f"{epoch},{num_off_iter},{global_off_iter},{loss},{cost_advantage},{mean_entropy},{std_entropy},{kl},{mean_cost},{std_cost},{lr}\n")
        csv_file_3.flush()
    

    # Main Training
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
        self.policy_optimizer = torch.optim.Adam(self.target_policy.parameters(), lr = self.policy_lr)

        # Initialize value neural network
        self.cost_value_nn = ValueNetwork(self.state_dim, self.device).to(self.device)
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value_nn.parameters(), lr = self.cost_value_lr)

        return last_valid_target_policy       

    def _initialize_logging(self):
        # Create log files
        log_file = open(os.path.join((self.out_path), 'log_file.txt'), 'a', encoding="utf-8")

        csv_file_1 = open(os.path.join(self.out_path, f"{self.env_id}.csv"), 'w')
        csv_file_1.write(",".join(['epoch', 'cost_value_loss', 'policy_loss', 'safety_weight', 'value_learning_rate', 'cost_advantage', 'mean_entropy', 'std_entropy', 'mean_cost', 'std_cost', 'num_off_iters','execution_time']))
        csv_file_1.write("\n")

        if self.heatmap_discretizer is not None:
            csv_file_2 = open(os.path.join(self.out_path, f"{self.env_id}-heatmap.csv"), 'w')
            csv_file_2.write(",".join(['epoch', 'heatmap_entropy', 'heatmap_cost']))
            csv_file_2.write("\n")
        else:
            csv_file_2 = None

        csv_file_3 = open(os.path.join(self.out_path, f"{self.env_id}_off_policy_iter.csv"), "w")
        csv_file_3.write(",".join(['epoch', 'off_policy_iter', 'global_off_policy_iter', 'loss', 'cost_advantage', 'mean_entropy', 'std_entropy', 'kl', 'mean_cost', 'std_cost', 'policy_learning_rate']))
        csv_file_3.write("\n")

        return log_file, csv_file_1, csv_file_2, csv_file_3

    def _run_initial_evaluation(self, log_file, csv_file_1, csv_file_2):
        # At epoch 0 do not optimize, just log stuff for the initial policy
        print(f"\nInitial epoch starts")
        t0 = time.time()

        # Entropy
        states, actions, costs, next_states, distances, indices = \
            self.collect_particles_and_compute_knn(0)

        with torch.inference_mode():
            importance_weights = self.compute_importance_weights(self.behavioral_policy, self.behavioral_policy, states, actions, True)
            mean_entropy, std_entropy = self.compute_entropy(distances, indices, self.use_behavioral, self.zeta, importance_weights)   
            cost_value_loss = self.compute_cost_advantage(states, costs, True)
            cost_advantage = self.compute_cost_advantage(states, costs)
            cost_advantage_loss = torch.mean(cost_advantage.view(-1) * importance_weights)     
            policy_loss = -mean_entropy + self.safety_weight * cost_advantage_loss                 

        print("\nEntropy computed")

        execution_time = time.time() - t0
        mean_entropy = mean_entropy.cpu().numpy()
        std_entropy = std_entropy.cpu().numpy()
        mean_cost, std_cost = self.compute_discounted_cost(costs)
        cost_value_loss = cost_value_loss.cpu().numpy()
        cost_advantage_loss = cost_advantage_loss.cpu().numpy()
        policy_loss = policy_loss.cpu().numpy()

        # Heatmap
        heatmap_entropy, heatmap_cost, heatmap_image = self._plot_heatmap(0)

        # Save initial policy
        torch.save(self.behavioral_policy.state_dict(), os.path.join(self.out_path, "0-policy.pt"))
        torch.save(self.cost_value_nn.state_dict(), os.path.join(self.out_path, "0-cost-value.pt"))

        # Log statistics for the initial policy
        self.log_epoch_statistics(
            log_file=log_file, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
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
            heatmap_image=heatmap_image,
            heatmap_entropy=heatmap_entropy,
            heatmap_cost=heatmap_cost,
            backtrack_iters=None,
            backtrack_lr=None
        )

        return cost_value_loss, policy_loss, heatmap_entropy, heatmap_cost, heatmap_image
    
    def _optimize_kl(self, states, actions, costs, distances, indices, original_lr, 
                     last_valid_target_policy, backtrack_iter, csv_file_3, epoch, num_off_iters, 
                     global_num_off_iters, mean_behavorial_costs, std_behavorial_costs):
        
        kl_threshold_reached = False

        while not kl_threshold_reached:
            print("\nOptimizing KL continues")
            importance_weights = self.compute_importance_weights(self.behavioral_policy, self.target_policy, states, actions)
            # Use the newly computed advantage loss
            cost_advantage = self.compute_cost_advantage(states, costs)
            cost_advantage_loss = torch.mean(cost_advantage.view(-1) * importance_weights)             
            # Update target policy network    
            loss, numeric_error, mean_entropy, std_entropy = self.policy_update(self.policy_optimizer, cost_advantage_loss,
                                                                                distances, indices, importance_weights)
            mean_entropy = mean_entropy.detach().cpu().numpy()
            std_entropy = std_entropy.detach().cpu().numpy()
            loss = loss.detach().cpu().numpy()

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
                self.log_off_iter_statistics(csv_file_3, epoch, num_off_iters - 1, global_num_off_iters - 1,
                                             loss, cost_advantage_loss, mean_entropy, std_entropy, kl, 
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
    
    def _epoch_train(self, cost_value_loss, policy_loss, last_valid_target_policy, log_file, csv_file_1, csv_file_2, csv_file_3, heatmap_entropy, heatmap_cost, heatmap_image):
        if self.use_backtracking:
            original_lr = self.policy_lr

        best_cost_value_loss = cost_value_loss
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

            # Update value network
            self.cost_value_optimizer.zero_grad()
            cost_value_loss = self.compute_cost_advantage(states, costs, True)            
            cost_value_loss.backward()
            self.cost_value_optimizer.step()
            cost_value_loss = cost_value_loss.detach().cpu().numpy()
            if cost_value_loss < best_cost_value_loss:
                torch.save(self.cost_value_nn.state_dict(), os.path.join(self.out_path, f"{epoch}-cost-value.pt"))

            last_valid_target_policy, backtrack_iter, num_off_iters, global_num_off_iters = self._optimize_kl(states, actions, costs, distances, indices, original_lr, last_valid_target_policy, backtrack_iter, csv_file_3, epoch, num_off_iters, global_num_off_iters, mean_behavorial_costs, std_behavorial_costs)

            # Compute entropy of new policy
            with torch.inference_mode():
                importance_weights = self.compute_importance_weights(last_valid_target_policy, last_valid_target_policy, states, actions, True)
                cost_advantage = self.compute_cost_advantage(states, costs)
                cost_advantage_loss = torch.mean(cost_advantage.view(-1) * importance_weights)                       
                mean_entropy, std_entropy = self.compute_entropy(distances, indices, self.use_behavioral, self.zeta, importance_weights)

            if torch.isnan(mean_entropy) or torch.isinf(mean_entropy):
                print("Aborting because final entropy is nan or inf...")
                print("There is most likely a problem in knn aliasing. Use a higher k.")
                exit()
            else:
                # End of epoch, prepare statistics to log
                # Update behavioral policy
                self.behavioral_policy.load_state_dict(last_valid_target_policy.state_dict())
                self.target_policy.load_state_dict(last_valid_target_policy.state_dict())

                loss = (-mean_entropy + self.safety_weight * cost_advantage_loss).cpu().numpy()
                mean_entropy = mean_entropy.cpu().numpy()
                std_entropy = std_entropy.cpu().numpy()
                mean_cost, std_cost = self.compute_discounted_cost(costs)
                execution_time = time.time() - t0

                # Log statistics for the initial policy
                self.log_epoch_statistics(
                    log_file=log_file, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
                    epoch=epoch,
                    cost_value_loss=cost_value_loss,
                    policy_loss=loss,
                    safety_weight=self.safety_weight,
                    value_lr=self.cost_value_lr,
                    cost_advantage=cost_advantage_loss,
                    mean_entropy=mean_entropy,
                    std_entropy=std_entropy,
                    mean_cost=mean_cost,
                    std_cost=std_cost,
                    num_off_iters=num_off_iters,
                    execution_time=execution_time,
                    heatmap_image=heatmap_image,
                    heatmap_entropy=heatmap_entropy,
                    heatmap_cost=heatmap_cost,
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

        cost_value_loss, policy_loss, heatmap_entropy, heatmap_cost, heatmap_image = self._run_initial_evaluation(log_file, csv_file_1, csv_file_2)

        best_epoch = self._epoch_train(cost_value_loss, policy_loss, last_valid_target_policy, log_file, csv_file_1, csv_file_2, csv_file_3, heatmap_entropy, heatmap_cost, heatmap_image)                                                                
        heatmap_entropy, heatmap_cost, heatmap_image = self._plot_heatmap(best_epoch)

        if self.env_id == "Pendulum-v1" or self.env_id == "MountainCarContinuous-v0":
            self.visualize_policy_comparison()    
            self.visualize_policy_heatmap(False)     

        if isinstance(self.envs, gymnasium.vector.VectorEnv) or isinstance(self.envs, safety_gymnasium.vector.VectorEnv):
            self.envs.close()