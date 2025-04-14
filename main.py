from src.agents.MEPOL import MEPOL
from src.agents.CEM import CEM

# obj = MEPOL(env_id = "SafetyPointGoal2-v0", k = 4, delta = 0.1, max_off_iters = 30,
#           use_backtracking = 1, backtrack_coeff = 2, max_backtrack_try = 10, eps = 0,
#           lambda_policy = 1e-5, episode_nr = 10, step_nr = 1000, epoch_nr = 500,
#           heatmap_discretizer = -1,
#           heatmap_cmap = -1, heatmap_labels = -1, heatmap_interp = -1,
#           seed = 0, out_path = "results/MEPOL", num_workers = 5)
# obj.train()

# obj = MEPOL(env_id = "SafetyPointGoal1-v0", k = 4, delta = 0.1, max_off_iters = 30,
#           use_backtracking = 1, backtrack_coeff = 2, max_backtrack_try = 10, eps = 1e-5,
#           lambda_policy = 1e-5, episode_nr = 24, step_nr = 500, epoch_nr = 2,
#           heatmap_cmap = 'Blues', 
#           heatmap_labels = ('X', 'Y'), heatmap_interp = 'spline16',
#           seed = 0, out_path = "results/SafetyPointGoal1/MEPOL_temp")
# obj.train()

# obj = CEM(env_id = "SafetyPointGoal1-v0", k = 4, delta = 0.1, max_off_iters = 30,
#           use_backtracking = 1, backtrack_coeff = 2, max_backtrack_try = 10, eps = 0, d = 25,
#           lambda_policy = 1e-5, lambda_value = 1e-5, lambda_omega = 1e-2, 
#           episode_nr = 24, step_nr = 500, epoch_nr = 2, heatmap_cmap = 'Blues', 
#           heatmap_labels = ('X', 'Y'), heatmap_interp = 'spline16',
#           seed = 0, out_path = "results/SafetyPointGoal1/MEPOL_temp")
# obj.train()

# MEPOL(env_id = "MountainCar-v0", k = 4, delta = 0.5, max_off_iters = 30,
#           use_backtracking = 1, backtrack_coeff = 2, max_backtrack_try = 10, eps = 1e-5,
#           lambda_policy = 1e-4, episode_nr = 24, step_nr = 400, epoch_nr = 300,
#           heatmap_cmap = 'Blues', 
#           heatmap_labels = ('Position', 'Velocity'), heatmap_interp = 'spline16',
#           seed = 0, out_path = "results/MountainCar/MEPOL")

# obj = MEPOL(env_id = "MountainCarContinuous-v0", k = 4, delta = 0.5, max_off_iters = 30,
#           use_backtracking = 1, backtrack_coeff = 2, max_backtrack_try = 10, eps = 1e-5,
#           lambda_policy = 1e-4, episode_nr = 24, step_nr = 400, epoch_nr = 2,
#           heatmap_cmap = 'Blues', 
#           heatmap_labels = ('Position', 'Velocity'), heatmap_interp = 'spline16',
#           seed = 0, out_path = "results/MountainCarContinuous/MEPOL_temp")
# obj.train()

# obj = CEM(env_id = "MountainCarContinuous-v0", k = 4, delta = 0.5, max_off_iters = 30,
#           use_backtracking = 1, backtrack_coeff = 2, max_backtrack_try = 10, eps = 1e-5, d = 0.5,
#           lambda_policy = 1e-4, lambda_value = 1e-4, lambda_omega = 1e-2, episode_nr = 24, 
#           step_nr = 400, epoch_nr = 300, heatmap_cmap = 'Blues', 
#           heatmap_labels = ('Position', 'Velocity'), heatmap_interp = 'spline16',
#           seed = 0, out_path = "results/MountainCarContinuous/CEM")
# obj.train()

obj = CEM(env_id = "MountainCarContinuous-v0", k = 4, delta = 0.5, max_off_iters = 30,
          use_backtracking = 1, backtrack_coeff = 2, max_backtrack_try = 10, eps = 1e-5, d = 0.5,
          lambda_policy = 1e-4, lambda_value = 1e-4, lambda_omega = 1e-2, episode_nr = 24, 
          step_nr = 400, epoch_nr = 300, heatmap_cmap = 'Blues', 
          heatmap_labels = ('Position', 'Velocity'), heatmap_interp = 'spline16',
          seed = 0, out_path = "results/MountainCarContinuous/CEM")
obj.train()

# obj = MEPOL(env_id = "MountainCarContinuous-v0", k = 4, delta = 0.5, max_off_iters = 30,
#           use_backtracking = 1, backtrack_coeff = 2, max_backtrack_try = 10, eps = 1e-5,
#           lambda_policy = 1e-4, episode_nr = 24, step_nr = 400, epoch_nr = 300,
#           heatmap_cmap = 'Blues', 
#           heatmap_labels = ('Position', 'Velocity'), heatmap_interp = 'spline16',
#           seed = 0, out_path = "results/MountainCarContinuous/MEPOL")
# obj.train()

# obj = MEPOL(env_id = "MountainCarContinuous-v0", T = 24, N = 400, heatmap_cmap = 'Blues', 
#             heatmap_labels = ('Position', 'Velocity'), heatmap_interp = 'spline16')
# obj.plot_heatmap()

# MEPOL(env_id = "CartPole-v1", k = 4, delta = 0.5, max_off_iters = 30,
#           use_backtracking = 1, backtrack_coeff = 2, max_backtrack_try = 10, eps = 1e-5,
#           lambda_policy = 1e-4, episode_nr = 24, step_nr = 300, epoch_nr = 300,
#           heatmap_cmap = 'Blues', 
#           heatmap_labels = ('Position', 'Velocity'), heatmap_interp = 'spline16',
#           seed = 0, out_path = "results/CartPole/MEPOL")