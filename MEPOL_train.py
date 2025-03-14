from src.agents.MEPOL import MEPOL

# MEPOL(env_id = "SafetyPointGoal2-v0", k = 4, delta = 0.1, max_off_iters = 30,
#           use_backtracking = 1, backtrack_coeff = 2, max_backtrack_try = 10, eps = 0,
#           lambda_policy = 1e-5, T = 10, N = 1000, epoch_nr = 500,
#           heatmap_every = 25, heatmap_discretizer = -1,
#           heatmap_cmap = -1, heatmap_labels = -1, heatmap_interp = -1,
#           seed = 0, out_path = "results/MEPOL", num_workers = 5)

MEPOL(env_id = "SafetyPointGoal1-v0", k = 4, delta = 0.1, max_off_iters = 30,
          use_backtracking = 1, backtrack_coeff = 2, max_backtrack_try = 10, eps = 1e-5,
          lambda_policy = 1e-5, T = 8, N = 500, epoch_nr = 500,
          heatmap_every = 25, heatmap_cmap = 'Blues', 
          heatmap_labels = ('X', 'Y'), heatmap_interp = 'spline16',
          seed = 0, out_path = "results/SafetyPointGoal/MEPOL")

# MEPOL(env_id = "MountainCar-v0", k = 4, delta = 0.5, max_off_iters = 30,
#           use_backtracking = 1, backtrack_coeff = 2, max_backtrack_try = 10, eps = 1e-5,
#           lambda_policy = 1e-4, T = 8, N = 400, epoch_nr = 300,
#           heatmap_every = 25, heatmap_cmap = 'Blues', 
#           heatmap_labels = ('Position', 'Velocity'), heatmap_interp = 'spline16',
#           seed = 0, out_path = "results/MountainCar/MEPOL")

# MEPOL(env_id = "MountainCarContinuous-v0", k = 4, delta = 0.5, max_off_iters = 30,
#           use_backtracking = 1, backtrack_coeff = 2, max_backtrack_try = 10, eps = 1e-5,
#           lambda_policy = 1e-4, T = 8, N = 400, epoch_nr = 300,
#           heatmap_every = 25, heatmap_cmap = 'Blues', 
#           heatmap_labels = ('Position', 'Velocity'), heatmap_interp = 'spline16',
#           seed = 0, out_path = "results/MountainCarContinuous/MEPOL")

# MEPOL(env_id = "CartPole-v1", k = 4, delta = 0.5, max_off_iters = 30,
#           use_backtracking = 1, backtrack_coeff = 2, max_backtrack_try = 10, eps = 1e-5,
#           lambda_policy = 1e-4, T = 8, N = 300, epoch_nr = 300,
#           heatmap_every = 25, heatmap_cmap = 'Blues', 
#           heatmap_labels = ('Position', 'Velocity'), heatmap_interp = 'spline16',
#           seed = 0, out_path = "results/CartPole/MEPOL")