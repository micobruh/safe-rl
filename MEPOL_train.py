from src.agents import MEPOL

MEPOL(env_id = "SafetyPointGoal2-v0", k = 4, delta = 0.1, max_off_iters = 30,
          use_backtracking = 1, backtrack_coeff = 2, max_backtrack_try = 10, eps = 0,
          lambda_policy = 1e-5, T = 10, N = 1000, epoch_nr = 500,
          heatmap_every = 25, heatmap_discretizer = -1, heatmap_episodes = 10, heatmap_num_steps = 1000,
          heatmap_cmap = -1, heatmap_labels = -1, heatmap_interp = -1,
          seed = 0, out_path = "results/MEPOL", num_workers = 5)