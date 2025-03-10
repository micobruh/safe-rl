from src.agents.MEPOL import MEPOL_heatmap

MEPOL_heatmap(env_id = "SafetyPointGoal2-v0", T = 8, N = 1000, heatmap_cmap = 'Blues', 
              heatmap_labels = ('Goal Feature 1', 'Goal Feature 2'), heatmap_interp = 'spline16', 
              transform_fn = lambda s: [s[12], s[13]])

# 14, 15?
# 18, 19?