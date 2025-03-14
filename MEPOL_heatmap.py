from src.agents.MEPOL import MEPOL_heatmap

MEPOL_heatmap(env_id = "MountainCarContinuous-v0", T = 8, N = 400, heatmap_cmap = 'Blues', 
              heatmap_labels = ('Position', 'Velocity'), heatmap_interp = 'spline16', 
              transform_fn = lambda s: [s[0], s[1]])

# 14, 15?
# 18, 19?