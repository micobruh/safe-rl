from src.agents.MEPOL import MEPOL

obj = MEPOL(env_id = "MountainCarContinuous-v0", T = 8, N = 400, heatmap_cmap = 'Blues', 
            heatmap_labels = ('Position', 'Velocity'), heatmap_interp = 'spline16')
obj.plot_heatmap()

# 14, 15?
# 18, 19?