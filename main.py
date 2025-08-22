from src.agents.CEM import CEM
from src.agents.RENYI import RENYI
from src.agents.BaseAgent import BaseAgent
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.pretrain_vae import Pretrain_VAE
from pathlib import Path
import re

# env_ids = ["MountainCarContinuous-v0", "CartPole-v1"]
# out_path = f"results/{env_id[: -3]}/RENYI"
# obj = Pretrain_VAE(env_id, out_path)
# obj.train()       

safety_weights = [0]
# env_ids = ["MountainCarContinuous-v0"]
# env_ids = ["SafetyPointGoal1-v0"]
# env_ids = ["CartPole-v1"]
env_ids = ["MountainCarContinuous-v0"]
# safety_weights = [0, 10, 100, 1000]
# safety_weights = [0, 10, 100]
# state_dependent_stds = [False, True]
# zetas = [1, 1.5, 2, 2.5, 3, 3.5]

zetas = [1]
# zetas = [0.5, 1, 1.5, 2]
state_dependent_stds = [False]
alphas = [0.5]

vals = [200 for i in range(48)]
idx = 0

# best_epoch = 283

# for env_id in env_ids:
#     for state_dependent_std in state_dependent_stds:
#         for zeta in zetas:
#             for safety_weight in safety_weights:
#                 # Create folder automatically via link
#                 state_depedent_str = "_Dependent" if state_dependent_std else ""
#                 # out_path = f"results/{env_id[: -3]}/CEM/Rényi/Zeta_{zeta}/Omega_{safety_weight}{state_depedent_str}"
#                 out_path = f"results/{env_id[: -3]}/RENYI/Rényi/Zeta_{zeta}/Omega_{safety_weight}{state_depedent_str}"
#                 os.makedirs(out_path, exist_ok=True)

#                 epoch_nr = 150
#                 if env_id == "SafetyPointGoal1-v0":
#                     epoch_nr = 200
                
#                 # Train the model
#                 # obj = CEM(env_id=env_id, safety_weight=safety_weight, alpha=0, zeta=zeta, epoch_nr=epoch_nr, 
#                 #           out_path=out_path, state_dependent_std=state_dependent_std, use_behavioral=False)         
#                 obj = RENYI(env_id=env_id, safety_weight=safety_weight, alpha=0, zeta=zeta, epoch_nr=epoch_nr, 
#                             out_path=out_path, state_dependent_std=state_dependent_std, use_behavioral=False)
#                 # obj.train()
#                 # obj.compute_best_epochs()
#                 obj.visualize_policy_heatmap(51, False)

#                 # Cleanup the memory
#                 if hasattr(obj, "envs") and obj.envs is not None:
#                     try:
#                         obj.envs.close()
#                     except Exception as e:
#                         print(f"Failed to close envs: {e}")
#                 del obj
#                 gc.collect()
#                 plt.close('all')                 

for env_id in env_ids:
    for state_dependent_std in state_dependent_stds:
        for alpha in alphas:
            for safety_weight in safety_weights:
                # Create folder automatically via link
                state_depedent_str = "_Dependent" if state_dependent_std else ""
                # out_path = f"results/{env_id[: -3]}/CEM/Behavioral/Alpha_{alpha}/Omega_{safety_weight}{state_depedent_str}"
                out_path = f"results/{env_id[: -3]}/RENYI/Behavioral/Alpha_{alpha}/Omega_{safety_weight}{state_depedent_str}"
                os.makedirs(out_path, exist_ok=True)

                epoch_nr = 150
                if env_id == "SafetyPointGoal1-v0":
                    epoch_nr = 200
                
                # Train the model
                # obj = CEM(env_id=env_id, safety_weight=safety_weight, alpha=alpha, zeta=1, epoch_nr=epoch_nr, 
                #           out_path=out_path, state_dependent_std=state_dependent_std, use_behavioral=True)        
                obj = RENYI(env_id=env_id, safety_weight=safety_weight, alpha=alpha, zeta=1, epoch_nr=epoch_nr, 
                            out_path=out_path, state_dependent_std=state_dependent_std, use_behavioral=True)                    
                
                # obj.train()
                obj.visualize_policy_heatmap(59)
                # obj.compute_best_epochs()

                # Cleanup the memory
                if hasattr(obj, "envs") and obj.envs is not None:
                    try:
                        obj.envs.close()
                    except Exception as e:
                        print(f"Failed to close envs: {e}")
                del obj
                gc.collect()
                plt.close('all')                                  