import torch
import gymnasium
import safety_gymnasium
from src.init_env import SafeCartPoleWrapper, SafeMountainCarWrapper
from src.policy import PolicyNetwork
import numpy as np

def vis_env(model_link, env_id):
    step_nr = 300
    if env_id == "MountainCar-v0" or env_id == "MountainCarContinuous-v0":
        envs = gymnasium.vector.make(env_id, max_episode_steps = step_nr, num_envs = 1)
        envs = SafeMountainCarWrapper(envs)
    elif env_id == "CartPole-v1":
        envs = gymnasium.vector.make(env_id, max_episode_steps = step_nr, num_envs = 1)
        envs = SafeCartPoleWrapper(envs)       
    else:
        envs = safety_gymnasium.vector.make(env_id, max_episode_steps = step_nr, num_envs = 1)

    state_dim = envs.single_state_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("\nThere is GPU")
    else:
        dev = "cpu"
        print("\nThere is no GPU")
    device = torch.device(dev)      

    policy = PolicyNetwork(state_dim, action_dim, False, device)
    policy.load_state_dict(torch.load(model_link))

    states = np.zeros((step_nr + 1, state_dim), dtype=np.float32)
    actions = np.zeros((step_nr, action_dim), dtype=np.float32)
    costs = np.zeros((step_nr, 1), dtype=np.float32)
    next_states = np.zeros((step_nr, state_dim), dtype=np.float32)

    # Reset the environments for this batch
    s, _ = envs.reset()

    for t in range(step_nr):
        # Sample action from policy
        a = policy.predict(s).cpu().numpy()

        # Record
        states[t] = s
        actions[t] = a

        # Step through environments
        s, _, cost, _, _, _ = envs.step(a)
        costs[t] = cost.reshape(-1, 1)
        next_states[t] = s

    # Final state
    states[step_nr] = s

    return states, actions, costs, next_states  

vis_env("results/MountainCarCntinuous/CEM/0-policy.pt", "MountainCarContinuous-v0")  