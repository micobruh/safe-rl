import safety_gymnasium
from argparse import ArgumentParser
from tqdm import tqdm, trange
from agents.CEM import PolicyNetwork, ValueNetwork
import torch

"""
Environments to test:
"SafetyPoint-v0"
"SafetyCarGoal1-v0"
"SafetyDoggo-v1"
"SafetyRacecarGoal1-v0"
"SafetyAntGoal1-v0"
"""

def parse_args():
    p = ArgumentParser(description = "Reward-Free Reinforcement Learning Trainer")
    p.add_argument("--ENV_ID", type = str, default = "SafetyCarGoal1-v0",
                   help = "Environment ID from Gymnasium Package")
    p.add_argument("--iter_nr", type = int, default = 1000,
                   help = "Number of iterations")
    p.add_argument("--step_nr", type = int, default = 1000,
                   help = "Number of steps")    
    p.add_argument("--reward_free", type = bool, default = False,
                   help = "Determine whether the simulation is reward free")
    return p.parse_args()

def main(env_id, iter_nr = 500, step_nr = 1000, reward_free = False):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    env = safety_gymnasium.make(env_id, render_mode='human')
    ep_ret, ep_cost = 0, 0
    
    for _ in trange(iter_nr):
        state, info = env.reset()
        # Set seeds
        # obs, _ = env.reset(seed=0)        
        terminated, truncated = False, False
        assert env.observation_space.contains(state)
        for _ in range(step_nr):         
            act = env.action_space.sample()
            assert env.action_space.contains(act)
            new_state, reward, cost, terminated, truncated, info = env.step(act)
            ep_cost += cost
            # Only add the reward during deployment i.e. not reward-free
            if not reward_free:
                ep_ret += reward
                if terminated or truncated:
                    break
            state = new_state

    env.close()
    return ep_ret, ep_cost

if __name__ == '__main__':
    args = parse_args()
    main(args.ENV_ID, args.iter_nr, args.step_nr, args.reward_free)