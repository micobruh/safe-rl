import safety_gymnasium
from argparse import ArgumentParser

def parse_args():
    p = ArgumentParser(description = "Reward-Free Reinforcement Learning Trainer")
    p.add_argument("--ENV_ID", type = str, default = "SafetyCarGoal1-v0",
                   help = "Environment ID from Gymnasium Package")
    p.add_argument("--iter_nr", type = int, default = 1000,
                   help = "Number of iterations")
    p.add_argument("--reward_free", type = bool, default = False,
                   help = "Determine whether the simulation is reward free")
    return p.parse_args()

def main(env_id, iter_nr = 1000, reward_free = False):
    env = safety_gymnasium.make(env_id, render_mode='human')
    obs, info = env.reset()
    # Set seeds
    # obs, _ = env.reset(seed=0)
    terminated, truncated = False, False
    ep_ret, ep_cost = 0, 0
    
    for _ in range(iter_nr):
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        obs, reward, cost, terminated, truncated, info = env.step(act)
        # Only add the reward during deployment i.e. not reward-free
        if not reward_free:
            ep_ret += reward
        ep_cost += cost
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    return ep_ret, ep_cost

if __name__ == '__main__':
    args = parse_args()
    main(args.ENV_ID, args.iter_nr, args.reward_free)