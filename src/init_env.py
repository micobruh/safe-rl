import gymnasium
import safety_gymnasium
import numpy as np
from gymnasium.vector import VectorEnvWrapper

class SafeMountainCarWrapper(VectorEnvWrapper):
    def __init__(self, env, safety_threshold=-0.5):
        super().__init__(env)
        self.safety_threshold = safety_threshold

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Add a safety cost when the car goes too far left
        cost = np.zeros(self.num_envs)
        for i in range(self.num_envs):
            cost[i] = -1 if state[i][0] < self.safety_threshold else 0
        
        return state, reward, cost, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self):
        return self.env.render()
    
class SafeCartPoleWrapper(VectorEnvWrapper):
    def __init__(self, env, safety_threshold=0.8):
        super().__init__(env)
        self.safety_threshold = safety_threshold

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Add a safety cost when the car goes too far left
        cost = np.zeros(self.num_envs)
        for i in range(self.num_envs):
            cost[i] = -1 if abs(state[i][0]) > self.safety_threshold else 0
        
        return state, reward, cost, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self):
        return self.env.render()    

# Use the wrapper
def create_envs(env_id, N, T):
    if env_id == "MountainCar-v0" or env_id == "MountainCarContinuous-v0":
        envs = gymnasium.vector.make(env_id, max_episode_steps = N, num_envs = T)
        envs = SafeMountainCarWrapper(envs)
    elif env_id == "CartPole-v1":
        envs = gymnasium.vector.make(env_id, max_episode_steps = N, num_envs = T)
        envs = SafeCartPoleWrapper(envs)       
    else:
        envs = safety_gymnasium.vector.make(env_id, max_episode_steps = N, num_envs = T)
    return envs