import gymnasium
import safety_gymnasium
import numpy as np
from gymnasium.vector import VectorEnvWrapper

class SafeMountainCarWrapper(VectorEnvWrapper):
    def __init__(self, env, safety_threshold=-0.9):
        super().__init__(env)
        self.safety_threshold = safety_threshold

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Add a safety cost when the car goes too far left
        cost = np.where(state[:, 0] < self.safety_threshold, 1, 0)
        
        return state, reward, cost, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self):
        return self.env.render()
    
    def get_safety_threshold(self):
        return self.safety_threshold    
    
class SafePendulumWrapper(VectorEnvWrapper):
    def __init__(self, env, safety_threshold_1=0, safety_threshold_2=2):
        super().__init__(env)      
        self.safety_threshold_1 = safety_threshold_1
        self.safety_threshold_2 = safety_threshold_2

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Add a safety cost when the pendulum swings too low and its angular velocity is too fast
        cost = np.where((state[:, 0] < self.safety_threshold_1) & (np.abs(state[:, 2]) > self.safety_threshold_2), 1, 0)
        
        return state, reward, cost, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self):
        return self.env.render()    
    
    def get_safety_threshold(self):
        return self.safety_threshold_1, self.safety_threshold_2       

class SafeCartPoleWrapper(VectorEnvWrapper):
    def __init__(self, env, safety_threshold=0.8):
        super().__init__(env)    
        self.safety_threshold = safety_threshold

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Add a safety cost when the car goes beyond the bounded region
        cost = np.where(abs(state[:, 0]) > self.safety_threshold, 1, 0)
        
        return state, reward, cost, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self):
        return self.env.render()     
    
    def get_safety_threshold(self):
        return self.safety_threshold

# Use the wrapper
def create_envs(env_id, N, T):
    if env_id == "MountainCar-v0" or env_id == "MountainCarContinuous-v0":
        envs = gymnasium.vector.make(env_id, max_episode_steps = N, num_envs = T)
        envs = SafeMountainCarWrapper(envs)
    elif env_id == "Pendulum-v1":
        envs = gymnasium.vector.make(env_id, max_episode_steps = N, num_envs = T)
        envs = SafePendulumWrapper(envs)               
    elif env_id == "CartPole-v1":
        envs = gymnasium.vector.make(env_id, max_episode_steps = N, num_envs = T)
        envs = SafeCartPoleWrapper(envs)       
    else:
        envs = safety_gymnasium.vector.make(env_id, max_episode_steps = N, num_envs = T)
    return envs