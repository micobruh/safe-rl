import torch
import torch.nn as nn
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        layers = []
        
        # 2 Hidden layers
        layers.extend((nn.Linear(state_dim, 400), nn.ReLU()))
        layers.extend((nn.Linear(400, 300), nn.ReLU()))
        self.net = nn.Sequential(*layers)

        # Output of the network: Mean and log standard deviation
        self.mean = nn.Linear(300, action_dim)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim, dtype = torch.float64))

        # Constant
        self.log_of_two_pi = torch.tensor(np.log(2 * np.pi), dtype = torch.float64)
        self.eps = 1e-7

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize all weights using xavier uniform distribution
        """
        nn.init.xavier_uniform_(self.mean.weight)

        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)

    def get_log_p(self, states, actions):
        mean, _ = self(states)
        return torch.sum(
            -0.5 * (
                self.log_of_two_pi
                + 2 * self.log_std
                + ((actions - mean) ** 2 / (torch.exp(self.log_std) + self.eps) ** 2)
            ), dim = 1
        )
    
    def forward(self, x, deterministic = False):
        mean = self.mean(self.net(x))
        # Stochasticity allows for exploration in RL when not deterministic
        output = mean if deterministic else mean + torch.randn(mean.size(), dtype = torch.float64) * torch.exp(self.log_std)
        return mean, output

    def predict(self, s, deterministic = False):
        with torch.no_grad():
            s = torch.tensor(s, dtype = torch.float64).unsqueeze(0)
            return self(s, deterministic = deterministic)[1][0]        

def train_supervised(env, policy, optimizer, train_steps = 20, batch_size = 1000):
    """
    To make sure the initial policy network always return policy as 0
    """
    for _ in range(train_steps):
        optimizer.zero_grad()

        states = torch.tensor([env.observation_space.sample() for _ in range(batch_size)], dtype = torch.float64)

        actions = policy(states)[0]
        loss = torch.mean((actions - torch.zeros_like(actions, dtype = torch.float64)) ** 2)

        loss.backward()
        optimizer.step()

    return policy