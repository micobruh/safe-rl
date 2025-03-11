import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

int_type = torch.int64
float_type = torch.float64
torch.set_default_dtype(float_type)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, is_discrete, device):
        super().__init__()

        layers = []
        
        # 2 Hidden layers
        layers.extend((nn.Linear(state_dim, 400), nn.ReLU()))
        layers.extend((nn.Linear(400, 300), nn.ReLU()))
        self.net = nn.Sequential(*layers)

        # Output of the network: Mean and log standard deviation
        self.mean = nn.Linear(300, action_dim)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim, dtype = float_type))

        # Constant
        self.log_of_two_pi = torch.tensor(np.log(2 * np.pi), dtype = float_type)
        self.eps = 1e-7

        self.is_discrete = is_discrete
        self.device = device

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
        if self.is_discrete:
            log_mean = torch.log(mean + self.eps)
            return log_mean.gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            return torch.sum(
                -0.5 * (
                    self.log_of_two_pi
                    + 2 * self.log_std
                    + ((actions - mean) ** 2 / (torch.exp(self.log_std) + self.eps) ** 2)
                ), dim = 1
            )
    
    def forward(self, x, deterministic = False):
        mean = self.mean(self.net(x))
        if self.is_discrete:
            output = F.softmax(mean, dim = -1)
            return mean, output
        else:
            # Stochasticity allows for exploration in RL when not deterministic
            output = mean if deterministic else mean + torch.randn(mean.size(), dtype = float_type, device = self.device) * torch.exp(self.log_std)
            return mean, output

    def predict(self, s, deterministic=False):
        with torch.inference_mode():
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s, dtype=float_type, device=self.device)

            # Ensure correct batch shape (do not add batch dimension if already batched)
            if s.dim() == 1:  # Single state input (state_dim,)
                s = s.unsqueeze(0)  # Convert to (1, state_dim)
            
            action = self(s, deterministic=deterministic)[1]
            if self.is_discrete:
                action = torch.argmax(action, dim=-1) if deterministic else torch.multinomial(action, num_samples=1)  # Sample from distribution
            return action  # Shape: (N, action_dim) if batched, (1, action_dim) if single
    

def train_supervised(envs, policy, learning_rate, device, train_steps = 20):
    """
    To make sure the initial policy network always return policy as 0
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr = learning_rate)
    for _ in range(train_steps):
        optimizer.zero_grad()

        states = torch.tensor(envs.observation_space.sample(), dtype = float_type, device = device)
        
        actions = policy(states)[0]
        loss = torch.mean((actions - torch.zeros_like(actions, dtype = float_type, device = device)) ** 2)

        loss.backward()
        optimizer.step()

    return policy