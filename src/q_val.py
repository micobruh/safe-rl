import torch
import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        layers = []
        
        # 2 Hidden layers
        layers.extend((nn.Linear(state_dim + action_dim, 64), nn.ReLU()))
        layers.extend((nn.Linear(64, 64), nn.ReLU()))
        self.net = nn.Sequential(*layers)

        # Output of the network: V value
        self.v_val = nn.Linear(64, 1)  # Single output for value estimate

    def initialize_weights(self):
        """
        Initialize all weights using xavier uniform distribution
        """
        nn.init.xavier_uniform_(self.mean.weight)

        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim = -1)  # Concatenate (s, a)
        return self.net(sa)  # Output Q(s, a)
    
    def predict(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype = torch.float64).unsqueeze(0)
            return self(s)[0]