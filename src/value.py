import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

int_type = torch.int64
float_type = torch.float64
torch.set_default_dtype(float_type)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, device, first_layer_dim=64, second_layer_dim=64):
        super().__init__()

        layers = []
        
        # 2 Hidden layers
        layers.extend((nn.Linear(state_dim, first_layer_dim), nn.ReLU()))
        layers.extend((nn.Linear(first_layer_dim, second_layer_dim), nn.ReLU()))
        self.net = nn.Sequential(*layers)

        # Output of the network: V value
        self.v_val = nn.Linear(second_layer_dim, 1)

        self.device = device

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize all weights using xavier uniform distribution
        """
        nn.init.xavier_uniform_(self.v_val.weight)

        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)
    
    def forward(self, x):
        v_val = self.v_val(self.net(x))
        return v_val

    def predict(self, s):
        with torch.inference_mode():
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s, dtype=float_type, device=self.device)

            # Ensure correct batch shape (do not add batch dimension if already batched)
            if s.dim() == 1:  # Single state input (state_dim,)
                s = s.unsqueeze(0)  # Convert to (1, state_dim)
            
            v_val = self(s)
            return v_val  # Shape: (N, 1) if batched, (1, 1) if single