import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

int_type = torch.int64
float_type = torch.float64
torch.set_default_dtype(float_type)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, first_layer_neuron, second_layer_neuron, state_dependent_std, is_discrete, device):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, first_layer_neuron), nn.ReLU(),
            nn.Linear(first_layer_neuron, second_layer_neuron), nn.ReLU(),
        )

        # Output of the network: Mean and log standard deviation
        self.mean_head = nn.Linear(second_layer_neuron, action_dim)
        self.state_dependent_std = state_dependent_std
        
        if self.state_dependent_std:
            # std depends on state
            self.log_std_head = nn.Linear(second_layer_neuron, action_dim)
        else:
            # single learnable parameter
            self.log_std_param = nn.Parameter(
                -0.5 * torch.ones(action_dim, dtype=float_type)
            )

        # Constant
        self.log_of_two_pi = torch.tensor(np.log(2 * np.pi), dtype = float_type)
        self.eps = 1e-7

        self.is_discrete = is_discrete
        self.device = device

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.mean_head.weight)
        if self.state_dependent_std:
            nn.init.xavier_uniform_(self.log_std_head.weight)
            nn.init.constant_(self.log_std_head.bias, -0.5)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def get_log_p(self, states, actions):
        # mean, _ = self(states)
        # if self.is_discrete:
        #     log_mean = torch.log(mean + self.eps)
        #     return log_mean.gather(1, actions).squeeze(-1)
        # else:
        #     return torch.sum(
        #         -0.5 * (
        #             self.log_of_two_pi
        #             + 2 * self.log_std
        #             + ((actions - mean) ** 2 / (torch.exp(self.log_std) + self.eps) ** 2)
        #         ), dim = 1
        #     )
        if self.is_discrete:
            logits, _ = self(states)
            log_probs = torch.log_softmax(logits, dim=-1)
            return log_probs.gather(1, actions).squeeze(-1)

        mean, log_std = self._forward_stats(states)
        std2 = torch.exp(2 * log_std)
        log_p = -0.5 * (
            self.log_of_two_pi + 2 * log_std +
            (actions - mean) ** 2 / (std2 + self.eps)
        )
        return torch.sum(log_p, dim=1)        

    def _forward_stats(self, x):
        h = self.net(x)
        mean = self.mean_head(h)

        if self.is_discrete:
            # discrete policies do not use std
            return mean, None

        if self.state_dependent_std:
            log_std = self.log_std_head(h)
            # Clamp for numerical stability
            # log_std = torch.clamp(log_std, -10.0, 2.0)
        else:
            log_std = self.log_std_param

        return mean, log_std

    def forward(self, x, deterministic=False):
        mean, log_std = self._forward_stats(x)

        if self.is_discrete:
            logits = mean
            probs = F.softmax(mean, dim=-1)
            return logits, probs

        # continuous action → sample / deterministic
        std = torch.exp(log_std)
        if deterministic:
            action = mean
        else:    
            action = mean + torch.randn_like(mean) * std

        return std, action

    def predict(self, s, deterministic=False):
        with torch.inference_mode():
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s, dtype=float_type, device=self.device)

            if s.dim() == 1:
                s = s.unsqueeze(0)

            if self.is_discrete:
                _, probs = self(s)
                if deterministic:
                    action = torch.argmax(probs, dim=-1)
                else:
                    action = torch.multinomial(probs, 1).squeeze(-1)
                return action, probs    
            
            std, action = self(s, deterministic=deterministic)
            return action, std
    

def train_supervised(envs, policy, learning_rate, device, train_steps = 20):
    """
    To make sure the initial policy network always return policy as 0
    """
    # Freeze std-related parameters (only if state-dependent)
    if policy.state_dependent_std:
        for param in policy.log_std_head.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=learning_rate)
    # optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    for _ in range(train_steps):
        optimizer.zero_grad()

        states = torch.tensor(envs.observation_space.sample(), dtype=float_type, device=device)
        
        actions = policy(states)[1]
        loss = torch.mean((actions - torch.zeros_like(actions, dtype=float_type, device=device)) ** 2)

        loss.backward()
        optimizer.step()

    return policy