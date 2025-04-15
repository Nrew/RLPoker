# networks.py
import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    from . import config
except ImportError:
    import config

class Actor(nn.Module):
    """Actor Network (Policy)"""
    def __init__(self,
        state_dim=config.STATE_DIM,
        action_dim=config.ACTION_DIM,
        hidden_units=config.HIDDEN_UNITS):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, action_dim),
            nn.Softmax(dim=-1) # Output probabilities for discrete actions
        )

    def forward(self, state):
        return self.network(state)

    def get_action(self, state):
        """Samples action, calculates log probability."""
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

class Critic(nn.Module):
    """Critic Network (Value Function)"""
    def __init__(self,
        state_dim=config.STATE_DIM,
        hidden_units=config.HIDDEN_UNITS):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1) # Output a single value estimate
        )

    def forward(self, state):
        return self.network(state)