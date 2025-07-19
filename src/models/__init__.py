"""Neural network models for PI-HMARL"""

import torch
import torch.nn as nn
from typing import Tuple

class PhysicsInformedModel(nn.Module):
    """Physics-informed neural network model"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Value network
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        actions = self.policy(states)
        values = self.value(states)
        return actions, values
    
    def compute_physics_loss(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        """Compute physics constraint violation"""
        # Simplified physics loss - in practice this would check physical laws
        return torch.mean((next_states - states - actions * 0.1) ** 2)

class MultiAgentTransformer(nn.Module):
    """Multi-agent transformer model"""
    
    def __init__(self, num_agents: int, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=state_dim + action_dim,
                nhead=4,
                dim_feedforward=hidden_dim
            ),
            num_layers=2
        )
        
        self.output = nn.Linear(state_dim + action_dim, 1)
    
    def forward(self, agent_states: torch.Tensor, agent_actions: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Combine states and actions
        x = torch.cat([agent_states, agent_actions], dim=-1)
        
        # Transformer encoding
        encoded = self.encoder(x)
        
        # Output Q-values
        q_values = self.output(encoded).squeeze(-1)
        return q_values

__all__ = ['PhysicsInformedModel', 'MultiAgentTransformer']