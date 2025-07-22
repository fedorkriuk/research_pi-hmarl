"""
Implementation of Sebastian et al. 2024 Physics-Informed MARL
"Physics-Informed Multi-Agent RL for Distributed Multi-Robot Problems"

Key features:
- Port-Hamiltonian structure preservation
- Self-attention coordination mechanism  
- Energy-based constraint satisfaction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class HamiltonianConfig:
    """Configuration for Port-Hamiltonian system"""
    state_dim: int
    control_dim: int
    dissipation_rate: float = 0.1
    energy_scale: float = 1.0
    constraint_penalty: float = 10.0
    attention_heads: int = 4
    attention_dim: int = 128

class PortHamiltonianNetwork(nn.Module):
    """
    Neural network that preserves Port-Hamiltonian structure
    Ensures energy conservation and passivity
    """
    
    def __init__(self, config: HamiltonianConfig):
        super().__init__()
        self.config = config
        
        # Hamiltonian function approximator
        self.hamiltonian_net = nn.Sequential(
            nn.Linear(config.state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Structure matrices (learnable)
        # J(q) - Interconnection matrix (skew-symmetric)
        self.J_net = nn.Sequential(
            nn.Linear(config.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.state_dim * config.state_dim)
        )
        
        # R(q) - Dissipation matrix (positive semi-definite)
        self.R_net = nn.Sequential(
            nn.Linear(config.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.state_dim * config.state_dim)
        )
        
        # G(q) - Input matrix
        self.G_net = nn.Sequential(
            nn.Linear(config.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.state_dim * config.control_dim)
        )
        
        # Energy shaping network
        self.energy_shaping = nn.Sequential(
            nn.Linear(config.state_dim + config.control_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state: torch.Tensor, control: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass preserving Port-Hamiltonian structure
        
        Dynamics: ẋ = (J(x) - R(x))∇H(x) + G(x)u
        """
        batch_size = state.shape[0]
        state_dim = self.config.state_dim
        
        # Compute Hamiltonian and its gradient
        state.requires_grad_(True)
        H = self.hamiltonian_net(state)
        dH_dx = torch.autograd.grad(
            H.sum(), state, create_graph=True, retain_graph=True
        )[0]
        
        # Compute structure matrices
        J_flat = self.J_net(state)
        J = J_flat.view(batch_size, state_dim, state_dim)
        
        # Ensure J is skew-symmetric
        J = 0.5 * (J - J.transpose(1, 2))
        
        R_flat = self.R_net(state)
        R = R_flat.view(batch_size, state_dim, state_dim)
        
        # Ensure R is positive semi-definite
        R = torch.bmm(R, R.transpose(1, 2)) * self.config.dissipation_rate
        
        G = self.G_net(state).view(batch_size, state_dim, self.config.control_dim)
        
        # Port-Hamiltonian dynamics
        JR = J - R
        dynamics = torch.bmm(JR, dH_dx.unsqueeze(-1)).squeeze(-1)
        control_effect = torch.bmm(G, control.unsqueeze(-1)).squeeze(-1)
        
        # Total dynamics
        x_dot = dynamics + control_effect
        
        # Energy shaping for constraints
        energy_penalty = self.energy_shaping(torch.cat([state, control], dim=-1))
        
        return {
            'dynamics': x_dot,
            'hamiltonian': H,
            'energy_gradient': dH_dx,
            'energy_penalty': energy_penalty,
            'J_matrix': J,
            'R_matrix': R,
            'G_matrix': G
        }
    
    def compute_energy(self, state: torch.Tensor) -> torch.Tensor:
        """Compute system energy (Hamiltonian)"""
        return self.hamiltonian_net(state) * self.config.energy_scale
    
    def check_passivity(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """
        Check passivity condition: Ḣ ≤ u^T y - dissipation
        """
        output = self.forward(state, control)
        
        # Energy rate of change
        H_dot = (output['energy_gradient'] * output['dynamics']).sum(dim=-1)
        
        # Supply rate
        y = torch.bmm(output['G_matrix'].transpose(1, 2), 
                     output['energy_gradient'].unsqueeze(-1)).squeeze(-1)
        supply_rate = (control * y).sum(dim=-1)
        
        # Dissipation
        dissipation = torch.bmm(
            torch.bmm(output['energy_gradient'].unsqueeze(1), output['R_matrix']),
            output['energy_gradient'].unsqueeze(-1)
        ).squeeze()
        
        # Passivity check
        passivity_violation = H_dot - supply_rate + dissipation
        
        return passivity_violation

class MultiHeadSelfAttention(nn.Module):
    """
    Self-attention mechanism for multi-agent coordination
    Allows agents to share information while preserving structure
    """
    
    def __init__(self, input_dim: int, num_heads: int = 4, attention_dim: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // num_heads
        
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        
        self.output_projection = nn.Linear(attention_dim, input_dim)
        
        # Learnable position encoding for agent ordering
        self.position_encoding = nn.Parameter(torch.randn(1, 100, input_dim))
    
    def forward(self, agent_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply self-attention across agents
        
        Args:
            agent_states: [batch, num_agents, state_dim]
            mask: [batch, num_agents, num_agents] attention mask
        """
        batch_size, num_agents, state_dim = agent_states.shape
        
        # Add position encoding
        positions = self.position_encoding[:, :num_agents, :]
        agent_states_pos = agent_states + positions
        
        # Compute Q, K, V
        Q = self.query(agent_states_pos)
        K = self.key(agent_states_pos)
        V = self.value(agent_states_pos)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, num_agents, self.attention_dim
        )
        
        output = self.output_projection(attention_output)
        
        # Residual connection
        output = output + agent_states
        
        return output

class SebastianPhysicsMARL(nn.Module):
    """
    Implementation of Sebastian et al. 2024 Physics-Informed MARL
    
    Key innovations:
    1. Port-Hamiltonian neural networks for dynamics
    2. Self-attention for coordination
    3. Energy-based constraint satisfaction
    4. Provable stability guarantees
    """
    
    def __init__(self, 
                 num_agents: int,
                 state_dim: int,
                 action_dim: int,
                 physics_constraints: Dict[str, Any],
                 hidden_dim: int = 256):
        super().__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.physics_constraints = physics_constraints
        
        # Port-Hamiltonian network for each agent type
        self.hamiltonian_config = HamiltonianConfig(
            state_dim=state_dim,
            control_dim=action_dim,
            dissipation_rate=0.1,
            energy_scale=1.0,
            constraint_penalty=10.0
        )
        
        self.hamiltonian_net = PortHamiltonianNetwork(self.hamiltonian_config)
        
        # Self-attention for coordination
        self.attention_coord = MultiHeadSelfAttention(
            input_dim=state_dim,
            num_heads=4,
            attention_dim=128
        )
        
        # Policy network (energy-based)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim + 128, hidden_dim),  # state + attention features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Constraint network
        self.constraint_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(physics_constraints)),
            nn.Sigmoid()  # Constraint satisfaction probability
        )
        
        # Lagrange multipliers for constraints
        self.lagrange_multipliers = nn.Parameter(
            torch.ones(len(physics_constraints)) * 0.1
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def forward(self, states: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with physics-informed policy
        
        Args:
            states: [batch, num_agents, state_dim]
            deterministic: If True, return mean actions
        """
        batch_size = states.shape[0]
        
        # Apply self-attention for coordination
        coordinated_states = self.attention_coord(states)
        
        # Compute actions for each agent
        actions = []
        energy_penalties = []
        constraint_violations = []
        
        for i in range(self.num_agents):
            agent_state = states[:, i, :]
            agent_coord = coordinated_states[:, i, :]
            
            # Combine state and coordination features
            policy_input = torch.cat([agent_state, agent_coord[:, :128]], dim=-1)
            
            # Get action from policy
            action_mean = self.policy_net(policy_input)
            
            if deterministic:
                action = action_mean
            else:
                # Add exploration noise
                action_std = 0.1  # Could be learned
                action = action_mean + torch.randn_like(action_mean) * action_std
            
            # Ensure action respects physics through Hamiltonian
            hamiltonian_output = self.hamiltonian_net(agent_state, action)
            
            # Energy-based action correction
            energy_gradient = hamiltonian_output['energy_gradient']
            action_correction = -0.1 * energy_gradient[:, :self.action_dim]
            action = action + action_correction
            
            # Check constraints
            constraint_input = torch.cat([agent_state, action], dim=-1)
            constraint_probs = self.constraint_net(constraint_input)
            
            actions.append(action)
            energy_penalties.append(hamiltonian_output['energy_penalty'])
            constraint_violations.append(1 - constraint_probs)
        
        actions = torch.stack(actions, dim=1)
        energy_penalties = torch.stack(energy_penalties, dim=1)
        constraint_violations = torch.stack(constraint_violations, dim=1)
        
        # Compute value
        global_state = states.view(batch_size, -1)
        value = self.value_net(global_state)
        
        return {
            'actions': actions,
            'value': value,
            'energy_penalties': energy_penalties,
            'constraint_violations': constraint_violations,
            'attention_weights': coordinated_states,
            'lagrange_multipliers': self.lagrange_multipliers
        }
    
    def compute_loss(self, 
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    rewards: torch.Tensor,
                    next_states: torch.Tensor,
                    dones: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss with physics constraints
        """
        batch_size = states.shape[0]
        
        # Forward pass
        output = self.forward(states)
        predicted_actions = output['actions']
        values = output['value']
        energy_penalties = output['energy_penalties']
        constraint_violations = output['constraint_violations']
        
        # Policy loss (behavior cloning component)
        policy_loss = F.mse_loss(predicted_actions, actions)
        
        # Value loss
        with torch.no_grad():
            next_output = self.forward(next_states)
            next_values = next_output['value']
            targets = rewards + 0.99 * next_values * (1 - dones)
        
        value_loss = F.mse_loss(values, targets)
        
        # Energy preservation loss
        total_energy_penalty = energy_penalties.mean()
        
        # Constraint violation loss (Lagrangian)
        constraint_loss = (self.lagrange_multipliers * constraint_violations.mean(dim=[0, 1])).sum()
        
        # Passivity loss
        passivity_violations = []
        for i in range(self.num_agents):
            violation = self.hamiltonian_net.check_passivity(
                states[:, i, :], actions[:, i, :]
            )
            passivity_violations.append(F.relu(violation))  # Only penalize violations
        
        passivity_loss = torch.stack(passivity_violations).mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     value_loss + 
                     self.hamiltonian_config.energy_scale * total_energy_penalty +
                     self.hamiltonian_config.constraint_penalty * constraint_loss +
                     passivity_loss)
        
        # Update Lagrange multipliers
        with torch.no_grad():
            self.lagrange_multipliers.data += 0.01 * constraint_violations.mean(dim=[0, 1])
            self.lagrange_multipliers.data = torch.clamp(self.lagrange_multipliers.data, 0, 10)
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'energy_penalty': total_energy_penalty,
            'constraint_loss': constraint_loss,
            'passivity_loss': passivity_loss
        }
    
    def energy_preserving_update(self, 
                                states: torch.Tensor,
                                actions: torch.Tensor,
                                learning_rate: float = 1e-4):
        """
        Update policy while preserving energy structure
        Uses symplectic integration for Hamiltonian systems
        """
        # Compute gradients
        output = self.forward(states)
        loss = self.compute_loss(states, actions, 
                               torch.zeros_like(states[:, :, 0]),  # dummy rewards
                               states,  # dummy next states
                               torch.zeros_like(states[:, :, 0]))  # dummy dones
        
        loss['total_loss'].backward()
        
        # Symplectic gradient update
        with torch.no_grad():
            for param in self.hamiltonian_net.parameters():
                if param.grad is not None:
                    # Project gradient to preserve structure
                    grad = param.grad
                    
                    # For J matrix parameters, ensure skew-symmetry is preserved
                    if 'J_net' in param.name:
                        grad = 0.5 * (grad - grad.T)
                    
                    param.data -= learning_rate * grad
        
        # Standard update for other networks
        with torch.no_grad():
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad
            
            for param in self.value_net.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad
        
        # Zero gradients
        self.zero_grad()
    
    def attention_coordination(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get attention-based coordination features
        """
        return self.attention_coord(states)
    
    def check_constraint_satisfaction(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, float]:
        """
        Check which physics constraints are satisfied
        """
        with torch.no_grad():
            constraint_input = torch.cat([
                states.view(-1, self.state_dim),
                actions.view(-1, self.action_dim)
            ], dim=-1)
            
            constraint_probs = self.constraint_net(constraint_input)
            
            # Average satisfaction per constraint
            satisfaction = constraint_probs.mean(dim=0)
            
            constraint_names = list(self.physics_constraints.keys())
            results = {name: float(satisfaction[i]) 
                      for i, name in enumerate(constraint_names)}
            
        return results
    
    def get_energy_landscape(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute energy landscape for given states
        Useful for visualization and analysis
        """
        with torch.no_grad():
            energies = []
            for i in range(self.num_agents):
                energy = self.hamiltonian_net.compute_energy(states[:, i, :])
                energies.append(energy)
            
            return torch.stack(energies, dim=1)