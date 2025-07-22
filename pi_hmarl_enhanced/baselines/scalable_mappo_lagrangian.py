"""
Scalable MAPPO-Lagrangian (Scal-MAPPO-L) Implementation
Combines Multi-Agent PPO with Lagrangian constraint handling for physics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from torch.distributions import Normal, Categorical

@dataclass
class ScalMAPPOConfig:
    """Configuration for Scalable MAPPO-Lagrangian"""
    num_agents: int
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    num_layers: int = 3
    use_rnn: bool = True
    rnn_hidden_dim: int = 128
    
    # PPO parameters
    clip_param: float = 0.2
    ppo_epochs: int = 10
    num_mini_batches: int = 4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Lagrangian parameters
    constraint_threshold: float = 0.1
    lagrange_lr: float = 0.01
    max_lagrange: float = 10.0
    
    # Scalability parameters
    communication_rounds: int = 2
    neighbor_radius: float = 10.0
    message_dim: int = 64

class ScalableAttentionModule(nn.Module):
    """
    Scalable attention mechanism for large agent populations
    Uses sparse attention based on proximity
    """
    
    def __init__(self, input_dim: int, message_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Message generation
        self.message_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )
        
        # Message aggregation
        self.message_aggregator = nn.Sequential(
            nn.Linear(message_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(input_dim + message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, 
                agent_states: torch.Tensor,
                agent_positions: torch.Tensor,
                neighbor_radius: float) -> torch.Tensor:
        """
        Scalable attention based on spatial proximity
        
        Args:
            agent_states: [batch, num_agents, state_dim]
            agent_positions: [batch, num_agents, pos_dim]
            neighbor_radius: Communication radius
        """
        batch_size, num_agents, state_dim = agent_states.shape
        
        # Generate messages
        messages = self.message_encoder(agent_states)
        
        # Compute pairwise distances
        pos_expanded_i = agent_positions.unsqueeze(2)  # [batch, num_agents, 1, pos_dim]
        pos_expanded_j = agent_positions.unsqueeze(1)  # [batch, 1, num_agents, pos_dim]
        distances = torch.norm(pos_expanded_i - pos_expanded_j, dim=-1)  # [batch, num_agents, num_agents]
        
        # Create adjacency mask based on radius
        adj_mask = (distances < neighbor_radius).float()
        adj_mask = adj_mask * (1 - torch.eye(num_agents).unsqueeze(0).to(adj_mask.device))
        
        # Normalize by number of neighbors
        num_neighbors = adj_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_mask = adj_mask / num_neighbors
        
        # Aggregate messages from neighbors
        aggregated_messages = []
        for i in range(num_agents):
            # Get messages from neighbors
            neighbor_messages = messages * adj_mask[:, i:i+1, :].transpose(1, 2)
            aggregated = neighbor_messages.sum(dim=1)
            
            # Combine with own message
            combined = torch.cat([messages[:, i], aggregated], dim=-1)
            final_message = self.message_aggregator(combined)
            aggregated_messages.append(final_message)
        
        aggregated_messages = torch.stack(aggregated_messages, dim=1)
        
        # Combine with original states
        combined_features = torch.cat([agent_states, aggregated_messages], dim=-1)
        output = self.output_projection(combined_features)
        
        # Residual connection
        return agent_states + output

class ScalableMappoLagrangian(nn.Module):
    """
    Scalable MAPPO with Lagrangian constraint handling
    
    Features:
    - Scalable to 50+ agents
    - Lagrangian method for physics constraints
    - Efficient communication through sparse attention
    - RNN support for partial observability
    """
    
    def __init__(self, config: ScalMAPPOConfig, physics_constraints: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.physics_constraints = physics_constraints
        self.num_constraints = len(physics_constraints)
        
        # Actor network (decentralized execution)
        self.actor = self._build_actor_network()
        
        # Critic network (centralized training)
        self.critic = self._build_critic_network()
        
        # Scalable attention for communication
        self.attention = ScalableAttentionModule(
            input_dim=config.state_dim,
            message_dim=config.message_dim,
            hidden_dim=config.hidden_dim
        )
        
        # Constraint prediction network
        self.constraint_net = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, self.num_constraints)
        )
        
        # Lagrange multipliers
        self.lagrange_multipliers = nn.Parameter(
            torch.zeros(self.num_constraints)
        )
        
        # RNN for partial observability
        if config.use_rnn:
            self.rnn = nn.GRU(
                config.state_dim,
                config.rnn_hidden_dim,
                batch_first=True
            )
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _build_actor_network(self) -> nn.Module:
        """Build actor network for each agent"""
        layers = []
        input_dim = self.config.state_dim
        
        if self.config.use_rnn:
            input_dim += self.config.rnn_hidden_dim
        
        for i in range(self.config.num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(input_dim, self.config.hidden_dim),
                    nn.LayerNorm(self.config.hidden_dim),
                    nn.ReLU()
                ])
            else:
                layers.extend([
                    nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                    nn.LayerNorm(self.config.hidden_dim),
                    nn.ReLU()
                ])
        
        # Output mean and log_std
        layers.append(nn.Linear(self.config.hidden_dim, self.config.action_dim * 2))
        
        return nn.Sequential(*layers)
    
    def _build_critic_network(self) -> nn.Module:
        """Build centralized critic network"""
        # Takes global state as input
        input_dim = self.config.state_dim * self.config.num_agents
        
        layers = []
        for i in range(self.config.num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(input_dim, self.config.hidden_dim * 2),
                    nn.LayerNorm(self.config.hidden_dim * 2),
                    nn.ReLU()
                ])
            else:
                layers.extend([
                    nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim * 2),
                    nn.LayerNorm(self.config.hidden_dim * 2),
                    nn.ReLU()
                ])
        
        layers.append(nn.Linear(self.config.hidden_dim * 2, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, 
                states: torch.Tensor,
                rnn_hidden: Optional[torch.Tensor] = None,
                positions: Optional[torch.Tensor] = None,
                deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with scalable communication
        
        Args:
            states: [batch, num_agents, state_dim]
            rnn_hidden: [num_agents, batch, rnn_hidden_dim]
            positions: [batch, num_agents, pos_dim] for proximity-based communication
            deterministic: If True, return mean actions
        """
        batch_size, num_agents, state_dim = states.shape
        
        # Extract positions if not provided
        if positions is None:
            # Assume positions are first 2-3 dimensions of state
            positions = states[:, :, :3]
        
        # Apply scalable attention for communication
        communicated_states = states
        for _ in range(self.config.communication_rounds):
            communicated_states = self.attention(
                communicated_states, positions, self.config.neighbor_radius
            )
        
        # Process through RNN if enabled
        if self.config.use_rnn and rnn_hidden is not None:
            rnn_input = communicated_states.transpose(0, 1)  # [num_agents, batch, state_dim]
            rnn_output = []
            new_hidden = []
            
            for i in range(num_agents):
                output, hidden = self.rnn(
                    rnn_input[i:i+1],
                    rnn_hidden[i:i+1] if rnn_hidden is not None else None
                )
                rnn_output.append(output)
                new_hidden.append(hidden)
            
            rnn_output = torch.cat(rnn_output, dim=0).transpose(0, 1)
            new_hidden = torch.cat(new_hidden, dim=0)
            
            actor_input = torch.cat([communicated_states, rnn_output], dim=-1)
        else:
            actor_input = communicated_states
            new_hidden = None
        
        # Get actions from actor
        actions = []
        log_probs = []
        
        for i in range(num_agents):
            actor_output = self.actor(actor_input[:, i])
            
            # Split mean and log_std
            action_mean = actor_output[:, :self.config.action_dim]
            action_log_std = actor_output[:, self.config.action_dim:]
            action_std = torch.exp(action_log_std.clamp(-20, 2))
            
            # Sample actions
            if deterministic:
                action = action_mean
            else:
                dist = Normal(action_mean, action_std)
                action = dist.rsample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                log_probs.append(log_prob)
            
            # Bound actions
            action = torch.tanh(action)
            actions.append(action)
        
        actions = torch.stack(actions, dim=1)
        
        if not deterministic:
            log_probs = torch.stack(log_probs, dim=1)
        else:
            log_probs = None
        
        # Get value from critic
        global_state = states.view(batch_size, -1)
        value = self.critic(global_state)
        
        # Check constraints
        constraint_violations = self._check_constraints(states, actions)
        
        return {
            'actions': actions,
            'log_probs': log_probs,
            'value': value,
            'rnn_hidden': new_hidden,
            'constraint_violations': constraint_violations,
            'lagrange_multipliers': self.lagrange_multipliers
        }
    
    def _check_constraints(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Check physics constraint violations"""
        batch_size, num_agents = states.shape[:2]
        
        violations = []
        for i in range(num_agents):
            state_action = torch.cat([states[:, i], actions[:, i]], dim=-1)
            constraint_values = self.constraint_net(state_action)
            violations.append(constraint_values)
        
        violations = torch.stack(violations, dim=1)
        
        # Convert to violations (positive when violated)
        violations = F.relu(violations - self.config.constraint_threshold)
        
        return violations
    
    def compute_loss(self,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    old_log_probs: torch.Tensor,
                    advantages: torch.Tensor,
                    returns: torch.Tensor,
                    rnn_hidden: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute MAPPO loss with Lagrangian constraints
        """
        # Forward pass
        output = self.forward(states, rnn_hidden)
        new_log_probs = output['log_probs']
        values = output['value']
        constraint_violations = output['constraint_violations']
        
        # PPO actor loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(-1), returns)
        
        # Entropy bonus
        entropy = -(new_log_probs * torch.exp(new_log_probs)).mean()
        
        # Lagrangian constraint loss
        weighted_violations = constraint_violations * self.lagrange_multipliers.unsqueeze(0).unsqueeze(0)
        constraint_loss = weighted_violations.sum() / (constraint_violations.shape[0] * constraint_violations.shape[1])
        
        # Total loss
        total_loss = (actor_loss + 
                     self.config.value_loss_coef * value_loss - 
                     self.config.entropy_coef * entropy +
                     constraint_loss)
        
        # Update Lagrange multipliers
        with torch.no_grad():
            avg_violations = constraint_violations.mean(dim=[0, 1])
            self.lagrange_multipliers.data += self.config.lagrange_lr * avg_violations
            self.lagrange_multipliers.data = torch.clamp(
                self.lagrange_multipliers.data, 0, self.config.max_lagrange
            )
        
        return {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'constraint_loss': constraint_loss,
            'avg_violations': avg_violations
        }
    
    def update(self,
              rollout_buffer: Dict[str, torch.Tensor],
              num_epochs: Optional[int] = None) -> Dict[str, float]:
        """
        Update policy using PPO with multiple epochs
        """
        num_epochs = num_epochs or self.config.ppo_epochs
        batch_size = rollout_buffer['states'].shape[0]
        
        # Normalize advantages
        advantages = rollout_buffer['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training stats
        stats = {
            'actor_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'constraint_loss': 0,
            'avg_violations': torch.zeros(self.num_constraints)
        }
        
        num_updates = 0
        
        for epoch in range(num_epochs):
            # Create random mini-batches
            indices = torch.randperm(batch_size)
            
            for start_idx in range(0, batch_size, batch_size // self.config.num_mini_batches):
                end_idx = min(start_idx + batch_size // self.config.num_mini_batches, batch_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = rollout_buffer['states'][batch_indices]
                batch_actions = rollout_buffer['actions'][batch_indices]
                batch_old_log_probs = rollout_buffer['log_probs'][batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = rollout_buffer['returns'][batch_indices]
                
                # Compute loss
                losses = self.compute_loss(
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_returns
                )
                
                # Backward pass
                losses['total_loss'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), self.config.max_grad_norm
                )
                
                # Optimizer step (assumed to be set externally)
                # optimizer.step()
                # optimizer.zero_grad()
                
                # Update stats
                for key in stats:
                    if key != 'avg_violations':
                        stats[key] += losses[key].item()
                    else:
                        stats[key] += losses[key].detach()
                
                num_updates += 1
        
        # Average stats
        for key in stats:
            if key != 'avg_violations':
                stats[key] /= num_updates
            else:
                stats[key] /= num_updates
        
        return stats
    
    def get_action(self,
                  state: torch.Tensor,
                  rnn_hidden: Optional[torch.Tensor] = None,
                  deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action for a single timestep (used during rollout)
        """
        with torch.no_grad():
            output = self.forward(state, rnn_hidden, deterministic=deterministic)
            
        return output['actions'], output['log_probs'], output['rnn_hidden']
    
    def evaluate_actions(self,
                        states: torch.Tensor,
                        actions: torch.Tensor,
                        rnn_hidden: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Evaluate actions for PPO update
        """
        # Get actor output
        output = self.forward(states, rnn_hidden)
        
        # Compute log probabilities of taken actions
        batch_size, num_agents = actions.shape[:2]
        log_probs = []
        
        for i in range(num_agents):
            actor_output = self.actor(states[:, i])
            action_mean = actor_output[:, :self.config.action_dim]
            action_log_std = actor_output[:, self.config.action_dim:]
            action_std = torch.exp(action_log_std.clamp(-20, 2))
            
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(actions[:, i]).sum(dim=-1)
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs, dim=1)
        
        return {
            'log_probs': log_probs,
            'values': output['value'],
            'constraint_violations': output['constraint_violations']
        }