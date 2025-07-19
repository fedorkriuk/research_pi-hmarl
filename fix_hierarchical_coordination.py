#!/usr/bin/env python
"""
GENIUS-LEVEL FIX: Hierarchical Multi-Agent Coordination
Implementing attention-based communication and coordination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttentionCoordination(nn.Module):
    """Multi-head attention for agent coordination"""
    
    def __init__(self, d_model: int = 128, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, agent_states: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            agent_states: [batch, n_agents, d_model]
            mask: [batch, n_agents, n_agents] attention mask
        """
        batch_size, n_agents, _ = agent_states.shape
        
        # Linear transformations
        Q = self.W_q(agent_states).view(batch_size, n_agents, self.n_heads, self.d_k)
        K = self.W_k(agent_states).view(batch_size, n_agents, self.n_heads, self.d_k)
        V = self.W_v(agent_states).view(batch_size, n_agents, self.n_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, n_heads, n_agents, d_k]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, n_agents, self.d_model)
        output = self.W_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + agent_states)
        
        return output, attn_weights


class HierarchicalCoordinationModule(nn.Module):
    """Hierarchical coordination with strategic, tactical, and operational levels"""
    
    def __init__(self, state_dim: int, action_dim: int, n_agents: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = 256
        
        # Strategic level - global objective understanding
        self.strategic_encoder = nn.Sequential(
            nn.Linear(state_dim * n_agents, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        self.strategic_objective = nn.Linear(self.hidden_dim, 64)  # Global objective embedding
        
        # Tactical level - task allocation and coordination
        self.tactical_attention = MultiHeadAttentionCoordination(d_model=128, n_heads=4)
        self.tactical_encoder = nn.Sequential(
            nn.Linear(state_dim + 64, 128),  # state + strategic objective
            nn.ReLU()
        )
        
        self.task_allocator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Softmax(dim=-1)  # Task assignment probabilities
        )
        
        # Operational level - action execution
        self.operational_encoder = nn.Sequential(
            nn.Linear(state_dim + 128 + 32, 256),  # state + tactical + task
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.action_head = nn.Linear(128, action_dim)
        
        # Communication module
        self.comm_encoder = nn.Linear(state_dim, 64)
        self.comm_decoder = nn.Linear(64, 32)  # Compressed messages
        
        # Value estimation for coordination
        self.value_head = nn.Linear(128, 1)
        
    def forward(self, states: Dict[int, torch.Tensor], adjacency: Optional[torch.Tensor] = None):
        """
        Hierarchical coordination forward pass
        
        Args:
            states: Dictionary of agent states
            adjacency: Communication adjacency matrix
        """
        n_agents = len(states)
        device = next(iter(states.values())).device
        
        # Stack agent states
        agent_states = torch.stack([states[i] for i in range(n_agents)])  # [n_agents, state_dim]
        batch_states = agent_states.unsqueeze(0)  # [1, n_agents, state_dim]
        
        # Strategic level - understand global objective
        global_state = agent_states.flatten()
        strategic_features = self.strategic_encoder(global_state)
        global_objective = self.strategic_objective(strategic_features)  # [64]
        
        # Tactical level - coordinate agents
        tactical_inputs = []
        for i in range(n_agents):
            agent_tactical = torch.cat([
                agent_states[i],
                global_objective
            ])
            tactical_inputs.append(self.tactical_encoder(agent_tactical))
        
        tactical_states = torch.stack(tactical_inputs).unsqueeze(0)  # [1, n_agents, 128]
        
        # Apply attention-based coordination
        coordinated_states, attention_weights = self.tactical_attention(tactical_states, mask=adjacency)
        coordinated_states = coordinated_states.squeeze(0)  # [n_agents, 128]
        
        # Task allocation
        task_assignments = {}
        for i in range(n_agents):
            task_probs = self.task_allocator(coordinated_states[i])
            task_assignments[i] = task_probs
        
        # Operational level - generate actions
        actions = {}
        values = {}
        messages = {}
        
        for i in range(n_agents):
            operational_input = torch.cat([
                agent_states[i],
                coordinated_states[i],
                task_assignments[i]
            ])
            
            operational_features = self.operational_encoder(operational_input)
            
            # Generate action
            actions[i] = self.action_head(operational_features)
            values[i] = self.value_head(operational_features)
            
            # Generate communication message
            comm_features = self.comm_encoder(agent_states[i])
            messages[i] = self.comm_decoder(comm_features)
        
        return {
            "actions": actions,
            "values": values,
            "messages": messages,
            "task_assignments": task_assignments,
            "attention_weights": attention_weights,
            "global_objective": global_objective
        }


class CoordinationRewardShaper:
    """Shape rewards to encourage coordination"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.coordination_weight = config.get("coordination_weight", 0.3)
        self.diversity_weight = config.get("diversity_weight", 0.1)
        self.communication_weight = config.get("communication_weight", 0.1)
        
    def shape_rewards(self, 
                     base_rewards: Dict[int, float],
                     coordination_metrics: Dict[str, Any]) -> Dict[int, float]:
        """
        Add coordination bonuses to base rewards
        """
        shaped_rewards = {}
        n_agents = len(base_rewards)
        
        # Calculate coordination bonuses
        for agent_id, base_reward in base_rewards.items():
            coordination_bonus = 0.0
            
            # 1. Task diversity bonus (agents working on different tasks)
            if "task_assignments" in coordination_metrics:
                task_probs = coordination_metrics["task_assignments"]
                entropy = -torch.sum(task_probs[agent_id] * torch.log(task_probs[agent_id] + 1e-8))
                diversity_bonus = entropy.item() * self.diversity_weight
                coordination_bonus += diversity_bonus
            
            # 2. Attention-based coordination bonus
            if "attention_weights" in coordination_metrics:
                attn = coordination_metrics["attention_weights"]
                # Reward attending to other agents (not just self)
                self_attention = attn[0, :, agent_id, agent_id].mean().item()
                other_attention = 1.0 - self_attention
                coordination_bonus += other_attention * self.coordination_weight
            
            # 3. Communication effectiveness bonus
            if "messages" in coordination_metrics:
                # Reward informative messages (high variance)
                msg = coordination_metrics["messages"][agent_id]
                msg_variance = torch.var(msg).item()
                coordination_bonus += msg_variance * self.communication_weight
            
            # 4. Team success bonus (shared reward)
            team_reward = np.mean(list(base_rewards.values()))
            coordination_bonus += team_reward * 0.1
            
            shaped_rewards[agent_id] = base_reward + coordination_bonus
        
        return shaped_rewards


class PhysicsInformedCoordination:
    """Integrate physics constraints into coordination"""
    
    def __init__(self, physics_model, safety_margin: float = 1.5):
        self.physics_model = physics_model
        self.safety_margin = safety_margin
        
    def validate_coordinated_actions(self, 
                                   states: Dict[int, np.ndarray],
                                   actions: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Validate and adjust actions based on physics constraints
        """
        validated_actions = {}
        
        # Check for potential collisions
        positions = {aid: state[:3] for aid, state in states.items()}
        
        for agent_id, action in actions.items():
            # Predict next state
            next_state = self.physics_model.predict(states[agent_id], action)
            next_pos = next_state[:3]
            
            # Check collision with other agents
            safe = True
            for other_id, other_pos in positions.items():
                if other_id != agent_id:
                    distance = np.linalg.norm(next_pos - other_pos)
                    if distance < self.safety_margin:
                        safe = False
                        break
            
            if safe:
                validated_actions[agent_id] = action
            else:
                # Compute safe action
                safe_action = self._compute_collision_avoidance(
                    states[agent_id], action, positions, agent_id
                )
                validated_actions[agent_id] = safe_action
        
        return validated_actions
    
    def _compute_collision_avoidance(self, state, action, positions, agent_id):
        """Compute collision-avoiding action"""
        # Simple repulsive force from other agents
        pos = state[:3]
        avoidance_force = np.zeros(3)
        
        for other_id, other_pos in positions.items():
            if other_id != agent_id:
                diff = pos - other_pos
                distance = np.linalg.norm(diff)
                if distance < self.safety_margin * 2:
                    # Repulsive force inversely proportional to distance
                    force = diff / (distance ** 2 + 1e-6)
                    avoidance_force += force
        
        # Modify action to include avoidance
        modified_action = action.copy()
        modified_action[:3] += avoidance_force * 0.5
        
        return modified_action


if __name__ == "__main__":
    # Test the coordination module
    n_agents = 4
    state_dim = 32
    action_dim = 4
    
    # Create coordination module
    coordinator = HierarchicalCoordinationModule(state_dim, action_dim, n_agents)
    
    # Create dummy states
    states = {i: torch.randn(state_dim) for i in range(n_agents)}
    
    # Forward pass
    outputs = coordinator(states)
    
    print("GENIUS COORDINATION MODULE TEST:")
    print(f"Actions shape: {outputs['actions'][0].shape}")
    print(f"Task assignments: {outputs['task_assignments'][0]}")
    print(f"Global objective shape: {outputs['global_objective'].shape}")
    print(f"Attention weights shape: {outputs['attention_weights'].shape}")
    
    # Test reward shaping
    base_rewards = {i: np.random.rand() * 10 for i in range(n_agents)}
    shaper = CoordinationRewardShaper({"coordination_weight": 0.3})
    shaped = shaper.shape_rewards(base_rewards, outputs)
    
    print("\nReward shaping:")
    for i in range(n_agents):
        print(f"Agent {i}: Base={base_rewards[i]:.2f}, Shaped={shaped[i]:.2f}")
    
    print("\nGENIUS FIX: Hierarchical coordination with attention implemented!")