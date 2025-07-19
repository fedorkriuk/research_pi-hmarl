"""Physics-Aware Attention Mechanism

This module implements attention mechanisms that consider physical constraints,
spatial relationships, and energy efficiency based on real-world parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import logging

from .base_attention import BaseMultiHeadAttention

logger = logging.getLogger(__name__)


class PhysicsAwareAttention(nn.Module):
    """Attention mechanism that incorporates physics constraints"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_agents: int,
        communication_range: float = 100.0,  # meters
        min_safety_distance: float = 5.0,    # meters
        max_velocity: float = 20.0,          # m/s
        energy_weight: float = 0.1,
        collision_weight: float = 0.3,
        dropout: float = 0.1
    ):
        """Initialize physics-aware attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_agents: Maximum number of agents
            communication_range: Maximum communication range
            min_safety_distance: Minimum safety distance
            max_velocity: Maximum velocity for dynamics
            energy_weight: Weight for energy efficiency
            collision_weight: Weight for collision avoidance
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_agents = num_agents
        self.communication_range = communication_range
        self.min_safety_distance = min_safety_distance
        self.max_velocity = max_velocity
        self.energy_weight = energy_weight
        self.collision_weight = collision_weight
        
        # Base attention mechanism
        self.base_attention = BaseMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Physics-based feature extractors
        self.distance_encoder = DistanceEncoder(embed_dim)
        self.velocity_encoder = VelocityEncoder(embed_dim)
        self.energy_encoder = EnergyEncoder(embed_dim)
        
        # Constraint processors
        self.collision_processor = CollisionConstraintProcessor(
            min_safety_distance=min_safety_distance
        )
        self.communication_processor = CommunicationConstraintProcessor(
            communication_range=communication_range
        )
        
        # Physics-aware projection
        self.physics_projection = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        logger.info(f"Initialized PhysicsAwareAttention with range={communication_range}m")
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        energy_levels: Optional[torch.Tensor] = None,
        return_physics_info: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with physics constraints
        
        Args:
            x: Agent features [batch_size, num_agents, embed_dim]
            positions: Agent positions [batch_size, num_agents, 3]
            velocities: Agent velocities [batch_size, num_agents, 3]
            energy_levels: Energy levels [batch_size, num_agents]
            return_physics_info: Whether to return physics information
            
        Returns:
            Output features and attention info
        """
        batch_size, num_agents, _ = x.size()
        device = x.device
        
        # Compute distance matrix
        distance_matrix = self._compute_distance_matrix(positions)
        
        # Encode physics features
        distance_features = self.distance_encoder(distance_matrix)
        
        velocity_features = torch.zeros(batch_size, num_agents, self.embed_dim, device=device)
        if velocities is not None:
            velocity_features = self.velocity_encoder(velocities, distance_matrix)
        
        energy_features = torch.zeros(batch_size, num_agents, self.embed_dim, device=device)
        if energy_levels is not None:
            energy_features = self.energy_encoder(energy_levels, distance_matrix)
        
        # Combine features
        physics_features = torch.cat([
            x,
            distance_features,
            velocity_features,
            energy_features
        ], dim=-1)
        
        # Project to attention dimension
        physics_aware_features = self.physics_projection(physics_features)
        
        # Compute attention masks based on physics
        comm_mask = self.communication_processor(distance_matrix)
        collision_mask = self.collision_processor(
            distance_matrix, velocities if velocities is not None else None
        )
        
        # Combine masks
        physics_mask = self._combine_masks(comm_mask, collision_mask)
        
        # Apply attention with physics constraints
        output, attention_weights = self.base_attention(
            physics_aware_features,
            physics_aware_features,
            physics_aware_features,
            attn_mask=physics_mask
        )
        
        # Prepare output info
        info = {
            "attention_weights": attention_weights,
            "communication_mask": comm_mask,
            "collision_mask": collision_mask,
            "distance_matrix": distance_matrix
        }
        
        if return_physics_info:
            info["physics_features"] = {
                "distance": distance_features,
                "velocity": velocity_features,
                "energy": energy_features
            }
        
        return output, info
    
    def _compute_distance_matrix(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix
        
        Args:
            positions: Agent positions [batch_size, num_agents, 3]
            
        Returns:
            Distance matrix [batch_size, num_agents, num_agents]
        """
        # Expand dimensions for broadcasting
        pos1 = positions.unsqueeze(2)  # [batch, agents, 1, 3]
        pos2 = positions.unsqueeze(1)  # [batch, 1, agents, 3]
        
        # Compute Euclidean distance
        distances = torch.norm(pos1 - pos2, dim=-1)
        
        return distances
    
    def _combine_masks(
        self,
        comm_mask: torch.Tensor,
        collision_mask: torch.Tensor
    ) -> torch.Tensor:
        """Combine communication and collision masks
        
        Args:
            comm_mask: Communication feasibility mask
            collision_mask: Collision avoidance mask
            
        Returns:
            Combined attention mask
        """
        # Combine with weights
        combined = (1 - self.collision_weight) * comm_mask + \
                  self.collision_weight * collision_mask
        
        # Convert to attention mask format (0 = attend, -inf = mask)
        attention_mask = torch.where(
            combined > 0.5,
            torch.zeros_like(combined),
            torch.full_like(combined, float('-inf'))
        )
        
        # Add head dimension
        attention_mask = attention_mask.unsqueeze(1)
        
        return attention_mask


class DistanceEncoder(nn.Module):
    """Encodes distance information into features"""
    
    def __init__(self, embed_dim: int):
        """Initialize distance encoder
        
        Args:
            embed_dim: Output embedding dimension
        """
        super().__init__()
        
        # Distance embedding network
        self.distance_net = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Relative position encoder
        self.position_net = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
    
    def forward(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        """Encode distances
        
        Args:
            distance_matrix: Pairwise distances [batch, agents, agents]
            
        Returns:
            Distance features [batch, agents, embed_dim]
        """
        batch_size, num_agents, _ = distance_matrix.size()
        
        # Average distance to other agents
        avg_distance = distance_matrix.sum(dim=-1) / (num_agents - 1)
        avg_distance = avg_distance.unsqueeze(-1)
        
        # Encode average distance
        distance_features = self.distance_net(avg_distance)
        
        return distance_features


class VelocityEncoder(nn.Module):
    """Encodes velocity and dynamics information"""
    
    def __init__(self, embed_dim: int):
        """Initialize velocity encoder
        
        Args:
            embed_dim: Output embedding dimension
        """
        super().__init__()
        
        self.velocity_net = nn.Sequential(
            nn.Linear(6, embed_dim),  # velocity + relative velocity
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(
        self,
        velocities: torch.Tensor,
        distance_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Encode velocity information
        
        Args:
            velocities: Agent velocities [batch, agents, 3]
            distance_matrix: Pairwise distances
            
        Returns:
            Velocity features [batch, agents, embed_dim]
        """
        batch_size, num_agents, _ = velocities.size()
        
        # Compute relative velocities to nearest neighbors
        _, nearest_indices = distance_matrix.topk(4, dim=-1, largest=False)
        
        velocity_features = []
        for b in range(batch_size):
            agent_features = []
            for i in range(num_agents):
                # Own velocity
                own_vel = velocities[b, i]
                
                # Average relative velocity to nearest neighbors
                neighbors = nearest_indices[b, i, 1:]  # Exclude self
                neighbor_vels = velocities[b, neighbors]
                rel_vel = (own_vel - neighbor_vels).mean(dim=0)
                
                # Combine features
                combined = torch.cat([own_vel, rel_vel])
                agent_features.append(combined)
            
            velocity_features.append(torch.stack(agent_features))
        
        velocity_features = torch.stack(velocity_features)
        
        # Encode
        encoded = self.velocity_net(velocity_features)
        
        return encoded


class EnergyEncoder(nn.Module):
    """Encodes energy and resource information"""
    
    def __init__(self, embed_dim: int):
        """Initialize energy encoder
        
        Args:
            embed_dim: Output embedding dimension
        """
        super().__init__()
        
        self.energy_net = nn.Sequential(
            nn.Linear(3, embed_dim // 2),  # energy + comm cost + motion cost
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Communication cost model (based on real radio power consumption)
        self.comm_cost_per_meter = 0.001  # Watts per meter
        self.base_comm_cost = 0.1         # Base power consumption
    
    def forward(
        self,
        energy_levels: torch.Tensor,
        distance_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Encode energy information
        
        Args:
            energy_levels: Current energy levels [batch, agents]
            distance_matrix: Pairwise distances
            
        Returns:
            Energy features [batch, agents, embed_dim]
        """
        batch_size, num_agents = energy_levels.size()
        
        # Compute communication energy costs
        comm_costs = self.base_comm_cost + self.comm_cost_per_meter * distance_matrix
        avg_comm_cost = comm_costs.mean(dim=-1)
        
        # Estimate motion energy cost (simplified)
        motion_cost = 0.5 * torch.ones_like(energy_levels)
        
        # Combine features
        energy_features = torch.stack([
            energy_levels,
            avg_comm_cost,
            motion_cost
        ], dim=-1)
        
        # Encode
        encoded = self.energy_net(energy_features)
        
        return encoded


class CollisionConstraintProcessor(nn.Module):
    """Processes collision avoidance constraints"""
    
    def __init__(self, min_safety_distance: float = 5.0):
        """Initialize collision processor
        
        Args:
            min_safety_distance: Minimum safety distance in meters
        """
        super().__init__()
        
        self.min_safety_distance = min_safety_distance
        
        # Learned safety margin adjustment
        self.safety_margin_net = nn.Sequential(
            nn.Linear(6, 32),  # distance + relative velocity
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        distance_matrix: torch.Tensor,
        velocities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate collision avoidance mask
        
        Args:
            distance_matrix: Pairwise distances
            velocities: Agent velocities
            
        Returns:
            Collision mask (1 = safe, 0 = potential collision)
        """
        batch_size, num_agents, _ = distance_matrix.size()
        
        # Base safety check
        safe_mask = (distance_matrix > self.min_safety_distance).float()
        
        # Adjust for velocities if provided
        if velocities is not None:
            # Compute time to collision (simplified)
            vel_magnitude = torch.norm(velocities, dim=-1, keepdim=True)
            
            # Estimate collision risk
            collision_time = distance_matrix / (vel_magnitude + 1e-6)
            
            # Agents on collision course need larger safety margin
            dynamic_margin = 1.0 + 2.0 * torch.exp(-collision_time / 5.0)
            adjusted_safety_distance = self.min_safety_distance * dynamic_margin
            
            # Update mask
            safe_mask = (distance_matrix > adjusted_safety_distance).float()
        
        # Ensure self-attention is always allowed
        eye = torch.eye(num_agents, device=distance_matrix.device)
        safe_mask = safe_mask + eye.unsqueeze(0)
        safe_mask = torch.clamp(safe_mask, 0, 1)
        
        return safe_mask


class CommunicationConstraintProcessor(nn.Module):
    """Processes communication range constraints"""
    
    def __init__(self, communication_range: float = 100.0):
        """Initialize communication processor
        
        Args:
            communication_range: Maximum communication range in meters
        """
        super().__init__()
        
        self.communication_range = communication_range
        
        # Learned communication quality model
        self.comm_quality_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        """Generate communication feasibility mask
        
        Args:
            distance_matrix: Pairwise distances
            
        Returns:
            Communication mask (1 = can communicate, 0 = out of range)
        """
        # Base range check
        in_range = (distance_matrix <= self.communication_range).float()
        
        # Model signal quality degradation
        normalized_distance = distance_matrix / self.communication_range
        signal_quality = self.comm_quality_net(normalized_distance.unsqueeze(-1)).squeeze(-1)
        
        # Combine range and quality
        comm_mask = in_range * signal_quality
        
        # Ensure self-communication is always perfect
        batch_size, num_agents, _ = distance_matrix.size()
        eye = torch.eye(num_agents, device=distance_matrix.device)
        comm_mask = comm_mask + eye.unsqueeze(0)
        comm_mask = torch.clamp(comm_mask, 0, 1)
        
        return comm_mask


class EnergyAwareAttention(PhysicsAwareAttention):
    """Extension of physics-aware attention with energy optimization"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_agents: int,
        battery_capacity: float = 5000.0,  # mAh
        critical_battery_level: float = 0.2,
        **kwargs
    ):
        """Initialize energy-aware attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_agents: Maximum number of agents
            battery_capacity: Battery capacity in mAh
            critical_battery_level: Critical battery threshold
            **kwargs: Additional arguments for parent class
        """
        super().__init__(embed_dim, num_heads, num_agents, **kwargs)
        
        self.battery_capacity = battery_capacity
        self.critical_battery_level = critical_battery_level
        
        # Energy priority network
        self.energy_priority_net = nn.Sequential(
            nn.Linear(2, 32),  # energy level + distance
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        energy_levels: Optional[torch.Tensor] = None,
        return_physics_info: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with energy-aware attention
        
        Args:
            x: Agent features
            positions: Agent positions
            velocities: Agent velocities
            energy_levels: Energy levels (0-1)
            return_physics_info: Whether to return physics info
            
        Returns:
            Output features and attention info
        """
        # Get base physics-aware attention
        output, info = super().forward(
            x, positions, velocities, energy_levels, return_physics_info
        )
        
        # Apply energy-aware modifications if energy levels provided
        if energy_levels is not None:
            # Identify low-energy agents
            low_energy_mask = energy_levels < self.critical_battery_level
            
            # Reduce attention from low-energy agents
            if low_energy_mask.any():
                attention_weights = info["attention_weights"]
                
                # Scale down attention from low-energy agents
                energy_scale = torch.where(
                    low_energy_mask,
                    energy_levels / self.critical_battery_level,
                    torch.ones_like(energy_levels)
                )
                
                # Apply scaling
                scaled_weights = attention_weights * energy_scale.unsqueeze(1).unsqueeze(-1)
                
                # Renormalize
                scaled_weights = scaled_weights / (scaled_weights.sum(dim=-1, keepdim=True) + 1e-8)
                
                info["attention_weights"] = scaled_weights
                info["energy_scale"] = energy_scale
        
        return output, info