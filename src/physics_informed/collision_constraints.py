"""Collision Constraints for Multi-Agent Systems

This module implements collision avoidance constraints using real safety
margins and physics-based models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class CollisionConstraints:
    """Enforces collision avoidance constraints"""
    
    def __init__(
        self,
        min_separation: float = 5.0,  # meters (based on drone safety)
        safety_margin: float = 1.5,   # Additional safety factor
        max_agents: int = 20,
        collision_penalty: float = 100.0,
        use_soft_constraints: bool = True
    ):
        """Initialize collision constraints
        
        Args:
            min_separation: Minimum separation distance
            safety_margin: Safety margin multiplier
            max_agents: Maximum number of agents
            collision_penalty: Penalty weight for violations
            use_soft_constraints: Whether to use soft constraints
        """
        self.min_separation = min_separation
        self.safety_margin = safety_margin
        self.safe_distance = min_separation * safety_margin
        self.max_agents = max_agents
        self.collision_penalty = collision_penalty
        self.use_soft_constraints = use_soft_constraints
        
        logger.info(f"Initialized CollisionConstraints with min_sep={min_separation}m")
    
    def compute_collision_losses(
        self,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        radii: Optional[torch.Tensor] = None,
        return_violations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Compute all collision-related losses
        
        Args:
            positions: Agent positions [batch, num_agents, 3]
            velocities: Agent velocities [batch, num_agents, 3]
            radii: Agent radii [num_agents] or scalar
            return_violations: Whether to return violation details
            
        Returns:
            Dictionary of collision losses
        """
        losses = {}
        
        # Distance-based collision loss
        distance_loss, violations = self.distance_constraint_loss(
            positions, radii, return_violations=True
        )
        losses["distance_collision"] = distance_loss
        
        # Velocity-based collision prediction
        if velocities is not None:
            ttc_loss = self.time_to_collision_loss(positions, velocities)
            losses["time_to_collision"] = ttc_loss
            
            # Dynamic separation based on velocity
            dynamic_loss = self.dynamic_separation_loss(positions, velocities, radii)
            losses["dynamic_separation"] = dynamic_loss
        
        # Potential field collision avoidance
        potential_loss = self.potential_field_loss(positions, radii)
        losses["collision_potential"] = potential_loss
        
        if return_violations:
            losses["violations"] = violations
        
        return losses
    
    def distance_constraint_loss(
        self,
        positions: torch.Tensor,
        radii: Optional[torch.Tensor] = None,
        return_violations: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Distance-based collision constraint
        
        Args:
            positions: Agent positions
            radii: Agent radii
            return_violations: Whether to return violations
            
        Returns:
            Loss and optionally violations
        """
        batch_size, num_agents, _ = positions.shape
        
        # Default radius if not provided
        if radii is None:
            radii = torch.ones(num_agents, device=positions.device) * 0.5  # 0.5m default
        elif radii.dim() == 0:
            radii = radii.expand(num_agents)
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(positions)
        
        # Required separation (sum of radii + safety)
        radii_sum = radii.unsqueeze(0) + radii.unsqueeze(1)
        required_separation = radii_sum + self.safe_distance
        
        # Expand for batch dimension
        required_separation = required_separation.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Collision violations
        violations = required_separation - distances
        
        # Mask diagonal (self-distances)
        mask = 1 - torch.eye(num_agents, device=positions.device).unsqueeze(0)
        violations = violations * mask
        
        if self.use_soft_constraints:
            # Soft constraint with smooth penalty
            penalty = F.softplus(violations * 10) * mask
            loss = self.collision_penalty * penalty.sum() / (num_agents * (num_agents - 1))
        else:
            # Hard constraint
            penalty = F.relu(violations) ** 2 * mask
            loss = self.collision_penalty * penalty.sum() / (num_agents * (num_agents - 1))
        
        if return_violations:
            return loss, violations
        return loss, None
    
    def time_to_collision_loss(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        min_ttc: float = 3.0  # seconds
    ) -> torch.Tensor:
        """Time-to-collision constraint
        
        Args:
            positions: Agent positions
            velocities: Agent velocities
            min_ttc: Minimum time to collision
            
        Returns:
            TTC loss
        """
        batch_size, num_agents, _ = positions.shape
        
        # Relative positions and velocities
        rel_positions = positions.unsqueeze(2) - positions.unsqueeze(1)
        rel_velocities = velocities.unsqueeze(2) - velocities.unsqueeze(1)
        
        # Distance and approach rate
        distances = torch.norm(rel_positions, dim=-1)
        approach_rates = -(rel_positions * rel_velocities).sum(dim=-1) / (distances + 1e-6)
        
        # Only consider approaching agents
        approaching = approach_rates > 0
        
        # Time to collision
        ttc = distances / (approach_rates + 1e-6)
        ttc = torch.where(approaching, ttc, torch.full_like(ttc, float('inf')))
        
        # Mask self and non-approaching
        mask = (1 - torch.eye(num_agents, device=positions.device).unsqueeze(0)) * approaching.float()
        
        # Penalize small TTC
        if self.use_soft_constraints:
            penalty = F.softplus((min_ttc - ttc) * 2) * mask
        else:
            penalty = F.relu(min_ttc - ttc) ** 2 * mask
        
        loss = penalty.sum() / (mask.sum() + 1e-6)
        
        return loss
    
    def dynamic_separation_loss(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        radii: Optional[torch.Tensor] = None,
        velocity_scale: float = 0.5  # seconds of lookahead
    ) -> torch.Tensor:
        """Dynamic separation based on velocity
        
        Args:
            positions: Agent positions
            velocities: Agent velocities
            radii: Agent radii
            velocity_scale: Time scale for velocity-based separation
            
        Returns:
            Dynamic separation loss
        """
        # Speed magnitude
        speeds = torch.norm(velocities, dim=-1)
        
        # Dynamic safety margin based on speed
        # Faster agents need more separation
        dynamic_margin = self.safe_distance + velocity_scale * speeds
        
        # Pairwise dynamic margins
        dynamic_margins = dynamic_margin.unsqueeze(2) + dynamic_margin.unsqueeze(1)
        
        # Current distances
        distances = self._compute_pairwise_distances(positions)
        
        # Violations
        violations = dynamic_margins - distances
        
        # Mask diagonal
        batch_size, num_agents = positions.shape[:2]
        mask = 1 - torch.eye(num_agents, device=positions.device).unsqueeze(0)
        violations = violations * mask
        
        # Loss
        if self.use_soft_constraints:
            penalty = F.softplus(violations * 5) * mask
        else:
            penalty = F.relu(violations) ** 2 * mask
        
        loss = penalty.sum() / (num_agents * (num_agents - 1))
        
        return loss
    
    def potential_field_loss(
        self,
        positions: torch.Tensor,
        radii: Optional[torch.Tensor] = None,
        potential_scale: float = 10.0
    ) -> torch.Tensor:
        """Potential field for collision avoidance
        
        Args:
            positions: Agent positions
            radii: Agent radii
            potential_scale: Scale factor for potential
            
        Returns:
            Potential field loss
        """
        # Pairwise distances
        distances = self._compute_pairwise_distances(positions)
        
        # Mask diagonal
        batch_size, num_agents = positions.shape[:2]
        mask = 1 - torch.eye(num_agents, device=positions.device).unsqueeze(0)
        
        # Repulsive potential: U = k/d²
        # Add small epsilon to avoid division by zero
        potential = potential_scale / (distances ** 2 + 0.1)
        potential = potential * mask
        
        # Only activate when close
        close_mask = (distances < self.safe_distance * 2) * mask
        active_potential = potential * close_mask
        
        # Total potential energy (should be minimized)
        loss = active_potential.sum() / (close_mask.sum() + 1e-6)
        
        return loss
    
    def _compute_pairwise_distances(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances
        
        Args:
            positions: Agent positions [batch, num_agents, dim]
            
        Returns:
            Distance matrix [batch, num_agents, num_agents]
        """
        # Expand dimensions for broadcasting
        pos1 = positions.unsqueeze(2)  # [batch, agents, 1, dim]
        pos2 = positions.unsqueeze(1)  # [batch, 1, agents, dim]
        
        # Compute distances
        distances = torch.norm(pos1 - pos2, dim=-1)
        
        return distances


class SafetyDistanceConstraint(nn.Module):
    """Neural network module for safety distance constraints"""
    
    def __init__(
        self,
        min_distance: float = 5.0,
        hidden_dim: int = 64,
        num_agents: int = 20
    ):
        """Initialize safety distance constraint
        
        Args:
            min_distance: Minimum safety distance
            hidden_dim: Hidden dimension
            num_agents: Number of agents
        """
        super().__init__()
        
        self.min_distance = min_distance
        self.num_agents = num_agents
        
        # Learned safety margin predictor
        self.safety_margin_net = nn.Sequential(
            nn.Linear(6, hidden_dim),  # Relative pos and vel
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Repulsive force network
        self.repulsion_net = nn.Sequential(
            nn.Linear(7, hidden_dim),  # Rel pos, vel, distance
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Force vector
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute safety forces and constraints
        
        Args:
            positions: Agent positions
            velocities: Agent velocities
            
        Returns:
            Safety forces and constraint violations
        """
        batch_size = positions.shape[0]
        forces = torch.zeros_like(positions)
        violations = []
        
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                # Relative state
                rel_pos = positions[:, j] - positions[:, i]
                rel_vel = velocities[:, j] - velocities[:, i]
                distance = torch.norm(rel_pos, dim=-1, keepdim=True)
                
                # Predict safety margin
                rel_state = torch.cat([rel_pos, rel_vel], dim=-1)
                margin_scale = self.safety_margin_net(rel_state)
                safety_distance = self.min_distance * (1 + margin_scale)
                
                # Check violation
                violation = safety_distance - distance
                violations.append(violation)
                
                # Compute repulsive force if too close
                if (violation > 0).any():
                    force_input = torch.cat([rel_pos, rel_vel, distance], dim=-1)
                    repulsion = self.repulsion_net(force_input)
                    
                    # Apply forces (Newton's third law)
                    forces[:, i] -= repulsion
                    forces[:, j] += repulsion
        
        violations = torch.cat(violations, dim=-1) if violations else torch.tensor(0.0)
        
        return forces, violations


class TimeToCollisionConstraint(nn.Module):
    """Neural network module for time-to-collision constraints"""
    
    def __init__(
        self,
        min_ttc: float = 3.0,
        hidden_dim: int = 64,
        predict_trajectories: bool = True
    ):
        """Initialize TTC constraint
        
        Args:
            min_ttc: Minimum time to collision
            hidden_dim: Hidden dimension
            predict_trajectories: Whether to predict future trajectories
        """
        super().__init__()
        
        self.min_ttc = min_ttc
        self.predict_trajectories = predict_trajectories
        
        if predict_trajectories:
            # Trajectory prediction network
            self.trajectory_net = nn.LSTM(
                input_size=6,  # Position + velocity
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True
            )
            
            self.trajectory_decoder = nn.Linear(hidden_dim, 3)  # Future position
        
        # Avoidance action network
        self.avoidance_net = nn.Sequential(
            nn.Linear(7, hidden_dim),  # State + TTC
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Avoidance acceleration
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        dt: float = 0.1,
        horizon: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute TTC constraints and avoidance actions
        
        Args:
            positions: Current positions
            velocities: Current velocities
            dt: Time step
            horizon: Prediction horizon
            
        Returns:
            Avoidance actions and TTC values
        """
        batch_size, num_agents, _ = positions.shape
        
        if self.predict_trajectories:
            # Predict future trajectories
            trajectories = self._predict_trajectories(
                positions, velocities, dt, horizon
            )
            
            # Check collisions along trajectories
            ttc_values = self._compute_trajectory_ttc(trajectories, dt)
        else:
            # Simple linear extrapolation
            ttc_values = self._compute_linear_ttc(positions, velocities)
        
        # Compute avoidance actions for risky pairs
        avoidance_actions = torch.zeros_like(velocities)
        
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                ttc = ttc_values[:, i, j]
                
                # If collision risk
                if (ttc < self.min_ttc).any():
                    # Relative state
                    rel_pos = positions[:, j] - positions[:, i]
                    rel_vel = velocities[:, j] - velocities[:, i]
                    
                    # Avoidance input
                    avoid_input = torch.cat([
                        rel_pos, rel_vel, ttc.unsqueeze(-1)
                    ], dim=-1)
                    
                    # Compute avoidance
                    avoidance = self.avoidance_net(avoid_input)
                    
                    # Apply symmetric avoidance
                    avoidance_actions[:, i] -= avoidance * 0.5
                    avoidance_actions[:, j] += avoidance * 0.5
        
        return avoidance_actions, ttc_values
    
    def _predict_trajectories(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        dt: float,
        horizon: int
    ) -> torch.Tensor:
        """Predict future trajectories
        
        Args:
            positions: Current positions
            velocities: Current velocities
            dt: Time step
            horizon: Steps to predict
            
        Returns:
            Predicted trajectories [batch, agents, horizon, 3]
        """
        batch_size, num_agents, _ = positions.shape
        
        trajectories = []
        state = torch.cat([positions, velocities], dim=-1)
        
        # Initialize LSTM hidden state
        h = torch.zeros(2, batch_size * num_agents, self.trajectory_net.hidden_size,
                       device=positions.device)
        c = torch.zeros_like(h)
        
        # Predict trajectory
        current_pos = positions
        for t in range(horizon):
            # LSTM forward
            state_flat = state.view(batch_size * num_agents, 1, -1)
            output, (h, c) = self.trajectory_net(state_flat, (h, c))
            
            # Decode position change
            delta_pos = self.trajectory_decoder(output.squeeze(1))
            delta_pos = delta_pos.view(batch_size, num_agents, 3)
            
            # Update position
            current_pos = current_pos + delta_pos * dt
            trajectories.append(current_pos)
            
            # Update state
            current_vel = delta_pos / dt
            state = torch.cat([current_pos, current_vel], dim=-1)
        
        return torch.stack(trajectories, dim=2)
    
    def _compute_linear_ttc(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor
    ) -> torch.Tensor:
        """Compute TTC with linear motion assumption
        
        Args:
            positions: Positions
            velocities: Velocities
            
        Returns:
            TTC matrix [batch, agents, agents]
        """
        batch_size, num_agents, _ = positions.shape
        
        ttc = torch.full(
            (batch_size, num_agents, num_agents),
            float('inf'),
            device=positions.device
        )
        
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # Relative motion
                rel_pos = positions[:, j] - positions[:, i]
                rel_vel = velocities[:, j] - velocities[:, i]
                
                # Distance and approach rate
                distance = torch.norm(rel_pos, dim=-1)
                approach_rate = -(rel_pos * rel_vel).sum(dim=-1) / (distance + 1e-6)
                
                # TTC (only if approaching)
                approaching = approach_rate > 0
                ttc_ij = torch.where(
                    approaching,
                    distance / (approach_rate + 1e-6),
                    torch.full_like(distance, float('inf'))
                )
                
                ttc[:, i, j] = ttc_ij
                ttc[:, j, i] = ttc_ij
        
        return ttc
    
    def _compute_trajectory_ttc(
        self,
        trajectories: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """Compute TTC from predicted trajectories
        
        Args:
            trajectories: Predicted trajectories
            dt: Time step
            
        Returns:
            TTC matrix
        """
        batch_size, num_agents, horizon, _ = trajectories.shape
        
        ttc = torch.full(
            (batch_size, num_agents, num_agents),
            float('inf'),
            device=trajectories.device
        )
        
        # Check each time step
        for t in range(horizon):
            positions_t = trajectories[:, :, t, :]
            distances = self._compute_pairwise_distances(positions_t)
            
            # Update TTC for first collision
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    if distances[:, i, j] < self.min_distance:
                        current_ttc = t * dt
                        ttc[:, i, j] = torch.min(ttc[:, i, j], current_ttc)
                        ttc[:, j, i] = ttc[:, i, j]
        
        return ttc
    
    def _compute_pairwise_distances(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances"""
        pos1 = positions.unsqueeze(2)
        pos2 = positions.unsqueeze(1)
        return torch.norm(pos1 - pos2, dim=-1)