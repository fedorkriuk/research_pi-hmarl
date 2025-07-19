"""Action Decomposer for Hierarchical Action Spaces

This module decomposes high-level actions into low-level control commands
based on real drone control systems.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ActionDecomposer(nn.Module):
    """Decomposes hierarchical actions across different levels"""
    
    def __init__(
        self,
        meta_action_dim: int,
        primitive_action_dim: int,
        hidden_dim: int = 64,
        use_learned_decomposition: bool = True
    ):
        """Initialize action decomposer
        
        Args:
            meta_action_dim: Dimension of meta-actions (options)
            primitive_action_dim: Dimension of primitive actions
            hidden_dim: Hidden layer dimension
            use_learned_decomposition: Whether to learn decomposition
        """
        super().__init__()
        
        self.meta_action_dim = meta_action_dim
        self.primitive_action_dim = primitive_action_dim
        self.use_learned_decomposition = use_learned_decomposition
        
        if use_learned_decomposition:
            # Learned decomposition network
            self.decomposition_net = nn.Sequential(
                nn.Linear(primitive_action_dim + meta_action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, primitive_action_dim)
            )
            
            # Option-specific action modulation
            self.option_modulation = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(primitive_action_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, primitive_action_dim),
                    nn.Tanh()
                ) for _ in range(meta_action_dim)
            ])
        
        # Predefined action mappings based on real drone control
        self.action_mappings = self._create_action_mappings()
        
        # Action constraints
        self.action_constraints = ActionConstraints()
        
        logger.info(f"Initialized ActionDecomposer")
    
    def forward(
        self,
        primitive_action: torch.Tensor,
        meta_action: int
    ) -> torch.Tensor:
        """Decompose action based on hierarchy
        
        Args:
            primitive_action: Low-level action from execution policy
            meta_action: High-level option/meta-action
            
        Returns:
            Final decomposed action
        """
        if self.use_learned_decomposition:
            # Create one-hot encoding of meta action
            meta_one_hot = torch.zeros(
                primitive_action.shape[0], 
                self.meta_action_dim,
                device=primitive_action.device
            )
            meta_one_hot[:, meta_action] = 1
            
            # Concatenate primitive action and meta action
            combined = torch.cat([primitive_action, meta_one_hot], dim=-1)
            
            # Apply decomposition network
            base_action = self.decomposition_net(combined)
            
            # Apply option-specific modulation
            modulated_action = base_action + self.option_modulation[meta_action](primitive_action)
            
            # Ensure action is in valid range
            final_action = torch.tanh(modulated_action)
        else:
            # Use predefined mappings
            final_action = self._apply_predefined_mapping(
                primitive_action, meta_action
            )
        
        # Apply action constraints
        final_action = self.action_constraints(final_action, meta_action)
        
        return final_action
    
    def _create_action_mappings(self) -> Dict[int, Dict[str, Any]]:
        """Create predefined action mappings for each option
        
        Returns:
            Dictionary of action mappings
        """
        mappings = {}
        
        # Hover option (0)
        mappings[0] = {
            "velocity_scale": [0.1, 0.1, 0.2],  # Minimal movement
            "yaw_scale": 0.1,
            "constraints": {
                "max_velocity": 1.0,  # m/s
                "max_yaw_rate": 0.5   # rad/s
            }
        }
        
        # Move forward option (1)
        mappings[1] = {
            "velocity_scale": [1.0, 0.3, 0.3],  # Forward emphasis
            "yaw_scale": 0.3,
            "constraints": {
                "max_velocity": 10.0,
                "max_yaw_rate": 1.0
            }
        }
        
        # Circle search option (2)
        mappings[2] = {
            "velocity_scale": [0.7, 0.7, 0.3],
            "yaw_scale": 1.0,  # Higher yaw for circling
            "constraints": {
                "max_velocity": 5.0,
                "max_yaw_rate": 2.0
            }
        }
        
        # Formation options (3, 4)
        mappings[3] = mappings[4] = {
            "velocity_scale": [0.5, 0.5, 0.5],  # Balanced movement
            "yaw_scale": 0.5,
            "constraints": {
                "max_velocity": 7.0,
                "max_yaw_rate": 1.5
            }
        }
        
        # Search pattern option (5)
        mappings[5] = {
            "velocity_scale": [0.8, 0.8, 0.4],
            "yaw_scale": 0.7,
            "constraints": {
                "max_velocity": 8.0,
                "max_yaw_rate": 1.8
            }
        }
        
        # Return to base option (6)
        mappings[6] = {
            "velocity_scale": [1.0, 1.0, 0.5],  # Efficient return
            "yaw_scale": 0.3,
            "constraints": {
                "max_velocity": 15.0,  # Higher speed for return
                "max_yaw_rate": 1.0
            }
        }
        
        # Emergency land option (7)
        mappings[7] = {
            "velocity_scale": [0.0, 0.0, 1.0],  # Vertical only
            "yaw_scale": 0.0,
            "constraints": {
                "max_velocity": 2.0,  # Slow descent
                "max_yaw_rate": 0.0
            }
        }
        
        # Default mapping for undefined options
        default_mapping = {
            "velocity_scale": [0.5, 0.5, 0.5],
            "yaw_scale": 0.5,
            "constraints": {
                "max_velocity": 5.0,
                "max_yaw_rate": 1.0
            }
        }
        
        # Fill remaining options with default
        for i in range(self.meta_action_dim):
            if i not in mappings:
                mappings[i] = default_mapping.copy()
        
        return mappings
    
    def _apply_predefined_mapping(
        self,
        primitive_action: torch.Tensor,
        meta_action: int
    ) -> torch.Tensor:
        """Apply predefined action mapping
        
        Args:
            primitive_action: Base action
            meta_action: Current option
            
        Returns:
            Mapped action
        """
        mapping = self.action_mappings.get(meta_action, self.action_mappings[0])
        
        # Apply velocity scaling
        scaled_action = primitive_action.clone()
        velocity_scale = torch.tensor(
            mapping["velocity_scale"] + [mapping["yaw_scale"]],
            device=primitive_action.device
        )
        
        # Ensure correct dimensions
        if scaled_action.shape[-1] >= len(velocity_scale):
            scaled_action[:, :len(velocity_scale)] *= velocity_scale
        
        return scaled_action


class ActionConstraints(nn.Module):
    """Applies real-world constraints to actions"""
    
    def __init__(self):
        """Initialize action constraints"""
        super().__init__()
        
        # Real drone constraints (DJI Mavic 3 specs)
        self.constraints = {
            "max_velocity": 21.0,      # m/s
            "max_acceleration": 10.0,   # m/s²
            "max_angular_velocity": 3.14,  # rad/s
            "max_tilt_angle": 0.785,   # 45 degrees
            "min_altitude": 0.5,       # m
            "max_altitude": 500.0      # m (regulatory limit)
        }
        
        # Safety margins
        self.safety_factor = 0.8
    
    def forward(
        self,
        action: torch.Tensor,
        option_id: int
    ) -> torch.Tensor:
        """Apply constraints to action
        
        Args:
            action: Raw action
            option_id: Current option
            
        Returns:
            Constrained action
        """
        constrained = action.clone()
        
        # Apply option-specific constraints
        if option_id == 7:  # Emergency landing
            # Only allow downward motion
            constrained[:, 0] = 0  # No x velocity
            constrained[:, 1] = 0  # No y velocity
            constrained[:, 2] = torch.clamp(constrained[:, 2], -1, 0)  # Only down
            if action.shape[-1] > 3:
                constrained[:, 3] = 0  # No yaw
        
        # Apply general safety constraints
        constrained = self._apply_safety_constraints(constrained)
        
        return constrained
    
    def _apply_safety_constraints(
        self,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Apply general safety constraints
        
        Args:
            action: Action to constrain
            
        Returns:
            Safe action
        """
        # Apply safety factor
        safe_action = action * self.safety_factor
        
        # Ensure actions are in [-1, 1] range
        safe_action = torch.clamp(safe_action, -1, 1)
        
        return safe_action


class ActionPrimitives:
    """Library of action primitives for common maneuvers"""
    
    def __init__(self):
        """Initialize action primitives"""
        self.primitives = self._create_primitives()
    
    def _create_primitives(self) -> Dict[str, np.ndarray]:
        """Create basic action primitives
        
        Returns:
            Dictionary of action primitives
        """
        primitives = {}
        
        # Basic movements
        primitives["stop"] = np.array([0, 0, 0, 0])
        primitives["forward"] = np.array([1, 0, 0, 0])
        primitives["backward"] = np.array([-1, 0, 0, 0])
        primitives["left"] = np.array([0, 1, 0, 0])
        primitives["right"] = np.array([0, -1, 0, 0])
        primitives["up"] = np.array([0, 0, 1, 0])
        primitives["down"] = np.array([0, 0, -1, 0])
        
        # Rotations
        primitives["yaw_left"] = np.array([0, 0, 0, 1])
        primitives["yaw_right"] = np.array([0, 0, 0, -1])
        
        # Diagonal movements
        primitives["forward_left"] = np.array([0.7, 0.7, 0, 0])
        primitives["forward_right"] = np.array([0.7, -0.7, 0, 0])
        primitives["backward_left"] = np.array([-0.7, 0.7, 0, 0])
        primitives["backward_right"] = np.array([-0.7, -0.7, 0, 0])
        
        # Complex maneuvers
        primitives["spiral_up"] = np.array([0.5, 0, 0.5, 0.7])
        primitives["spiral_down"] = np.array([0.5, 0, -0.5, 0.7])
        primitives["circle_left"] = np.array([0.7, 0.7, 0, 0.5])
        primitives["circle_right"] = np.array([0.7, -0.7, 0, -0.5])
        
        return primitives
    
    def get_primitive(self, name: str) -> Optional[np.ndarray]:
        """Get action primitive by name
        
        Args:
            name: Primitive name
            
        Returns:
            Action primitive or None
        """
        return self.primitives.get(name)
    
    def combine_primitives(
        self,
        primitives: List[str],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """Combine multiple primitives
        
        Args:
            primitives: List of primitive names
            weights: Optional weights for combination
            
        Returns:
            Combined action
        """
        if not primitives:
            return np.zeros(4)
        
        if weights is None:
            weights = [1.0 / len(primitives)] * len(primitives)
        
        combined = np.zeros(4)
        for primitive, weight in zip(primitives, weights):
            if primitive in self.primitives:
                combined += weight * self.primitives[primitive]
        
        # Normalize to valid range
        combined = np.clip(combined, -1, 1)
        
        return combined