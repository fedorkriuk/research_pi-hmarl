"""Execution Policy for Low-Level Control

This module implements the execution policy that generates low-level
control actions based on real drone control interfaces.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ExecutionPolicy(nn.Module):
    """Low-level execution policy for direct control"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        control_frequency: float = 20.0,  # Hz
        action_std_init: float = 0.5,
        use_tanh: bool = True
    ):
        """Initialize execution policy
        
        Args:
            state_dim: Dimension of local state
            action_dim: Dimension of control actions
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            control_frequency: Control loop frequency (Hz)
            action_std_init: Initial action standard deviation
            use_tanh: Whether to use tanh activation on outputs
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.control_frequency = control_frequency
        self.control_period = 1.0 / control_frequency
        self.use_tanh = use_tanh
        
        # Real-time constraint
        self.max_inference_time = 0.020  # 20ms for real-time control
        
        # Policy network
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Action mean
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        
        # Action log std (learnable parameter)
        self.action_log_std = nn.Parameter(
            torch.ones(action_dim) * np.log(action_std_init)
        )
        
        # Value function
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Control limits based on real drone specifications
        self.action_limits = {
            "velocity": 20.0,      # m/s (DJI Mavic 3 max)
            "acceleration": 10.0,   # m/s²
            "angular_velocity": 3.14,  # rad/s
            "thrust": 1.0          # Normalized thrust
        }
        
        # Safety filters
        self.safety_filter = SafetyFilter(
            action_dim=action_dim,
            limits=self.action_limits
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized ExecutionPolicy with {control_frequency}Hz control")
    
    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass
        
        Args:
            state: Local state features
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action and info dict
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Get action distribution
        action_mean = self.action_mean(features)
        action_std = torch.exp(self.action_log_std)
        
        if deterministic:
            action = action_mean
        else:
            # Sample from normal distribution
            dist = Normal(action_mean, action_std)
            action = dist.rsample()  # Reparameterization trick
        
        # Apply tanh squashing if enabled
        if self.use_tanh:
            action = torch.tanh(action)
        
        # Apply safety filtering
        action = self.safety_filter(action, state)
        
        # Compute log probability
        if deterministic:
            log_prob = torch.zeros(state.shape[0], device=state.device)
        else:
            if self.use_tanh:
                # Account for tanh squashing in log prob
                log_prob = dist.log_prob(action).sum(dim=-1)
                log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=-1)
            else:
                log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Get value estimate
        value = self.value_head(features).squeeze(-1)
        
        info = {
            "value": value,
            "log_prob": log_prob,
            "action_mean": action_mean,
            "action_std": action_std,
            "pre_safety_action": action.clone()
        }
        
        return action, info
    
    def get_action_distribution(
        self,
        state: torch.Tensor
    ) -> Normal:
        """Get action distribution for given state
        
        Args:
            state: State tensor
            
        Returns:
            Action distribution
        """
        features = self.feature_extractor(state)
        action_mean = self.action_mean(features)
        action_std = torch.exp(self.action_log_std)
        
        return Normal(action_mean, action_std)
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Evaluate actions for given states (for training)
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            Dictionary with log_probs, values, entropy
        """
        features = self.feature_extractor(states)
        
        # Get action distribution
        action_mean = self.action_mean(features)
        action_std = torch.exp(self.action_log_std)
        dist = Normal(action_mean, action_std)
        
        # Compute log probabilities
        if self.use_tanh:
            # Inverse tanh to get pre-squashed actions
            pre_tanh_value = torch.atanh(torch.clamp(actions, -0.999, 0.999))
            log_probs = dist.log_prob(pre_tanh_value).sum(dim=-1)
            log_probs -= (2 * (np.log(2) - pre_tanh_value - F.softplus(-2 * pre_tanh_value))).sum(dim=-1)
        else:
            log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Get values
        values = self.value_head(features).squeeze(-1)
        
        # Compute entropy
        entropy = dist.entropy().sum(dim=-1)
        
        return {
            "log_probs": log_probs,
            "values": values,
            "entropy": entropy
        }
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class SafetyFilter(nn.Module):
    """Safety filter for control actions"""
    
    def __init__(
        self,
        action_dim: int,
        limits: Dict[str, float],
        smoothing: float = 0.9
    ):
        """Initialize safety filter
        
        Args:
            action_dim: Action dimension
            limits: Action limits
            smoothing: Temporal smoothing factor
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.limits = limits
        self.smoothing = smoothing
        
        # Previous action for smoothing
        self.register_buffer(
            "prev_action",
            torch.zeros(1, action_dim)
        )
        
        # Real drone control mappings
        self.control_mapping = {
            0: ("velocity_x", -limits["velocity"], limits["velocity"]),
            1: ("velocity_y", -limits["velocity"], limits["velocity"]),
            2: ("velocity_z", -limits["velocity"], limits["velocity"]),
            3: ("yaw_rate", -limits["angular_velocity"], limits["angular_velocity"])
        }
    
    def forward(
        self,
        action: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Apply safety filtering
        
        Args:
            action: Raw action
            state: Current state
            
        Returns:
            Safe action
        """
        batch_size = action.shape[0]
        
        # Expand prev_action if needed
        if self.prev_action.shape[0] != batch_size:
            self.prev_action = self.prev_action.expand(batch_size, -1)
        
        # Apply control limits
        safe_action = action.clone()
        for idx, (control_type, min_val, max_val) in self.control_mapping.items():
            if idx < self.action_dim:
                safe_action[:, idx] = torch.clamp(
                    action[:, idx],
                    min_val / self.limits["velocity"],  # Normalize
                    max_val / self.limits["velocity"]
                )
        
        # Apply temporal smoothing for stability
        safe_action = self.smoothing * self.prev_action + (1 - self.smoothing) * safe_action
        
        # Check state-dependent constraints
        safe_action = self._apply_state_constraints(safe_action, state)
        
        # Update previous action
        self.prev_action = safe_action.detach()
        
        return safe_action
    
    def _apply_state_constraints(
        self,
        action: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Apply state-dependent safety constraints
        
        Args:
            action: Current action
            state: Current state
            
        Returns:
            Constrained action
        """
        # Example: Reduce velocity near ground (assuming altitude in state)
        # This would be more sophisticated in practice
        if state.shape[1] > 2:  # Assuming altitude is 3rd element
            altitude = state[:, 2]
            low_altitude_mask = altitude < 5.0  # meters
            
            # Reduce vertical velocity near ground
            action = action.clone()
            action[low_altitude_mask, 2] = torch.clamp(
                action[low_altitude_mask, 2],
                -0.5,  # Slower descent near ground
                0.5
            )
        
        return action


class MotorMixer:
    """Converts high-level commands to motor speeds"""
    
    def __init__(self, num_motors: int = 4):
        """Initialize motor mixer
        
        Args:
            num_motors: Number of motors
        """
        self.num_motors = num_motors
        
        # Quadcopter mixing matrix (X configuration)
        # Maps [thrust, roll, pitch, yaw] to motor commands
        self.mixing_matrix = np.array([
            [0.25,  0.25,  0.25, -0.25],  # Front-right
            [0.25, -0.25, -0.25, -0.25],  # Rear-left
            [0.25, -0.25,  0.25,  0.25],  # Front-left
            [0.25,  0.25, -0.25,  0.25]   # Rear-right
        ])
    
    def mix(
        self,
        thrust: float,
        roll: float,
        pitch: float,
        yaw: float
    ) -> np.ndarray:
        """Mix control inputs to motor commands
        
        Args:
            thrust: Total thrust (0-1)
            roll: Roll command (-1 to 1)
            pitch: Pitch command (-1 to 1)
            yaw: Yaw command (-1 to 1)
            
        Returns:
            Motor commands (0-1) for each motor
        """
        control_vector = np.array([thrust, roll, pitch, yaw])
        motor_commands = self.mixing_matrix @ control_vector
        
        # Clamp to valid range
        motor_commands = np.clip(motor_commands, 0, 1)
        
        return motor_commands