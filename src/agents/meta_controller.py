"""Meta-Controller for High-Level Strategic Decisions

This module implements the meta-controller that makes high-level strategic
decisions based on real mission planning constraints.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MetaController(nn.Module):
    """Meta-controller for strategic decision making"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        planning_horizon: float = 30.0,  # seconds
        use_lstm: bool = True,
        dropout: float = 0.1
    ):
        """Initialize meta-controller
        
        Args:
            state_dim: Dimension of strategic state
            action_dim: Number of high-level actions/options
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            planning_horizon: Planning horizon in seconds
            use_lstm: Whether to use LSTM for temporal modeling
            dropout: Dropout rate
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.planning_horizon = planning_horizon
        self.use_lstm = use_lstm
        
        # Strategic state encoder
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Temporal modeling
        if use_lstm:
            self.lstm = nn.LSTM(
                hidden_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.hidden_state = None
        else:
            # Feed-forward layers
            layers = []
            for i in range(num_layers):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout)
                ])
            self.ff_layers = nn.Sequential(*layers)
        
        # Policy head (discrete high-level actions)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Mission context encoder
        self.mission_context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized MetaController with planning horizon {planning_horizon}s")
    
    def forward(
        self,
        state: torch.Tensor,
        mission_context: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass
        
        Args:
            state: Strategic state features
            mission_context: Optional mission context
            deterministic: Whether to use deterministic policy
            
        Returns:
            Selected action and info dict
        """
        batch_size = state.shape[0]
        
        # Encode state
        x = self.input_layer(state)
        x = F.relu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Add mission context if available
        if mission_context is not None:
            context_features = self.mission_context(mission_context)
            x = x + context_features
        
        # Temporal modeling
        if self.use_lstm:
            # Add sequence dimension
            x = x.unsqueeze(1)
            
            if self.hidden_state is None or self.hidden_state[0].shape[1] != batch_size:
                # Initialize hidden state
                self.hidden_state = (
                    torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim).to(x.device),
                    torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim).to(x.device)
                )
            
            x, self.hidden_state = self.lstm(x, self.hidden_state)
            x = x.squeeze(1)  # Remove sequence dimension
            
            # Detach hidden state to prevent backprop through time
            self.hidden_state = (self.hidden_state[0].detach(), self.hidden_state[1].detach())
        else:
            x = self.ff_layers(x)
        
        # Get policy logits and value
        logits = self.policy_head(x)
        value = self.value_head(x)
        
        # Action selection
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            action_probs = F.softmax(logits, dim=-1)
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)) + 1e-8)
        else:
            # Sample from categorical distribution
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action_probs = dist.probs
        
        # Compute entropy for exploration
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)
        
        info = {
            "value": value.squeeze(-1),
            "log_prob": log_prob,
            "entropy": entropy,
            "action_probs": action_probs,
            "logits": logits
        }
        
        return action, info
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate for state
        
        Args:
            state: Strategic state
            
        Returns:
            Value estimate
        """
        x = self.input_layer(state)
        x = F.relu(x)
        x = self.layer_norm(x)
        
        if self.use_lstm:
            x = x.unsqueeze(1)
            if self.hidden_state is not None:
                x, _ = self.lstm(x, self.hidden_state)
            else:
                x, _ = self.lstm(x)
            x = x.squeeze(1)
        else:
            x = self.ff_layers(x)
        
        value = self.value_head(x)
        return value.squeeze(-1)
    
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
        batch_size = states.shape[0]
        
        # Reset LSTM state for batch processing
        if self.use_lstm:
            self.hidden_state = None
        
        # Forward pass
        x = self.input_layer(states)
        x = F.relu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        if self.use_lstm:
            x = x.unsqueeze(1)
            x, _ = self.lstm(x)
            x = x.squeeze(1)
        else:
            x = self.ff_layers(x)
        
        # Get outputs
        logits = self.policy_head(x)
        values = self.value_head(x).squeeze(-1)
        
        # Compute log probabilities
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return {
            "log_probs": log_probs,
            "values": values,
            "entropy": entropy
        }
    
    def reset_hidden_state(self):
        """Reset LSTM hidden state"""
        self.hidden_state = None
    
    def plan_mission(
        self,
        initial_state: torch.Tensor,
        mission_objectives: List[Dict[str, Any]],
        constraints: Dict[str, float]
    ) -> List[int]:
        """Plan mission given objectives and constraints
        
        Args:
            initial_state: Initial strategic state
            mission_objectives: List of mission objectives
            constraints: Operational constraints (time, energy, etc.)
            
        Returns:
            Sequence of high-level actions
        """
        plan = []
        state = initial_state
        total_time = 0.0
        
        # Reset hidden state for planning
        self.reset_hidden_state()
        
        with torch.no_grad():
            while total_time < self.planning_horizon and len(plan) < 10:
                # Get action
                action, info = self.forward(state, deterministic=True)
                plan.append(action.item())
                
                # Estimate time for action (simplified)
                estimated_time = 5.0  # seconds per high-level action
                total_time += estimated_time
                
                # Check constraints
                if total_time > constraints.get("max_time", float('inf')):
                    break
                
                # Simple state transition (would be more complex in practice)
                state = state + torch.randn_like(state) * 0.1
        
        return plan
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)


class MissionObjective:
    """Represents a mission objective for planning"""
    
    def __init__(
        self,
        objective_type: str,
        target_position: Optional[np.ndarray] = None,
        priority: int = 1,
        time_constraint: Optional[float] = None,
        energy_constraint: Optional[float] = None
    ):
        """Initialize mission objective
        
        Args:
            objective_type: Type of objective
            target_position: Target position if applicable
            priority: Priority level (1-5)
            time_constraint: Time limit in seconds
            energy_constraint: Energy budget in Joules
        """
        self.objective_type = objective_type
        self.target_position = target_position
        self.priority = priority
        self.time_constraint = time_constraint
        self.energy_constraint = energy_constraint
        self.completed = False
        self.progress = 0.0
    
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert objective to tensor representation
        
        Args:
            device: Torch device
            
        Returns:
            Tensor representation
        """
        features = []
        
        # Objective type encoding (one-hot)
        type_encoding = {
            "search": [1, 0, 0, 0],
            "rescue": [0, 1, 0, 0],
            "formation": [0, 0, 1, 0],
            "surveillance": [0, 0, 0, 1]
        }
        features.extend(type_encoding.get(self.objective_type, [0, 0, 0, 0]))
        
        # Target position
        if self.target_position is not None:
            features.extend(self.target_position.tolist())
        else:
            features.extend([0, 0, 0])
        
        # Priority and constraints
        features.append(self.priority / 5.0)
        features.append(self.time_constraint / 300.0 if self.time_constraint else 0)
        features.append(self.energy_constraint / 1000.0 if self.energy_constraint else 0)
        features.append(float(self.completed))
        features.append(self.progress)
        
        return torch.tensor(features, dtype=torch.float32, device=device)