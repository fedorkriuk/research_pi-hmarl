"""Hierarchical State Encoder

This module implements multi-level state representation using real sensor
specifications for hierarchical decision making.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class HierarchicalStateEncoder(nn.Module):
    """Encodes observations into hierarchical state representations"""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """Initialize hierarchical state encoder
        
        Args:
            state_dim: Raw state dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of encoding layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # Local state encoder (high-frequency sensor data)
        self.local_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Tactical state encoder (medium-term patterns)
        self.tactical_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Strategic state encoder (long-term mission state)
        self.strategic_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism for feature aggregation
        if use_attention:
            self.cross_level_attention = CrossLevelAttention(
                hidden_dim=hidden_dim,
                num_heads=4
            )
        
        # Sensor-specific encoders (real sensor specs)
        self.sensor_encoders = self._create_sensor_encoders()
        
        # Time encoding for temporal context
        self.time_encoder = TimeEncoder(hidden_dim)
        
        logger.info("Initialized HierarchicalStateEncoder")
    
    def forward(
        self,
        state: torch.Tensor,
        sensor_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode state hierarchically
        
        Args:
            state: Raw state observation
            sensor_data: Optional sensor-specific data
            
        Returns:
            Local, tactical, and strategic features
        """
        # Encode local features
        local_features = self.local_encoder(state)
        
        # Process sensor-specific data if available
        if sensor_data:
            sensor_features = self._process_sensor_data(sensor_data)
            local_features = local_features + sensor_features
        
        # Encode tactical features
        tactical_features = self.tactical_encoder(local_features)
        
        # Encode strategic features
        strategic_features = self.strategic_encoder(tactical_features)
        
        # Apply cross-level attention if enabled
        if self.use_attention:
            local_features, tactical_features, strategic_features = \
                self.cross_level_attention(
                    local_features, tactical_features, strategic_features
                )
        
        return local_features, tactical_features, strategic_features
    
    def _create_sensor_encoders(self) -> nn.ModuleDict:
        """Create encoders for different sensor types
        
        Returns:
            Dictionary of sensor encoders
        """
        encoders = nn.ModuleDict()
        
        # GPS encoder (position, velocity)
        encoders['gps'] = nn.Sequential(
            nn.Linear(6, 32),  # lat, lon, alt, vx, vy, vz
            nn.ReLU(),
            nn.Linear(32, self.hidden_dim)
        )
        
        # IMU encoder (acceleration, gyroscope)
        encoders['imu'] = nn.Sequential(
            nn.Linear(6, 32),  # ax, ay, az, gx, gy, gz
            nn.ReLU(),
            nn.Linear(32, self.hidden_dim)
        )
        
        # Camera encoder (simplified - would use CNN in practice)
        encoders['camera'] = nn.Sequential(
            nn.Linear(64, 128),  # Flattened features
            nn.ReLU(),
            nn.Linear(128, self.hidden_dim)
        )
        
        # LiDAR encoder (point cloud features)
        encoders['lidar'] = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_dim)
        )
        
        # Battery sensor
        encoders['battery'] = nn.Sequential(
            nn.Linear(4, 16),  # voltage, current, temp, soc
            nn.ReLU(),
            nn.Linear(16, self.hidden_dim)
        )
        
        return encoders
    
    def _process_sensor_data(
        self,
        sensor_data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Process sensor-specific data
        
        Args:
            sensor_data: Dictionary of sensor readings
            
        Returns:
            Aggregated sensor features
        """
        features = []
        
        for sensor_type, data in sensor_data.items():
            if sensor_type in self.sensor_encoders:
                encoded = self.sensor_encoders[sensor_type](data)
                features.append(encoded)
        
        if features:
            # Aggregate features
            aggregated = torch.mean(torch.stack(features), dim=0)
        else:
            aggregated = torch.zeros(
                sensor_data[list(sensor_data.keys())[0]].shape[0],
                self.hidden_dim,
                device=list(sensor_data.values())[0].device
            )
        
        return aggregated


class CrossLevelAttention(nn.Module):
    """Cross-level attention for hierarchical features"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        """Initialize cross-level attention
        
        Args:
            hidden_dim: Feature dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention layers
        self.local_to_tactical = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.tactical_to_strategic = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.strategic_to_tactical = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.tactical_to_local = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        local_features: torch.Tensor,
        tactical_features: torch.Tensor,
        strategic_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply cross-level attention
        
        Args:
            local_features: Local level features
            tactical_features: Tactical level features
            strategic_features: Strategic level features
            
        Returns:
            Updated features for all levels
        """
        # Add batch dimension if needed
        if local_features.dim() == 1:
            local_features = local_features.unsqueeze(0).unsqueeze(0)
            tactical_features = tactical_features.unsqueeze(0).unsqueeze(0)
            strategic_features = strategic_features.unsqueeze(0).unsqueeze(0)
            squeeze_output = True
        else:
            local_features = local_features.unsqueeze(1)
            tactical_features = tactical_features.unsqueeze(1)
            strategic_features = strategic_features.unsqueeze(1)
            squeeze_output = False
        
        # Bottom-up attention
        tactical_update, _ = self.local_to_tactical(
            tactical_features, local_features, local_features
        )
        tactical_features = self.norm1(tactical_features + tactical_update)
        
        strategic_update, _ = self.tactical_to_strategic(
            strategic_features, tactical_features, tactical_features
        )
        strategic_features = self.norm2(strategic_features + strategic_update)
        
        # Top-down attention
        tactical_guidance, _ = self.strategic_to_tactical(
            tactical_features, strategic_features, strategic_features
        )
        tactical_features = self.norm3(tactical_features + tactical_guidance)
        
        local_guidance, _ = self.tactical_to_local(
            local_features, tactical_features, tactical_features
        )
        local_features = local_features + local_guidance
        
        # Remove added dimensions
        if squeeze_output:
            local_features = local_features.squeeze(0).squeeze(0)
            tactical_features = tactical_features.squeeze(0).squeeze(0)
            strategic_features = strategic_features.squeeze(0).squeeze(0)
        else:
            local_features = local_features.squeeze(1)
            tactical_features = tactical_features.squeeze(1)
            strategic_features = strategic_features.squeeze(1)
        
        return local_features, tactical_features, strategic_features


class TimeEncoder(nn.Module):
    """Encodes temporal information for state representation"""
    
    def __init__(self, hidden_dim: int):
        """Initialize time encoder
        
        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        # Sinusoidal encoding for different time scales
        self.time_scales = [1, 60, 3600, 86400]  # sec, min, hour, day
        self.encoding_dim = len(self.time_scales) * 2
        
        self.projection = nn.Linear(self.encoding_dim, hidden_dim)
    
    def forward(self, timestamp: float) -> torch.Tensor:
        """Encode timestamp
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            Time encoding
        """
        encodings = []
        
        for scale in self.time_scales:
            normalized_time = timestamp / scale
            encodings.append(torch.sin(2 * np.pi * normalized_time))
            encodings.append(torch.cos(2 * np.pi * normalized_time))
        
        time_encoding = torch.tensor(encodings, dtype=torch.float32)
        projected = self.projection(time_encoding)
        
        return projected


class StateAggregator:
    """Aggregates multi-agent states for team-level understanding"""
    
    def __init__(self, aggregation_type: str = "mean"):
        """Initialize state aggregator
        
        Args:
            aggregation_type: Type of aggregation (mean, max, attention)
        """
        self.aggregation_type = aggregation_type
    
    def aggregate(
        self,
        agent_states: Dict[int, Dict[str, torch.Tensor]],
        level: str = "tactical"
    ) -> torch.Tensor:
        """Aggregate agent states
        
        Args:
            agent_states: Dictionary of agent states
            level: Hierarchy level to aggregate
            
        Returns:
            Aggregated state
        """
        if not agent_states:
            return torch.zeros(128)  # Default hidden dim
        
        # Extract features for specified level
        features = []
        for agent_id, state_dict in agent_states.items():
            if level in state_dict:
                features.append(state_dict[level])
        
        if not features:
            return torch.zeros_like(list(agent_states.values())[0]["local"])
        
        # Stack features
        stacked = torch.stack(features)
        
        # Aggregate
        if self.aggregation_type == "mean":
            aggregated = torch.mean(stacked, dim=0)
        elif self.aggregation_type == "max":
            aggregated = torch.max(stacked, dim=0)[0]
        elif self.aggregation_type == "sum":
            aggregated = torch.sum(stacked, dim=0)
        else:
            # Default to mean
            aggregated = torch.mean(stacked, dim=0)
        
        return aggregated


class SensorFusion:
    """Fuses data from multiple sensors with real specifications"""
    
    def __init__(self):
        """Initialize sensor fusion"""
        # Real sensor specifications
        self.sensor_specs = {
            "gps": {
                "frequency": 10,  # Hz
                "accuracy": 2.0,  # meters
                "latency": 0.1    # seconds
            },
            "imu": {
                "frequency": 100,  # Hz
                "accuracy": 0.01,  # m/s²
                "latency": 0.01
            },
            "camera": {
                "frequency": 30,   # Hz
                "resolution": (640, 480),
                "latency": 0.033
            },
            "lidar": {
                "frequency": 20,   # Hz
                "range": 100,      # meters
                "latency": 0.05
            }
        }
    
    def fuse(
        self,
        sensor_data: Dict[str, np.ndarray],
        timestamps: Dict[str, float]
    ) -> np.ndarray:
        """Fuse sensor data accounting for latency and accuracy
        
        Args:
            sensor_data: Raw sensor readings
            timestamps: Sensor timestamps
            
        Returns:
            Fused state estimate
        """
        # Weight sensors by accuracy and recency
        weights = {}
        current_time = max(timestamps.values())
        
        for sensor, data in sensor_data.items():
            if sensor in self.sensor_specs:
                spec = self.sensor_specs[sensor]
                
                # Age penalty
                age = current_time - timestamps[sensor]
                age_weight = np.exp(-age / spec["latency"])
                
                # Accuracy weight
                accuracy_weight = 1.0 / spec["accuracy"] if "accuracy" in spec else 1.0
                
                weights[sensor] = age_weight * accuracy_weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Weighted fusion
        fused_state = np.zeros_like(list(sensor_data.values())[0])
        for sensor, data in sensor_data.items():
            if sensor in weights:
                fused_state += weights[sensor] * data
        
        return fused_state