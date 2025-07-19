"""Observation and Action Space Definitions

This module defines observation and action spaces for multi-agent
scenarios using real drone capabilities.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from ..data.real_parameter_extractor import RealParameterExtractor

logger = logging.getLogger(__name__)


@dataclass
class ObservationConfig:
    """Configuration for observation space"""
    # Local observation parameters
    max_nearby_agents: int = 10
    include_relative_positions: bool = True
    include_relative_velocities: bool = True
    include_communication: bool = True
    
    # Self observation parameters
    include_position: bool = True
    include_velocity: bool = True
    include_orientation: bool = True
    include_battery: bool = True
    include_motor_state: bool = False
    
    # Global observation parameters (if not partial)
    include_global_state: bool = False
    include_all_agents: bool = False
    
    # Sensor parameters
    position_noise_std: float = 0.1  # meters
    velocity_noise_std: float = 0.05  # m/s
    orientation_noise_std: float = 0.01  # radians


class ObservationSpace:
    """Manages observation space for agents"""
    
    def __init__(
        self,
        world_size: Tuple[float, float, float],
        sensor_range: float = 30.0,
        observation_type: str = "hierarchical",
        config: Optional[ObservationConfig] = None
    ):
        """Initialize observation space
        
        Args:
            world_size: Size of the world
            sensor_range: Sensor detection range
            observation_type: Type of observation (local, global, hierarchical)
            config: Observation configuration
        """
        self.world_size = world_size
        self.sensor_range = sensor_range
        self.observation_type = observation_type
        self.config = config or ObservationConfig()
        
        # Build observation space
        self._build_observation_space()
        
        logger.info(f"Initialized ObservationSpace (type: {observation_type})")
    
    def _build_observation_space(self):
        """Build the observation space based on configuration"""
        observation_dims = []
        
        # Self observation
        if self.config.include_position:
            observation_dims.append(("position", 3))
        
        if self.config.include_velocity:
            observation_dims.append(("velocity", 3))
        
        if self.config.include_orientation:
            observation_dims.append(("orientation", 4))  # Quaternion
        
        if self.config.include_battery:
            observation_dims.append(("battery", 1))
        
        if self.config.include_motor_state:
            observation_dims.append(("motor_speeds", 4))
        
        # Calculate self observation size
        self.self_obs_size = sum(dim for _, dim in observation_dims)
        
        # Nearby agents observation
        nearby_obs_size = 0
        if self.config.include_relative_positions:
            nearby_obs_size += 3
        if self.config.include_relative_velocities:
            nearby_obs_size += 3
        
        self.nearby_agent_obs_size = nearby_obs_size
        self.max_nearby_agents = self.config.max_nearby_agents
        
        # Communication observation
        self.comm_obs_size = 10 if self.config.include_communication else 0
        
        # Total observation size
        total_size = (
            self.self_obs_size +
            self.nearby_agent_obs_size * self.max_nearby_agents +
            self.comm_obs_size
        )
        
        # Global observation (if included)
        if self.config.include_global_state:
            total_size += 10  # Global state features
        
        # Create gymnasium space
        if self.observation_type == "hierarchical":
            # Hierarchical observation with multiple components
            self.observation_space = spaces.Dict({
                "local": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.self_obs_size + 
                          self.nearby_agent_obs_size * self.max_nearby_agents,),
                    dtype=np.float32
                ),
                "communication": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.comm_obs_size,),
                    dtype=np.float32
                ),
                "global": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(10,),
                    dtype=np.float32
                ) if self.config.include_global_state else spaces.Box(
                    low=0, high=0, shape=(0,), dtype=np.float32
                )
            })
        else:
            # Flat observation
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_size,),
                dtype=np.float32
            )
        
        self.total_obs_size = total_size
        logger.debug(f"Observation space size: {total_size}")
    
    def get_observation_space(self) -> spaces.Space:
        """Get the gymnasium observation space
        
        Returns:
            Observation space
        """
        return self.observation_space
    
    def build_observation(
        self,
        agent_id: int,
        agent_state: Dict[str, Any],
        nearby_agents: List[Dict[str, Any]],
        messages: List[Any],
        global_state: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Build observation for an agent
        
        Args:
            agent_id: Agent ID
            agent_state: Current agent state
            nearby_agents: List of nearby agent information
            messages: Received messages
            global_state: Global environment state (if available)
            
        Returns:
            Observation array
        """
        observations = []
        
        # Self observation
        if self.config.include_position:
            pos = agent_state["position"]
            # Normalize position to world size
            norm_pos = pos / np.array(self.world_size)
            # Add noise
            if self.config.position_noise_std > 0:
                norm_pos += np.random.normal(0, self.config.position_noise_std/100, 3)
            observations.extend(norm_pos)
        
        if self.config.include_velocity:
            vel = agent_state["velocity"]
            # Normalize velocity (assume max 30 m/s)
            norm_vel = vel / 30.0
            # Add noise
            if self.config.velocity_noise_std > 0:
                norm_vel += np.random.normal(0, self.config.velocity_noise_std/30, 3)
            observations.extend(norm_vel)
        
        if self.config.include_orientation:
            orientation = agent_state.get("orientation", [0, 0, 0, 1])
            # Add noise to orientation
            if self.config.orientation_noise_std > 0:
                # Add noise as small rotation
                noise_angle = np.random.normal(0, self.config.orientation_noise_std)
                noise_axis = np.random.randn(3)
                noise_axis /= np.linalg.norm(noise_axis)
                # Apply noise (simplified)
                orientation = orientation  # TODO: Proper quaternion multiplication
            observations.extend(orientation)
        
        if self.config.include_battery:
            battery = agent_state.get("battery_soc", 1.0)
            observations.append(battery)
        
        if self.config.include_motor_state:
            motor_speeds = agent_state.get("motor_speeds", np.zeros(4))
            # Normalize motor speeds (assume max 10000 RPM)
            norm_motors = motor_speeds / 10000.0
            observations.extend(norm_motors)
        
        # Nearby agents observation
        nearby_obs = []
        
        # Sort nearby agents by distance
        nearby_agents = sorted(nearby_agents, key=lambda x: x["distance"])
        
        for i in range(self.max_nearby_agents):
            if i < len(nearby_agents):
                neighbor = nearby_agents[i]
                
                if self.config.include_relative_positions:
                    rel_pos = neighbor["relative_position"] / self.sensor_range
                    nearby_obs.extend(rel_pos)
                
                if self.config.include_relative_velocities:
                    rel_vel = neighbor["relative_velocity"] / 30.0
                    nearby_obs.extend(rel_vel)
            else:
                # Pad with zeros
                nearby_obs.extend([0.0] * self.nearby_agent_obs_size)
        
        observations.extend(nearby_obs)
        
        # Communication observation
        if self.config.include_communication:
            comm_obs = self._process_messages(messages)
            observations.extend(comm_obs)
        
        # Global observation
        if self.config.include_global_state and global_state is not None:
            global_obs = self._process_global_state(global_state)
            observations.extend(global_obs)
        
        # Convert to numpy array
        observation = np.array(observations, dtype=np.float32)
        
        # Handle hierarchical observation
        if self.observation_type == "hierarchical":
            local_size = self.self_obs_size + self.nearby_agent_obs_size * self.max_nearby_agents
            
            obs_dict = {
                "local": observation[:local_size],
                "communication": observation[local_size:local_size + self.comm_obs_size],
                "global": observation[local_size + self.comm_obs_size:] 
                         if self.config.include_global_state else np.array([])
            }
            
            return obs_dict
        
        return observation
    
    def _process_messages(self, messages: List[Any]) -> List[float]:
        """Process communication messages into observation
        
        Args:
            messages: List of received messages
            
        Returns:
            Communication observation vector
        """
        comm_obs = np.zeros(self.comm_obs_size)
        
        if messages:
            # Simple encoding: average of message vectors
            # In practice, could use attention or other aggregation
            if isinstance(messages[0], (list, np.ndarray)):
                message_array = np.array(messages[:5])  # Max 5 messages
                comm_obs[:message_array.size] = message_array.flatten()[:self.comm_obs_size]
        
        return comm_obs.tolist()
    
    def _process_global_state(self, global_state: Dict[str, Any]) -> List[float]:
        """Process global state into observation
        
        Args:
            global_state: Global environment state
            
        Returns:
            Global observation vector
        """
        global_obs = []
        
        # Example global features
        global_obs.append(global_state.get("time", 0.0) / 1000.0)  # Normalized time
        global_obs.append(global_state.get("num_agents", 0) / 50.0)  # Normalized agent count
        global_obs.append(global_state.get("mission_progress", 0.0))
        
        # Pad to fixed size
        while len(global_obs) < 10:
            global_obs.append(0.0)
        
        return global_obs[:10]


class ActionSpace:
    """Manages action space for agents"""
    
    def __init__(
        self,
        action_type: str = "continuous",
        parameter_extractor: Optional[RealParameterExtractor] = None
    ):
        """Initialize action space
        
        Args:
            action_type: Type of action space (continuous, discrete)
            parameter_extractor: Real parameter extractor for action limits
        """
        self.action_type = action_type
        self.parameter_extractor = parameter_extractor or RealParameterExtractor()
        
        logger.info(f"Initialized ActionSpace (type: {action_type})")
    
    def get_action_space(self, agent_type: str) -> spaces.Space:
        """Get action space for specific agent type
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Gymnasium action space
        """
        # Get agent specifications
        specs = self.parameter_extractor.get_drone_specs(agent_type)
        if specs is None:
            specs = self.parameter_extractor.get_drone_specs("dji_mavic_3")
        
        if self.action_type == "continuous":
            # Continuous action space
            # Actions: [thrust, roll, pitch, yaw_rate]
            # OR: [vx, vy, vz, yaw_rate] for velocity control
            # OR: [motor1, motor2, motor3, motor4] for direct motor control
            
            # Velocity control (more stable for RL)
            max_velocity = specs.max_speed
            max_vertical_velocity = min(specs.max_ascent_speed, specs.max_descent_speed)
            max_yaw_rate = np.pi  # rad/s
            
            action_space = spaces.Box(
                low=np.array([-max_velocity, -max_velocity, -max_vertical_velocity, -max_yaw_rate]),
                high=np.array([max_velocity, max_velocity, max_vertical_velocity, max_yaw_rate]),
                dtype=np.float32
            )
            
        elif self.action_type == "discrete":
            # Discrete action space
            # Actions: stop, forward, backward, left, right, up, down, rotate_left, rotate_right
            action_space = spaces.Discrete(9)
            
        elif self.action_type == "multi_discrete":
            # Multi-discrete for each control axis
            # [forward/back, left/right, up/down, rotate]
            action_space = spaces.MultiDiscrete([3, 3, 3, 3])
            
        elif self.action_type == "hybrid":
            # Hybrid action space (discrete + continuous)
            # Discrete: mode selection
            # Continuous: control parameters
            action_space = spaces.Dict({
                "mode": spaces.Discrete(4),  # hover, move, follow, land
                "parameters": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(4,),
                    dtype=np.float32
                )
            })
        
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")
        
        return action_space
    
    def process_action(
        self,
        action: np.ndarray,
        agent_type: str,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process raw action into control commands
        
        Args:
            action: Raw action from policy
            agent_type: Type of agent
            current_state: Current agent state
            
        Returns:
            Processed control commands
        """
        specs = self.parameter_extractor.get_drone_specs(agent_type)
        if specs is None:
            specs = self.parameter_extractor.get_drone_specs("dji_mavic_3")
        
        if self.action_type == "continuous":
            # Velocity control
            target_velocity = action[:3]
            target_yaw_rate = action[3]
            
            # Apply safety limits
            speed = np.linalg.norm(target_velocity[:2])
            if speed > specs.max_speed:
                target_velocity[:2] *= specs.max_speed / speed
            
            target_velocity[2] = np.clip(
                target_velocity[2],
                -specs.max_descent_speed,
                specs.max_ascent_speed
            )
            
            return {
                "type": "velocity",
                "target_velocity": target_velocity,
                "target_yaw_rate": target_yaw_rate
            }
            
        elif self.action_type == "discrete":
            # Convert discrete action to velocity commands
            action_map = {
                0: [0, 0, 0, 0],      # stop
                1: [5, 0, 0, 0],      # forward
                2: [-5, 0, 0, 0],     # backward
                3: [0, -5, 0, 0],     # left
                4: [0, 5, 0, 0],      # right
                5: [0, 0, 2, 0],      # up
                6: [0, 0, -2, 0],     # down
                7: [0, 0, 0, -0.5],   # rotate left
                8: [0, 0, 0, 0.5],    # rotate right
            }
            
            velocity_command = action_map.get(int(action), [0, 0, 0, 0])
            
            return {
                "type": "velocity",
                "target_velocity": np.array(velocity_command[:3]),
                "target_yaw_rate": velocity_command[3]
            }
        
        else:
            raise ValueError(f"Action processing not implemented for {self.action_type}")
    
    def get_action_meanings(self, action_type: str) -> List[str]:
        """Get human-readable action meanings
        
        Args:
            action_type: Type of action space
            
        Returns:
            List of action meanings
        """
        if action_type == "discrete":
            return [
                "stop",
                "forward",
                "backward", 
                "left",
                "right",
                "up",
                "down",
                "rotate_left",
                "rotate_right"
            ]
        elif action_type == "continuous":
            return [
                "velocity_x",
                "velocity_y",
                "velocity_z",
                "yaw_rate"
            ]
        else:
            return []