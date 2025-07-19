"""Base Multi-Agent Environment for PI-HMARL

This module implements the foundational multi-agent environment framework
that supports hierarchical learning, physics integration, and real-parameter
synthetic scenarios.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import logging

from ..data.real_parameter_extractor import RealParameterExtractor, DroneSpecifications
from .agent_manager import AgentManager
from .spaces import ObservationSpace, ActionSpace
from .communication import CommunicationProtocol
from .state_manager import StateManager
from .reward_calculator import RewardCalculator
from .episode_manager import EpisodeManager

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Configuration for multi-agent environment"""
    # Environment parameters
    world_size: Tuple[float, float, float] = (200.0, 200.0, 50.0)
    timestep: float = 0.01  # 100Hz
    max_episode_steps: int = 10000
    
    # Agent parameters
    min_agents: int = 2
    max_agents: int = 50
    default_agent_type: str = "dji_mavic_3"
    
    # Physics parameters
    enable_physics: bool = True
    enable_collisions: bool = True
    enable_energy_constraints: bool = True
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    
    # Communication parameters
    enable_communication: bool = True
    communication_range: float = 50.0  # meters
    communication_latency: float = 0.001  # seconds
    
    # Observation parameters
    observation_type: str = "hierarchical"  # hierarchical, local, global
    partial_observability: bool = True
    sensor_range: float = 30.0  # meters
    
    # Action parameters
    action_type: str = "continuous"  # continuous, discrete
    action_repeat: int = 1
    
    # Reward parameters
    reward_type: str = "multi_objective"
    sparse_rewards: bool = False
    
    # Rendering
    render_mode: Optional[str] = None  # None, "human", "rgb_array"


class MultiAgentEnvironment(gym.Env):
    """Base multi-agent environment with Gymnasium interface"""
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        parameter_extractor: Optional[RealParameterExtractor] = None
    ):
        """Initialize multi-agent environment
        
        Args:
            config: Environment configuration
            parameter_extractor: Real parameter extractor for physics
        """
        super().__init__()
        
        self.config = config or EnvConfig()
        self.parameter_extractor = parameter_extractor or RealParameterExtractor()
        
        # Initialize components
        self.agent_manager = AgentManager(
            min_agents=self.config.min_agents,
            max_agents=self.config.max_agents,
            parameter_extractor=self.parameter_extractor
        )
        
        self.observation_space_manager = ObservationSpace(
            world_size=self.config.world_size,
            sensor_range=self.config.sensor_range,
            observation_type=self.config.observation_type
        )
        
        self.action_space_manager = ActionSpace(
            action_type=self.config.action_type,
            parameter_extractor=self.parameter_extractor
        )
        
        self.communication_protocol = CommunicationProtocol(
            communication_range=self.config.communication_range,
            latency=self.config.communication_latency,
            parameter_extractor=self.parameter_extractor
        )
        
        self.state_manager = StateManager(
            world_size=self.config.world_size,
            enable_physics=self.config.enable_physics
        )
        
        self.reward_calculator = RewardCalculator(
            reward_type=self.config.reward_type,
            sparse_rewards=self.config.sparse_rewards
        )
        
        self.episode_manager = EpisodeManager(
            max_steps=self.config.max_episode_steps,
            timestep=self.config.timestep
        )
        
        # Set up spaces
        self._setup_spaces()
        
        # Rendering
        self.render_mode = self.config.render_mode
        self.viewer = None
        
        # Episode state
        self._reset_episode_state()
        
        logger.info(f"Initialized MultiAgentEnvironment with config: {self.config}")
    
    def _setup_spaces(self):
        """Set up observation and action spaces"""
        # Get a sample agent to determine spaces
        sample_agent_type = self.config.default_agent_type
        
        # Observation space (per agent)
        self.observation_space = self.observation_space_manager.get_observation_space()
        
        # Action space (per agent)
        self.action_space = self.action_space_manager.get_action_space(sample_agent_type)
    
    def _reset_episode_state(self):
        """Reset episode-specific state"""
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False
        self.truncated = False
        self.info = {}
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
        """Reset environment to initial state
        
        Args:
            seed: Random seed
            options: Reset options (num_agents, agent_types, initial_positions, etc.)
            
        Returns:
            observations: Dict mapping agent_id to observation
            infos: Dict mapping agent_id to info dict
        """
        super().reset(seed=seed)
        
        # Reset episode state
        self._reset_episode_state()
        
        # Parse options
        options = options or {}
        num_agents = options.get("num_agents", np.random.randint(
            self.config.min_agents, 
            min(self.config.max_agents, 10) + 1
        ))
        agent_types = options.get("agent_types", 
                                 [self.config.default_agent_type] * num_agents)
        initial_positions = options.get("initial_positions", None)
        scenario = options.get("scenario", "default")
        
        # Reset agent manager
        self.agent_manager.reset(num_agents, agent_types, initial_positions)
        
        # Reset state manager
        self.state_manager.reset(self.agent_manager.get_all_agents())
        
        # Reset communication
        self.communication_protocol.reset(self.agent_manager.active_agents)
        
        # Reset episode manager
        self.episode_manager.reset(scenario)
        
        # Get initial observations
        observations = self._get_observations()
        infos = self._get_infos()
        
        logger.debug(f"Environment reset with {num_agents} agents")
        
        return observations, infos
    
    def step(
        self,
        actions: Dict[int, np.ndarray]
    ) -> Tuple[
        Dict[int, np.ndarray],  # observations
        Dict[int, float],       # rewards
        Dict[int, bool],        # terminated
        Dict[int, bool],        # truncated
        Dict[int, Dict[str, Any]]  # info
    ]:
        """Execute one environment step
        
        Args:
            actions: Dict mapping agent_id to action
            
        Returns:
            observations: Next observations for each agent
            rewards: Rewards for each agent
            terminated: Whether each agent is done
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Validate actions
        if not self._validate_actions(actions):
            raise ValueError("Invalid actions provided")
        
        # Apply action repeat
        for _ in range(self.config.action_repeat):
            # Update physics state
            if self.config.enable_physics:
                self.state_manager.apply_actions(actions, self.config.timestep)
            
            # Update communication
            if self.config.enable_communication:
                messages = self._collect_messages(actions)
                self.communication_protocol.update(
                    self.state_manager.get_positions(),
                    messages,
                    self.config.timestep
                )
            
            # Step physics simulation
            self.state_manager.step(self.config.timestep)
        
        # Update episode manager
        self.episode_manager.step()
        self.current_step += 1
        
        # Check for agent additions/removals
        self._handle_dynamic_agents()
        
        # Get observations
        observations = self._get_observations()
        
        # Calculate rewards
        rewards = self.reward_calculator.calculate_rewards(
            self.state_manager,
            self.agent_manager,
            self.episode_manager,
            actions
        )
        
        # Check termination conditions
        terminated = self._check_terminations()
        truncated = self._check_truncation()
        
        # Get info
        infos = self._get_infos()
        
        # Update episode reward
        self.episode_reward += sum(rewards.values())
        
        return observations, rewards, terminated, truncated, infos
    
    def _validate_actions(self, actions: Dict[int, np.ndarray]) -> bool:
        """Validate provided actions
        
        Args:
            actions: Actions to validate
            
        Returns:
            Whether actions are valid
        """
        for agent_id, action in actions.items():
            if agent_id not in self.agent_manager.active_agents:
                logger.warning(f"Action provided for inactive agent {agent_id}")
                return False
            
            # Check action shape and bounds
            if not self.action_space.contains(action):
                logger.warning(f"Invalid action for agent {agent_id}")
                return False
        
        return True
    
    def _collect_messages(self, actions: Dict[int, np.ndarray]) -> Dict[int, Any]:
        """Collect messages from agent actions
        
        Args:
            actions: Agent actions
            
        Returns:
            Messages to send
        """
        messages = {}
        
        # Extract communication components from actions if present
        for agent_id, action in actions.items():
            if self.config.action_type == "continuous" and len(action) > 4:
                # Assume last elements are communication
                message = action[4:]  # After control actions
                messages[agent_id] = message
            elif self.config.action_type == "discrete":
                # Could have discrete message type
                messages[agent_id] = None
        
        return messages
    
    def _handle_dynamic_agents(self):
        """Handle dynamic agent addition/removal"""
        # Check for agents to remove (e.g., battery depleted)
        agents_to_remove = []
        
        for agent_id in self.agent_manager.active_agents:
            agent_state = self.state_manager.get_agent_state(agent_id)
            
            # Remove if battery depleted
            if agent_state["battery_soc"] <= 0.0:
                agents_to_remove.append(agent_id)
                logger.info(f"Removing agent {agent_id} due to battery depletion")
        
        # Remove agents
        for agent_id in agents_to_remove:
            self.agent_manager.remove_agent(agent_id)
            self.state_manager.remove_agent(agent_id)
            self.communication_protocol.remove_agent(agent_id)
        
        # Could also add new agents based on scenario requirements
        # This would be triggered by episode manager or external events
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all active agents
        
        Returns:
            Dict mapping agent_id to observation
        """
        observations = {}
        
        for agent_id in self.agent_manager.active_agents:
            # Get agent state
            agent_state = self.state_manager.get_agent_state(agent_id)
            
            # Get nearby agents
            nearby_agents = self._get_nearby_agents(agent_id)
            
            # Get communication data
            messages = self.communication_protocol.get_received_messages(agent_id)
            
            # Build observation
            observation = self.observation_space_manager.build_observation(
                agent_id=agent_id,
                agent_state=agent_state,
                nearby_agents=nearby_agents,
                messages=messages,
                global_state=self.state_manager.get_global_state() 
                    if not self.config.partial_observability else None
            )
            
            observations[agent_id] = observation
        
        return observations
    
    def _get_nearby_agents(self, agent_id: int) -> List[Dict[str, Any]]:
        """Get information about nearby agents
        
        Args:
            agent_id: ID of observing agent
            
        Returns:
            List of nearby agent information
        """
        nearby = []
        agent_pos = self.state_manager.get_agent_state(agent_id)["position"]
        
        for other_id in self.agent_manager.active_agents:
            if other_id == agent_id:
                continue
            
            other_state = self.state_manager.get_agent_state(other_id)
            distance = np.linalg.norm(other_state["position"] - agent_pos)
            
            if distance <= self.config.sensor_range:
                nearby.append({
                    "id": other_id,
                    "relative_position": other_state["position"] - agent_pos,
                    "relative_velocity": other_state["velocity"] - 
                                       self.state_manager.get_agent_state(agent_id)["velocity"],
                    "distance": distance
                })
        
        return nearby
    
    def _check_terminations(self) -> Dict[int, bool]:
        """Check termination conditions for each agent
        
        Returns:
            Dict mapping agent_id to terminated status
        """
        terminated = {}
        
        for agent_id in self.agent_manager.active_agents:
            agent_state = self.state_manager.get_agent_state(agent_id)
            
            # Check various termination conditions
            agent_terminated = False
            
            # Battery depleted
            if agent_state["battery_soc"] <= 0.0:
                agent_terminated = True
            
            # Collision
            if self.config.enable_collisions:
                if agent_state.get("collision", False):
                    agent_terminated = True
            
            # Out of bounds
            pos = agent_state["position"]
            if (pos[0] < 0 or pos[0] > self.config.world_size[0] or
                pos[1] < 0 or pos[1] > self.config.world_size[1] or
                pos[2] < 0 or pos[2] > self.config.world_size[2]):
                agent_terminated = True
            
            # Mission completed (scenario-specific)
            if self.episode_manager.is_mission_completed(agent_id):
                agent_terminated = True
            
            terminated[agent_id] = agent_terminated
        
        return terminated
    
    def _check_truncation(self) -> Dict[int, bool]:
        """Check if episode should be truncated
        
        Returns:
            Dict mapping agent_id to truncated status
        """
        # Episode truncated if max steps reached
        episode_truncated = self.current_step >= self.config.max_episode_steps
        
        # All agents get same truncation status
        truncated = {
            agent_id: episode_truncated 
            for agent_id in self.agent_manager.active_agents
        }
        
        return truncated
    
    def _get_infos(self) -> Dict[int, Dict[str, Any]]:
        """Get info dictionaries for all agents
        
        Returns:
            Dict mapping agent_id to info dict
        """
        infos = {}
        
        for agent_id in self.agent_manager.active_agents:
            agent_state = self.state_manager.get_agent_state(agent_id)
            
            info = {
                "position": agent_state["position"].copy(),
                "velocity": agent_state["velocity"].copy(),
                "battery_soc": agent_state["battery_soc"],
                "power_consumption": agent_state.get("power_consumption", 0.0),
                "communication_active": len(
                    self.communication_protocol.get_connected_agents(agent_id)
                ) > 0,
                "episode_step": self.current_step,
                "physics_violations": self.state_manager.get_constraint_violations(agent_id)
            }
            
            # Add scenario-specific info
            scenario_info = self.episode_manager.get_agent_info(agent_id)
            info.update(scenario_info)
            
            infos[agent_id] = info
        
        return infos
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return None
        
        # Import here to avoid dependency if not rendering
        from .visualization import EnvironmentVisualizer
        
        if self.viewer is None:
            self.viewer = EnvironmentVisualizer(
                world_size=self.config.world_size,
                render_mode=self.render_mode
            )
        
        # Get current state for rendering
        agent_states = {
            agent_id: self.state_manager.get_agent_state(agent_id)
            for agent_id in self.agent_manager.active_agents
        }
        
        obstacles = self.state_manager.get_obstacles()
        targets = self.episode_manager.get_targets()
        
        return self.viewer.render(
            agent_states=agent_states,
            obstacles=obstacles,
            targets=targets,
            timestep=self.current_step * self.config.timestep
        )
    
    def close(self):
        """Clean up environment resources"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
        logger.info("Environment closed")
    
    def get_agent_ids(self) -> List[int]:
        """Get list of active agent IDs
        
        Returns:
            List of agent IDs
        """
        return list(self.agent_manager.active_agents)
    
    def get_num_agents(self) -> int:
        """Get number of active agents
        
        Returns:
            Number of agents
        """
        return len(self.agent_manager.active_agents)
    
    def add_agent(
        self,
        agent_type: str = None,
        position: Optional[np.ndarray] = None
    ) -> int:
        """Add a new agent to the environment
        
        Args:
            agent_type: Type of agent
            position: Initial position
            
        Returns:
            New agent ID
        """
        if self.get_num_agents() >= self.config.max_agents:
            raise ValueError("Maximum number of agents reached")
        
        agent_type = agent_type or self.config.default_agent_type
        
        # Add to agent manager
        agent_id = self.agent_manager.add_agent(agent_type, position)
        
        # Add to state manager
        self.state_manager.add_agent(
            agent_id,
            self.agent_manager.get_agent(agent_id)
        )
        
        # Add to communication
        self.communication_protocol.add_agent(agent_id)
        
        logger.info(f"Added agent {agent_id} of type {agent_type}")
        
        return agent_id
    
    def remove_agent(self, agent_id: int):
        """Remove an agent from the environment
        
        Args:
            agent_id: ID of agent to remove
        """
        if agent_id not in self.agent_manager.active_agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Remove from all components
        self.agent_manager.remove_agent(agent_id)
        self.state_manager.remove_agent(agent_id)
        self.communication_protocol.remove_agent(agent_id)
        
        logger.info(f"Removed agent {agent_id}")


# Convenience function
def create_environment(
    config: Optional[EnvConfig] = None,
    parameter_extractor: Optional[RealParameterExtractor] = None
) -> MultiAgentEnvironment:
    """Create a multi-agent environment
    
    Args:
        config: Environment configuration
        parameter_extractor: Real parameter extractor
        
    Returns:
        MultiAgentEnvironment instance
    """
    return MultiAgentEnvironment(config, parameter_extractor)