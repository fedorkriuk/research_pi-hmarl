"""Agent Manager for Multi-Agent Environment

This module manages agent creation, removal, and state tracking with
support for heterogeneous agents using real specifications.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

from ..data.real_parameter_extractor import RealParameterExtractor, DroneSpecifications

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """Container for agent information"""
    id: int
    type: str
    specifications: DroneSpecifications
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))
    battery_soc: float = 1.0
    active: bool = True
    creation_time: float = 0.0
    
    # Additional state
    motor_speeds: np.ndarray = field(default_factory=lambda: np.zeros(4))
    power_consumption: float = 0.0
    communication_buffer: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation"""
        return {
            "id": self.id,
            "type": self.type,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "orientation": self.orientation.copy(),
            "battery_soc": self.battery_soc,
            "active": self.active,
            "mass": self.specifications.mass,
            "max_speed": self.specifications.max_speed,
            "motor_speeds": self.motor_speeds.copy(),
            "power_consumption": self.power_consumption
        }


class AgentManager:
    """Manages agents in the multi-agent environment"""
    
    def __init__(
        self,
        min_agents: int = 2,
        max_agents: int = 50,
        parameter_extractor: Optional[RealParameterExtractor] = None
    ):
        """Initialize agent manager
        
        Args:
            min_agents: Minimum number of agents
            max_agents: Maximum number of agents
            parameter_extractor: Real parameter extractor for agent specs
        """
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.parameter_extractor = parameter_extractor or RealParameterExtractor()
        
        # Agent storage
        self.agents: Dict[int, Agent] = {}
        self.active_agents: set = set()
        self.next_agent_id: int = 0
        
        # Agent type distribution
        self.agent_type_counts: Dict[str, int] = {}
        
        logger.info(f"Initialized AgentManager (min: {min_agents}, max: {max_agents})")
    
    def reset(
        self,
        num_agents: int,
        agent_types: Optional[List[str]] = None,
        initial_positions: Optional[List[Tuple[float, float, float]]] = None
    ):
        """Reset agent manager with new agents
        
        Args:
            num_agents: Number of agents to create
            agent_types: List of agent types
            initial_positions: Initial positions for agents
        """
        # Validate number of agents
        if num_agents < self.min_agents or num_agents > self.max_agents:
            raise ValueError(
                f"Number of agents must be between {self.min_agents} and {self.max_agents}"
            )
        
        # Clear existing agents
        self.agents.clear()
        self.active_agents.clear()
        self.agent_type_counts.clear()
        self.next_agent_id = 0
        
        # Default agent types
        if agent_types is None:
            agent_types = ["dji_mavic_3"] * num_agents
        
        # Generate initial positions if not provided
        if initial_positions is None:
            initial_positions = self._generate_initial_positions(num_agents)
        
        # Create agents
        for i in range(num_agents):
            agent_type = agent_types[i] if i < len(agent_types) else "dji_mavic_3"
            position = initial_positions[i] if i < len(initial_positions) else (0, 0, 5)
            
            self.add_agent(agent_type, np.array(position))
        
        logger.info(f"Reset with {num_agents} agents")
    
    def _generate_initial_positions(
        self, 
        num_agents: int,
        spacing: float = 5.0
    ) -> List[Tuple[float, float, float]]:
        """Generate initial positions for agents
        
        Args:
            num_agents: Number of positions to generate
            spacing: Minimum spacing between agents
            
        Returns:
            List of initial positions
        """
        positions = []
        
        # Arrange in grid pattern
        grid_size = int(np.ceil(np.sqrt(num_agents)))
        
        for i in range(num_agents):
            row = i // grid_size
            col = i % grid_size
            
            x = col * spacing
            y = row * spacing
            z = 5.0  # Default altitude
            
            positions.append((x, y, z))
        
        # Center the grid
        center_offset = (grid_size - 1) * spacing / 2
        positions = [
            (x - center_offset, y - center_offset, z)
            for x, y, z in positions
        ]
        
        return positions
    
    def add_agent(
        self,
        agent_type: str,
        position: Optional[np.ndarray] = None
    ) -> int:
        """Add a new agent
        
        Args:
            agent_type: Type of agent
            position: Initial position
            
        Returns:
            Agent ID
        """
        if len(self.active_agents) >= self.max_agents:
            raise ValueError(f"Maximum number of agents ({self.max_agents}) reached")
        
        # Get agent specifications
        specs = self.parameter_extractor.get_drone_specs(agent_type)
        if specs is None:
            logger.warning(f"Unknown agent type: {agent_type}, using default")
            specs = self.parameter_extractor.get_drone_specs("dji_mavic_3")
            agent_type = "dji_mavic_3"
        
        # Default position
        if position is None:
            position = np.array([0.0, 0.0, 5.0])
        
        # Create agent
        agent = Agent(
            id=self.next_agent_id,
            type=agent_type,
            specifications=specs,
            position=position
        )
        
        # Store agent
        self.agents[self.next_agent_id] = agent
        self.active_agents.add(self.next_agent_id)
        
        # Update counts
        self.agent_type_counts[agent_type] = self.agent_type_counts.get(agent_type, 0) + 1
        
        # Increment ID counter
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        
        logger.debug(f"Added agent {agent_id} of type {agent_type}")
        
        return agent_id
    
    def remove_agent(self, agent_id: int):
        """Remove an agent
        
        Args:
            agent_id: ID of agent to remove
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        
        # Update counts
        self.agent_type_counts[agent.type] -= 1
        if self.agent_type_counts[agent.type] == 0:
            del self.agent_type_counts[agent.type]
        
        # Remove from active set
        self.active_agents.discard(agent_id)
        
        # Mark as inactive (keep in dict for reference)
        agent.active = False
        
        logger.debug(f"Removed agent {agent_id}")
    
    def get_agent(self, agent_id: int) -> Agent:
        """Get agent by ID
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent object
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        return self.agents[agent_id]
    
    def get_all_agents(self) -> Dict[int, Agent]:
        """Get all agents
        
        Returns:
            Dictionary of all agents
        """
        return self.agents.copy()
    
    def get_active_agents(self) -> List[Agent]:
        """Get list of active agents
        
        Returns:
            List of active agents
        """
        return [self.agents[aid] for aid in self.active_agents]
    
    def update_agent_state(
        self,
        agent_id: int,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        battery_soc: Optional[float] = None,
        motor_speeds: Optional[np.ndarray] = None,
        power_consumption: Optional[float] = None
    ):
        """Update agent state
        
        Args:
            agent_id: Agent ID
            position: New position
            velocity: New velocity
            orientation: New orientation
            battery_soc: New battery state
            motor_speeds: New motor speeds
            power_consumption: New power consumption
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        
        if position is not None:
            agent.position = position.copy()
        if velocity is not None:
            agent.velocity = velocity.copy()
        if orientation is not None:
            agent.orientation = orientation.copy()
        if battery_soc is not None:
            agent.battery_soc = battery_soc
        if motor_speeds is not None:
            agent.motor_speeds = motor_speeds.copy()
        if power_consumption is not None:
            agent.power_consumption = power_consumption
    
    def get_agent_positions(self) -> Dict[int, np.ndarray]:
        """Get positions of all active agents
        
        Returns:
            Dictionary mapping agent_id to position
        """
        return {
            aid: self.agents[aid].position.copy()
            for aid in self.active_agents
        }
    
    def get_agent_velocities(self) -> Dict[int, np.ndarray]:
        """Get velocities of all active agents
        
        Returns:
            Dictionary mapping agent_id to velocity
        """
        return {
            aid: self.agents[aid].velocity.copy()
            for aid in self.active_agents
        }
    
    def get_nearby_agents(
        self,
        agent_id: int,
        radius: float
    ) -> List[int]:
        """Get agents within radius of given agent
        
        Args:
            agent_id: Reference agent ID
            radius: Search radius
            
        Returns:
            List of nearby agent IDs
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        ref_position = self.agents[agent_id].position
        nearby = []
        
        for aid in self.active_agents:
            if aid == agent_id:
                continue
            
            distance = np.linalg.norm(self.agents[aid].position - ref_position)
            if distance <= radius:
                nearby.append(aid)
        
        return nearby
    
    def get_pairwise_distances(self) -> Dict[Tuple[int, int], float]:
        """Get pairwise distances between all active agents
        
        Returns:
            Dictionary mapping (agent_i, agent_j) to distance
        """
        distances = {}
        agent_list = list(self.active_agents)
        
        for i, aid1 in enumerate(agent_list):
            for j, aid2 in enumerate(agent_list[i+1:], i+1):
                distance = np.linalg.norm(
                    self.agents[aid1].position - self.agents[aid2].position
                )
                distances[(aid1, aid2)] = distance
                distances[(aid2, aid1)] = distance
        
        return distances
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about managed agents
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_agents": len(self.agents),
            "active_agents": len(self.active_agents),
            "agent_types": dict(self.agent_type_counts),
            "average_battery": np.mean([
                agent.battery_soc for agent in self.get_active_agents()
            ]) if self.active_agents else 0.0,
            "average_altitude": np.mean([
                agent.position[2] for agent in self.get_active_agents()
            ]) if self.active_agents else 0.0
        }
        
        return stats
    
    def validate_formation(
        self,
        formation_type: str,
        tolerance: float = 1.0
    ) -> Tuple[bool, float]:
        """Validate if agents are in specified formation
        
        Args:
            formation_type: Type of formation (line, v, circle, etc.)
            tolerance: Position tolerance in meters
            
        Returns:
            (is_valid, formation_error)
        """
        if len(self.active_agents) < 2:
            return False, float('inf')
        
        positions = np.array([
            self.agents[aid].position 
            for aid in sorted(self.active_agents)
        ])
        
        if formation_type == "line":
            # Check if agents are in a line
            if len(positions) == 2:
                return True, 0.0
            
            # Fit line and check deviations
            center = np.mean(positions, axis=0)
            centered = positions - center
            
            # PCA to find principal direction
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov[:2, :2])  # Just x,y
            main_direction = eigenvectors[:, np.argmax(eigenvalues)]
            
            # Project points and check deviations
            projections = centered[:, :2] @ main_direction
            perpendicular = centered[:, :2] - np.outer(projections, main_direction)
            errors = np.linalg.norm(perpendicular, axis=1)
            
            max_error = np.max(errors)
            return max_error < tolerance, max_error
        
        elif formation_type == "v":
            # V-formation check
            if len(positions) < 3:
                return False, float('inf')
            
            # Assume first agent is leader
            leader_pos = positions[0]
            followers = positions[1:]
            
            # Check if followers form V shape
            # Simplified: check if they're behind and to sides
            relative_pos = followers - leader_pos
            
            behind = np.all(relative_pos[:, 0] < -2.0)  # Behind leader
            spread = np.std(relative_pos[:, 1]) > 2.0   # Spread to sides
            
            if behind and spread:
                # Calculate deviation from ideal V
                ideal_angle = np.pi / 6  # 30 degrees
                angles = np.arctan2(relative_pos[:, 1], -relative_pos[:, 0])
                angle_errors = np.abs(np.abs(angles) - ideal_angle)
                max_error = np.max(angle_errors) * 10  # Convert to approx meters
                
                return max_error < tolerance, max_error
            
            return False, float('inf')
        
        elif formation_type == "circle":
            # Circle formation
            center = np.mean(positions[:, :2], axis=0)
            distances = np.linalg.norm(positions[:, :2] - center, axis=1)
            
            radius = np.mean(distances)
            errors = np.abs(distances - radius)
            max_error = np.max(errors)
            
            return max_error < tolerance, max_error
        
        else:
            logger.warning(f"Unknown formation type: {formation_type}")
            return False, float('inf')
    
    def get_communication_graph(self, range_limit: float) -> Dict[int, List[int]]:
        """Get communication graph based on range
        
        Args:
            range_limit: Maximum communication range
            
        Returns:
            Adjacency list representation
        """
        graph = {aid: [] for aid in self.active_agents}
        
        for aid1 in self.active_agents:
            nearby = self.get_nearby_agents(aid1, range_limit)
            graph[aid1] = nearby
        
        return graph