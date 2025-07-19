"""Formation Control Scenario

This module implements formation control scenarios where multiple agents
maintain specific geometric formations while navigating.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class FormationType(Enum):
    """Types of formations"""
    LINE = "line"
    CIRCLE = "circle"
    WEDGE = "wedge"
    DIAMOND = "diamond"
    SQUARE = "square"
    CUSTOM = "custom"


@dataclass
class FormationPattern:
    """Defines a formation pattern"""
    formation_type: FormationType
    reference_positions: Dict[int, np.ndarray]  # Agent index to relative position
    scale: float = 1.0
    rotation: float = 0.0  # Rotation angle in radians
    
    def get_position(self, agent_index: int, center: np.ndarray) -> np.ndarray:
        """Get agent position in formation
        
        Args:
            agent_index: Agent's index in formation
            center: Formation center
            
        Returns:
            Target position
        """
        if agent_index not in self.reference_positions:
            return center
        
        # Get reference position
        ref_pos = self.reference_positions[agent_index] * self.scale
        
        # Apply rotation
        cos_r = np.cos(self.rotation)
        sin_r = np.sin(self.rotation)
        
        rotated_x = ref_pos[0] * cos_r - ref_pos[1] * sin_r
        rotated_y = ref_pos[0] * sin_r + ref_pos[1] * cos_r
        
        # Translate to center
        return center + np.array([rotated_x, rotated_y, ref_pos[2]])
    
    @staticmethod
    def create_line_formation(num_agents: int, spacing: float = 5.0) -> 'FormationPattern':
        """Create line formation"""
        positions = {}
        for i in range(num_agents):
            offset = (i - (num_agents - 1) / 2) * spacing
            positions[i] = np.array([offset, 0, 0])
        
        return FormationPattern(FormationType.LINE, positions)
    
    @staticmethod
    def create_circle_formation(num_agents: int, radius: float = 10.0) -> 'FormationPattern':
        """Create circle formation"""
        positions = {}
        for i in range(num_agents):
            angle = 2 * np.pi * i / num_agents
            positions[i] = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0
            ])
        
        return FormationPattern(FormationType.CIRCLE, positions)
    
    @staticmethod
    def create_wedge_formation(num_agents: int, angle: float = np.pi/3, spacing: float = 5.0) -> 'FormationPattern':
        """Create wedge/V formation"""
        positions = {}
        positions[0] = np.array([0, 0, 0])  # Leader at front
        
        for i in range(1, num_agents):
            side = 1 if i % 2 == 1 else -1
            index = (i + 1) // 2
            
            x = -index * spacing * np.cos(angle / 2)
            y = side * index * spacing * np.sin(angle / 2)
            
            positions[i] = np.array([x, y, 0])
        
        return FormationPattern(FormationType.WEDGE, positions)


class FormationAgent:
    """Agent capable of formation control"""
    
    def __init__(
        self,
        agent_id: str,
        index: int,
        position: np.ndarray,
        max_speed: float = 10.0,  # OPTIMIZED: Increased from 5.0
        communication_range: float = 50.0
    ):
        """Initialize formation agent
        
        Args:
            agent_id: Unique identifier
            index: Index in formation
            position: Initial position
            max_speed: Maximum movement speed
            communication_range: Communication range
        """
        self.agent_id = agent_id
        self.index = index
        self.position = position.copy()
        self.velocity = np.zeros(3)
        self.max_speed = max_speed
        self.communication_range = communication_range
        
        # Formation state
        self.formation_position = None
        self.formation_error = 0.0
        self.in_formation = False
        
        # Neighbor tracking
        self.neighbors: Dict[str, Dict[str, Any]] = {}
        
        # Control gains - OPTIMIZED for faster convergence
        self.position_gain = 4.0  # Increased from 2.0 for stronger position control
        self.velocity_gain = 1.5  # Increased damping for stability
        self.neighbor_gain = 1.0  # Increased neighbor influence
        
        logger.info(f"Initialized FormationAgent {agent_id}")
    
    def update_neighbors(self, other_agents: List['FormationAgent']):
        """Update neighbor information
        
        Args:
            other_agents: List of other agents
        """
        self.neighbors.clear()
        
        for agent in other_agents:
            if agent.agent_id == self.agent_id:
                continue
            
            distance = np.linalg.norm(agent.position - self.position)
            
            if distance <= self.communication_range:
                self.neighbors[agent.agent_id] = {
                    'position': agent.position.copy(),
                    'velocity': agent.velocity.copy(),
                    'distance': distance,
                    'index': agent.index
                }
    
    def compute_formation_control(
        self,
        formation: FormationPattern,
        formation_center: np.ndarray,
        formation_velocity: np.ndarray
    ) -> np.ndarray:
        """Compute formation control input
        
        Args:
            formation: Formation pattern
            formation_center: Center of formation
            formation_velocity: Formation velocity
            
        Returns:
            Control velocity
        """
        # Get target position in formation
        target_position = formation.get_position(self.index, formation_center)
        self.formation_position = target_position
        
        # Position error
        position_error = target_position - self.position
        self.formation_error = np.linalg.norm(position_error)
        
        # Basic proportional control
        control = self.position_gain * position_error
        
        # Add formation velocity feedforward
        control += formation_velocity
        
        # Add velocity damping
        control -= self.velocity_gain * self.velocity
        
        # Add consensus term from neighbors
        if self.neighbors:
            consensus = np.zeros(3)
            
            for neighbor_id, neighbor_info in self.neighbors.items():
                # Relative position to neighbor's formation position
                neighbor_target = formation.get_position(
                    neighbor_info['index'],
                    formation_center
                )
                
                relative_error = (
                    (neighbor_info['position'] - self.position) -
                    (neighbor_target - target_position)
                )
                
                consensus += self.neighbor_gain * relative_error
            
            control += consensus / len(self.neighbors)
        
        # Check if in formation
        self.in_formation = self.formation_error < 1.0
        
        return control
    
    def apply_control(self, control: np.ndarray, dt: float):
        """Apply control input
        
        Args:
            control: Control velocity
            dt: Time step
        """
        # Limit control magnitude
        control_magnitude = np.linalg.norm(control)
        if control_magnitude > self.max_speed:
            control = control * self.max_speed / control_magnitude
        
        # Update velocity
        self.velocity = control
        
        # Update position
        self.position += self.velocity * dt
    
    def avoid_obstacles(
        self,
        obstacles: List[Dict[str, Any]],
        avoidance_gain: float = 5.0
    ) -> np.ndarray:
        """Compute obstacle avoidance
        
        Args:
            obstacles: List of obstacles
            avoidance_gain: Avoidance control gain
            
        Returns:
            Avoidance velocity
        """
        avoidance = np.zeros(3)
        
        for obstacle in obstacles:
            obs_pos = obstacle['position']
            obs_radius = obstacle['radius']
            
            # Vector from obstacle to agent
            diff = self.position - obs_pos
            distance = np.linalg.norm(diff)
            
            # Check if within influence range
            influence_range = obs_radius + 5.0
            
            if distance < influence_range and distance > 0:
                # Repulsive force
                force_magnitude = avoidance_gain * (1.0 / distance - 1.0 / influence_range)
                avoidance += force_magnitude * diff / distance
        
        return avoidance


class ObstacleAvoidance:
    """Obstacle avoidance for formation"""
    
    def __init__(self, safety_margin: float = 2.0):
        """Initialize obstacle avoidance
        
        Args:
            safety_margin: Safety margin around obstacles
        """
        self.safety_margin = safety_margin
        self.avoidance_gain = 10.0
        self.formation_deform_gain = 0.5
    
    def compute_formation_deformation(
        self,
        agents: List[FormationAgent],
        obstacles: List[Dict[str, Any]],
        formation: FormationPattern
    ) -> Dict[int, np.ndarray]:
        """Compute formation deformation for obstacle avoidance
        
        Args:
            agents: List of agents
            obstacles: List of obstacles
            formation: Current formation
            
        Returns:
            Position adjustments for each agent
        """
        adjustments = {}
        
        for agent in agents:
            # Check if agent's formation position conflicts with obstacles
            if agent.formation_position is None:
                continue
            
            total_adjustment = np.zeros(3)
            
            for obstacle in obstacles:
                obs_pos = obstacle['position']
                obs_radius = obstacle['radius'] + self.safety_margin
                
                # Distance from formation position to obstacle
                diff = agent.formation_position - obs_pos
                distance = np.linalg.norm(diff[:2])  # 2D distance
                
                if distance < obs_radius:
                    # Need to adjust position
                    if distance > 0:
                        # Push away from obstacle
                        adjustment = diff[:2] / distance * (obs_radius - distance)
                        total_adjustment[:2] += adjustment
                    else:
                        # At obstacle center, push in random direction
                        angle = np.random.random() * 2 * np.pi
                        total_adjustment[:2] += obs_radius * np.array([np.cos(angle), np.sin(angle)])
            
            adjustments[agent.index] = total_adjustment * self.formation_deform_gain
        
        return adjustments


class DynamicFormation:
    """Dynamic formation that can change shape"""
    
    def __init__(self):
        """Initialize dynamic formation"""
        self.current_formation = None
        self.target_formation = None
        self.transition_progress = 1.0
        self.transition_duration = 5.0
        self.transition_start_time = 0.0
    
    def set_formation(self, formation: FormationPattern):
        """Set current formation
        
        Args:
            formation: Formation pattern
        """
        self.current_formation = formation
        self.target_formation = formation
        self.transition_progress = 1.0
    
    def transition_to(
        self,
        new_formation: FormationPattern,
        duration: float,
        current_time: float
    ):
        """Start transition to new formation
        
        Args:
            new_formation: Target formation
            duration: Transition duration
            current_time: Current time
        """
        if self.current_formation is None:
            self.set_formation(new_formation)
            return
        
        self.target_formation = new_formation
        self.transition_duration = duration
        self.transition_start_time = current_time
        self.transition_progress = 0.0
    
    def update(self, current_time: float):
        """Update formation transition
        
        Args:
            current_time: Current time
        """
        if self.transition_progress < 1.0:
            elapsed = current_time - self.transition_start_time
            self.transition_progress = min(1.0, elapsed / self.transition_duration)
            
            if self.transition_progress >= 1.0:
                self.current_formation = self.target_formation
    
    def get_interpolated_position(
        self,
        agent_index: int,
        center: np.ndarray
    ) -> np.ndarray:
        """Get interpolated position during transition
        
        Args:
            agent_index: Agent index
            center: Formation center
            
        Returns:
            Interpolated position
        """
        if self.transition_progress >= 1.0:
            return self.current_formation.get_position(agent_index, center)
        
        # Get positions in both formations
        current_pos = self.current_formation.get_position(agent_index, center)
        target_pos = self.target_formation.get_position(agent_index, center)
        
        # Smooth interpolation
        t = self.transition_progress
        smooth_t = t * t * (3 - 2 * t)  # Smoothstep
        
        return current_pos + smooth_t * (target_pos - current_pos)


class FormationController:
    """Controls formation movement and behavior"""
    
    def __init__(self, num_agents: int):
        """Initialize formation controller
        
        Args:
            num_agents: Number of agents
        """
        self.num_agents = num_agents
        self.formation = FormationPattern.create_line_formation(num_agents)
        self.dynamic_formation = DynamicFormation()
        self.dynamic_formation.set_formation(self.formation)
        
        # Formation state
        self.center_position = np.array([50.0, 50.0, 0.0])
        self.target_position = self.center_position.copy()
        self.velocity = np.zeros(3)
        self.heading = 0.0
        
        # Path following
        self.waypoints = []
        self.current_waypoint_index = 0
        
        # Control parameters - OPTIMIZED for faster navigation
        self.max_speed = 8.0  # Increased from 2.0 for much faster movement
        self.arrival_threshold = 5.0  # Increased for easier waypoint hitting
        
        logger.info(f"Initialized FormationController for {num_agents} agents")
    
    def set_waypoints(self, waypoints: List[np.ndarray]):
        """Set waypoints for formation to follow
        
        Args:
            waypoints: List of waypoints
        """
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        
        if waypoints:
            self.target_position = waypoints[0]
    
    def update(self, dt: float, current_time: float):
        """Update formation controller
        
        Args:
            dt: Time step
            current_time: Current time
        """
        # Update dynamic formation
        self.dynamic_formation.update(current_time)
        
        # Path following
        if self.waypoints and self.current_waypoint_index < len(self.waypoints):
            target = self.waypoints[self.current_waypoint_index]
            
            # Check arrival
            distance = np.linalg.norm(self.center_position - target)
            
            if distance < self.arrival_threshold:
                # Move to next waypoint
                self.current_waypoint_index += 1
                
                if self.current_waypoint_index < len(self.waypoints):
                    self.target_position = self.waypoints[self.current_waypoint_index]
            
            # Compute velocity towards target - OPTIMIZED
            if distance > 0:
                direction = (target - self.center_position) / distance
                # Use more aggressive speed scaling for faster progress
                speed = min(self.max_speed, max(distance * 0.5, 2.0))  # At least 2.0 speed
                self.velocity = direction * speed
                
                # Update heading
                self.heading = np.arctan2(direction[1], direction[0])
        else:
            # No waypoints or reached end
            self.velocity = np.zeros(3)
        
        # Update center position
        self.center_position += self.velocity * dt
        
        # Update formation orientation
        if np.linalg.norm(self.velocity[:2]) > 0.1:
            self.dynamic_formation.current_formation.rotation = self.heading
    
    def change_formation(
        self,
        new_formation_type: FormationType,
        current_time: float,
        transition_duration: float = 5.0
    ):
        """Change formation type
        
        Args:
            new_formation_type: New formation type
            current_time: Current time
            transition_duration: Time for transition
        """
        if new_formation_type == FormationType.LINE:
            new_formation = FormationPattern.create_line_formation(self.num_agents)
        elif new_formation_type == FormationType.CIRCLE:
            new_formation = FormationPattern.create_circle_formation(self.num_agents)
        elif new_formation_type == FormationType.WEDGE:
            new_formation = FormationPattern.create_wedge_formation(self.num_agents)
        else:
            return
        
        self.dynamic_formation.transition_to(
            new_formation,
            transition_duration,
            current_time
        )
    
    def get_formation_quality(self, agents: List[FormationAgent]) -> float:
        """Calculate formation quality metric
        
        Args:
            agents: List of agents
            
        Returns:
            Quality score (0-1)
        """
        if not agents:
            return 0.0
        
        total_error = 0.0
        in_formation_count = 0
        
        for agent in agents:
            total_error += agent.formation_error
            
            if agent.in_formation:
                in_formation_count += 1
        
        # Average error (normalized)
        avg_error = total_error / len(agents)
        error_score = np.exp(-avg_error / 5.0)  # Exponential decay
        
        # Formation percentage
        formation_score = in_formation_count / len(agents)
        
        # Combined score
        return 0.5 * error_score + 0.5 * formation_score


class FormationControlScenario:
    """Main formation control scenario"""
    
    def __init__(
        self,
        num_agents: int = 6,
        environment_size: Tuple[float, float] = (200.0, 200.0),
        num_obstacles: int = 5
    ):
        """Initialize formation control scenario
        
        Args:
            num_agents: Number of agents
            environment_size: Size of environment
            num_obstacles: Number of obstacles
        """
        self.num_agents = num_agents
        self.environment_size = environment_size
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize controller
        self.controller = FormationController(num_agents)
        
        # Initialize obstacles
        self.obstacles = self._generate_obstacles(num_obstacles)
        self.obstacle_avoidance = ObstacleAvoidance()
        
        # Scenario state
        self.time = 0.0
        self.formation_history = []
        
        logger.info(
            f"Initialized FormationControlScenario with {num_agents} agents"
        )
    
    def _initialize_agents(self) -> List[FormationAgent]:
        """Initialize agents
        
        Returns:
            List of agents
        """
        agents = []
        
        # Start in a rough circle
        radius = 20.0
        center = np.array([self.environment_size[0] / 2, self.environment_size[1] / 2, 0])
        
        for i in range(self.num_agents):
            angle = 2 * np.pi * i / self.num_agents
            position = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
            
            agent = FormationAgent(
                agent_id=f"agent_{i}",
                index=i,
                position=position,
                max_speed=5.0,
                communication_range=50.0
            )
            
            agents.append(agent)
        
        return agents
    
    def _generate_obstacles(self, num_obstacles: int) -> List[Dict[str, Any]]:
        """Generate random obstacles
        
        Args:
            num_obstacles: Number of obstacles
            
        Returns:
            List of obstacles
        """
        obstacles = []
        
        for i in range(num_obstacles):
            position = np.array([
                np.random.uniform(20, self.environment_size[0] - 20),
                np.random.uniform(20, self.environment_size[1] - 20),
                0
            ])
            
            obstacle = {
                'position': position,
                'radius': np.random.uniform(5, 10),
                'type': 'static'
            }
            
            obstacles.append(obstacle)
        
        return obstacles
    
    def create_mission_waypoints(self) -> List[np.ndarray]:
        """Create waypoints for a mission
        
        Returns:
            List of waypoints
        """
        waypoints = []
        
        # Square patrol pattern
        margin = 30
        corners = [
            np.array([margin, margin, 0]),
            np.array([self.environment_size[0] - margin, margin, 0]),
            np.array([self.environment_size[0] - margin, self.environment_size[1] - margin, 0]),
            np.array([margin, self.environment_size[1] - margin, 0])
        ]
        
        # Add intermediate points for smoother path
        for i in range(len(corners)):
            start = corners[i]
            end = corners[(i + 1) % len(corners)]
            
            # Add start
            waypoints.append(start)
            
            # Add intermediate points
            for t in [0.25, 0.5, 0.75]:
                intermediate = start + t * (end - start)
                waypoints.append(intermediate)
        
        return waypoints
    
    def step(self, dt: float = 0.1):
        """Run one simulation step
        
        Args:
            dt: Time step
        """
        self.time += dt
        
        # Update controller
        self.controller.update(dt, self.time)
        
        # Update agent neighbors
        for agent in self.agents:
            agent.update_neighbors(self.agents)
        
        # Compute obstacle avoidance adjustments
        position_adjustments = self.obstacle_avoidance.compute_formation_deformation(
            self.agents,
            self.obstacles,
            self.controller.dynamic_formation.current_formation
        )
        
        # Control agents
        for agent in self.agents:
            # Get formation position (with transitions)
            formation_pos = self.controller.dynamic_formation.get_interpolated_position(
                agent.index,
                self.controller.center_position
            )
            
            # Apply obstacle avoidance adjustment
            if agent.index in position_adjustments:
                formation_pos += position_adjustments[agent.index]
            
            # Compute formation control
            control = agent.compute_formation_control(
                self.controller.dynamic_formation.current_formation,
                self.controller.center_position,
                self.controller.velocity
            )
            
            # Add obstacle avoidance
            avoidance = agent.avoid_obstacles(self.obstacles)
            control += avoidance
            
            # Apply control
            agent.apply_control(control, dt)
        
        # Record formation quality
        quality = self.controller.get_formation_quality(self.agents)
        self.formation_history.append({
            'time': self.time,
            'quality': quality,
            'formation_type': self.controller.dynamic_formation.current_formation.formation_type.value
        })
        
        # Formation changes based on environment
        self._check_formation_change()
    
    def _check_formation_change(self):
        """Check if formation should change based on environment"""
        # Example: Change formation when passing through narrow areas
        
        # Get formation bounds
        positions = [agent.position for agent in self.agents]
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        formation_width = max_pos[0] - min_pos[0]
        
        # Check for nearby obstacles
        center = self.controller.center_position
        nearby_obstacles = []
        
        for obstacle in self.obstacles:
            distance = np.linalg.norm(obstacle['position'] - center)
            if distance < 30:  # Within 30 units
                nearby_obstacles.append(obstacle)
        
        # Change formation based on conditions
        current_type = self.controller.dynamic_formation.current_formation.formation_type
        
        if len(nearby_obstacles) > 2 and current_type != FormationType.LINE:
            # Many obstacles - use line formation
            self.controller.change_formation(FormationType.LINE, self.time)
        elif len(nearby_obstacles) == 0 and current_type != FormationType.WEDGE:
            # Open space - use wedge formation
            self.controller.change_formation(FormationType.WEDGE, self.time)
    
    def get_state(self) -> Dict[str, Any]:
        """Get scenario state
        
        Returns:
            State dictionary
        """
        formation_quality = self.controller.get_formation_quality(self.agents)
        
        return {
            'time': self.time,
            'formation_type': self.controller.dynamic_formation.current_formation.formation_type.value,
            'formation_quality': formation_quality,
            'formation_center': self.controller.center_position.tolist(),
            'agents': {
                agent.agent_id: {
                    'position': agent.position.tolist(),
                    'velocity': agent.velocity.tolist(),
                    'in_formation': agent.in_formation,
                    'formation_error': agent.formation_error
                }
                for agent in self.agents
            },
            'waypoint_progress': f"{self.controller.current_waypoint_index}/{len(self.controller.waypoints)}"
        }
    
    def run_mission(self):
        """Run a complete mission"""
        # Create mission waypoints
        waypoints = self.create_mission_waypoints()
        self.controller.set_waypoints(waypoints)
        
        # Run simulation
        max_time = 200.0
        dt = 0.1
        
        while self.time < max_time:
            self.step(dt)
            
            # Log periodically
            if int(self.time) % 10 == 0:
                state = self.get_state()
                logger.info(
                    f"Time: {state['time']:.1f}s, "
                    f"Formation: {state['formation_type']}, "
                    f"Quality: {state['formation_quality']:.2f}"
                )
            
            # Check mission completion
            if self.controller.current_waypoint_index >= len(waypoints):
                logger.info("Mission completed!")
                break
        
        # Summary
        avg_quality = np.mean([h['quality'] for h in self.formation_history])
        logger.info(f"Average formation quality: {avg_quality:.2f}")


# Example usage
def run_formation_control():
    """Run formation control scenario"""
    scenario = FormationControlScenario(
        num_agents=7,
        environment_size=(300.0, 300.0),
        num_obstacles=8
    )
    
    scenario.run_mission()