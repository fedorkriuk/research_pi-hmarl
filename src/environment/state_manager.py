"""State Manager for Multi-Agent Environment

This module manages global and local state for all agents, including
physics state, constraints, and environmental factors.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class GlobalState:
    """Container for global environment state"""
    time: float = 0.0
    timestep: float = 0.01
    
    # Environmental conditions
    wind_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    air_density: float = 1.225  # kg/m³
    temperature: float = 20.0   # Celsius
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    
    # Mission state
    mission_progress: float = 0.0
    targets_found: int = 0
    total_targets: int = 0
    
    # System state
    total_energy_consumed: float = 0.0
    total_distance_traveled: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "time": self.time,
            "wind_velocity": self.wind_velocity.tolist(),
            "temperature": self.temperature,
            "mission_progress": self.mission_progress,
            "targets_found": self.targets_found
        }


@dataclass
class AgentPhysicsState:
    """Physics state for a single agent"""
    # Kinematic state
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    orientation: np.ndarray  # Quaternion
    angular_velocity: np.ndarray
    
    # Dynamic state
    forces: np.ndarray
    torques: np.ndarray
    
    # Energy state
    battery_soc: float
    power_consumption: float
    motor_speeds: np.ndarray
    
    # Collision state
    collision: bool = False
    collision_agents: List[int] = field(default_factory=list)
    min_separation: float = float('inf')


class StateManager:
    """Manages environment and agent states"""
    
    def __init__(
        self,
        world_size: Tuple[float, float, float],
        enable_physics: bool = True,
        physics_timestep: float = 0.01
    ):
        """Initialize state manager
        
        Args:
            world_size: Size of the world
            enable_physics: Whether to enable physics simulation
            physics_timestep: Physics simulation timestep
        """
        self.world_size = world_size
        self.enable_physics = enable_physics
        self.physics_timestep = physics_timestep
        
        # Global state
        self.global_state = GlobalState(timestep=physics_timestep)
        
        # Agent states
        self.agent_states: Dict[int, AgentPhysicsState] = {}
        
        # Obstacles
        self.obstacles: List[Dict[str, Any]] = []
        
        # Constraint tracking
        self.constraint_violations: Dict[int, Dict[str, bool]] = {}
        
        # Previous states for derivative calculation
        self._prev_velocities: Dict[int, np.ndarray] = {}
        self._prev_positions: Dict[int, np.ndarray] = {}
        
        logger.info(f"Initialized StateManager with world size {world_size}")
    
    def reset(self, agents: Dict[int, Any]):
        """Reset state manager
        
        Args:
            agents: Dictionary of agents
        """
        # Reset global state
        self.global_state = GlobalState(timestep=self.physics_timestep)
        
        # Reset agent states
        self.agent_states.clear()
        self.constraint_violations.clear()
        self._prev_velocities.clear()
        self._prev_positions.clear()
        
        # Initialize agent states
        for agent_id, agent in agents.items():
            self.add_agent(agent_id, agent)
        
        # Clear obstacles
        self.obstacles.clear()
        
        logger.debug(f"Reset state for {len(agents)} agents")
    
    def add_agent(self, agent_id: int, agent: Any):
        """Add an agent to state management
        
        Args:
            agent_id: Agent ID
            agent: Agent object
        """
        # Initialize physics state
        state = AgentPhysicsState(
            position=agent.position.copy(),
            velocity=agent.velocity.copy(),
            acceleration=np.zeros(3),
            orientation=agent.orientation.copy(),
            angular_velocity=np.zeros(3),
            forces=np.zeros(3),
            torques=np.zeros(3),
            battery_soc=agent.battery_soc,
            power_consumption=0.0,
            motor_speeds=agent.motor_speeds.copy()
        )
        
        self.agent_states[agent_id] = state
        self._prev_positions[agent_id] = agent.position.copy()
        self._prev_velocities[agent_id] = agent.velocity.copy()
        self.constraint_violations[agent_id] = {}
    
    def remove_agent(self, agent_id: int):
        """Remove an agent from state management
        
        Args:
            agent_id: Agent ID
        """
        if agent_id in self.agent_states:
            del self.agent_states[agent_id]
        if agent_id in self._prev_positions:
            del self._prev_positions[agent_id]
        if agent_id in self._prev_velocities:
            del self._prev_velocities[agent_id]
        if agent_id in self.constraint_violations:
            del self.constraint_violations[agent_id]
    
    def apply_actions(self, actions: Dict[int, np.ndarray], dt: float):
        """Apply control actions to agents
        
        Args:
            actions: Dictionary of actions per agent
            dt: Time step
        """
        for agent_id, action in actions.items():
            if agent_id not in self.agent_states:
                continue
            
            state = self.agent_states[agent_id]
            
            # Simple velocity control for now
            # Action is [vx, vy, vz, yaw_rate]
            if len(action) >= 3:
                target_velocity = action[:3]
                
                # Apply velocity change with acceleration limits
                max_accel = 10.0  # m/s²
                velocity_change = target_velocity - state.velocity
                accel_magnitude = np.linalg.norm(velocity_change) / dt
                
                if accel_magnitude > max_accel:
                    velocity_change *= max_accel * dt / np.linalg.norm(velocity_change)
                
                state.velocity += velocity_change
                state.acceleration = velocity_change / dt
                
                # Yaw rate control
                if len(action) >= 4:
                    state.angular_velocity[2] = action[3]
            
            # Calculate forces (F = ma)
            # Assuming unit mass for simplicity (will be corrected in physics integration)
            state.forces = state.acceleration
    
    def step(self, dt: float):
        """Step the physics simulation
        
        Args:
            dt: Time step
        """
        # Update global time
        self.global_state.time += dt
        
        if not self.enable_physics:
            # Simple kinematic update
            for agent_id, state in self.agent_states.items():
                # Update position
                state.position += state.velocity * dt
                
                # Update orientation (simple yaw integration)
                yaw_change = state.angular_velocity[2] * dt
                # Simplified quaternion update for yaw
                c = np.cos(yaw_change / 2)
                s = np.sin(yaw_change / 2)
                yaw_quat = np.array([0, 0, s, c])
                state.orientation = self._quaternion_multiply(state.orientation, yaw_quat)
        else:
            # Full physics update
            self._physics_step(dt)
        
        # Update derivatives
        self._update_derivatives(dt)
        
        # Check constraints
        self._check_constraints()
        
        # Update energy
        self._update_energy(dt)
    
    def _physics_step(self, dt: float):
        """Perform physics simulation step
        
        Args:
            dt: Time step
        """
        for agent_id, state in self.agent_states.items():
            # Apply gravity
            gravity_force = np.array([0, 0, -9.81])  # Assuming unit mass
            total_force = state.forces + gravity_force
            
            # Apply drag
            drag_coefficient = 0.1
            drag_force = -drag_coefficient * state.velocity * np.linalg.norm(state.velocity)
            total_force += drag_force
            
            # Update velocity (F = ma, assuming m=1 for now)
            state.acceleration = total_force
            state.velocity += state.acceleration * dt
            
            # Update position
            state.position += state.velocity * dt
            
            # Update orientation
            # Convert angular velocity to quaternion rate
            omega = state.angular_velocity
            q = state.orientation
            q_dot = 0.5 * self._quaternion_multiply(
                np.array([omega[0], omega[1], omega[2], 0]), q
            )
            state.orientation += q_dot * dt
            state.orientation /= np.linalg.norm(state.orientation)  # Normalize
            
            # Boundary conditions
            self._apply_boundary_conditions(state)
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions
        
        Args:
            q1: First quaternion [x, y, z, w]
            q2: Second quaternion
            
        Returns:
            Result quaternion
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
    
    def _apply_boundary_conditions(self, state: AgentPhysicsState):
        """Apply boundary conditions to agent state
        
        Args:
            state: Agent physics state
        """
        # World boundaries
        for i in range(3):
            if state.position[i] < 0:
                state.position[i] = 0
                state.velocity[i] = abs(state.velocity[i]) * 0.5  # Bounce with damping
            elif state.position[i] > self.world_size[i]:
                state.position[i] = self.world_size[i]
                state.velocity[i] = -abs(state.velocity[i]) * 0.5
        
        # Ground collision
        if state.position[2] <= 0:
            state.position[2] = 0
            state.velocity[2] = 0
            state.collision = True
    
    def _update_derivatives(self, dt: float):
        """Update derivative quantities
        
        Args:
            dt: Time step
        """
        for agent_id, state in self.agent_states.items():
            # Calculate acceleration from velocity change
            if agent_id in self._prev_velocities:
                state.acceleration = (state.velocity - self._prev_velocities[agent_id]) / dt
            
            # Store current values
            self._prev_positions[agent_id] = state.position.copy()
            self._prev_velocities[agent_id] = state.velocity.copy()
    
    def _check_constraints(self):
        """Check physics constraints for all agents"""
        agent_ids = list(self.agent_states.keys())
        
        for i, agent_id in enumerate(agent_ids):
            state = self.agent_states[agent_id]
            violations = {}
            
            # Velocity constraint
            speed = np.linalg.norm(state.velocity)
            violations["velocity"] = speed > 25.0  # Max speed exceeded
            
            # Acceleration constraint
            accel_mag = np.linalg.norm(state.acceleration)
            violations["acceleration"] = accel_mag > 15.0  # Max acceleration exceeded
            
            # Energy constraint
            violations["energy"] = state.battery_soc <= 0.05  # Critical battery
            
            # Collision detection
            state.collision_agents.clear()
            state.min_separation = float('inf')
            
            for j, other_id in enumerate(agent_ids):
                if i != j:
                    other_state = self.agent_states[other_id]
                    separation = np.linalg.norm(state.position - other_state.position)
                    
                    if separation < state.min_separation:
                        state.min_separation = separation
                    
                    if separation < 1.0:  # Collision threshold
                        state.collision = True
                        state.collision_agents.append(other_id)
                        violations["collision"] = True
            
            # Altitude constraint
            violations["altitude"] = state.position[2] < 0 or state.position[2] > self.world_size[2]
            
            self.constraint_violations[agent_id] = violations
    
    def _update_energy(self, dt: float):
        """Update energy consumption
        
        Args:
            dt: Time step
        """
        for agent_id, state in self.agent_states.items():
            # Simple power model
            # Power = base + velocity_dependent + acceleration_dependent
            base_power = 50.0  # Watts
            velocity_power = 0.5 * np.linalg.norm(state.velocity) ** 2
            accel_power = 10.0 * np.linalg.norm(state.acceleration)
            
            state.power_consumption = base_power + velocity_power + accel_power
            
            # Update battery (assuming 50Wh battery)
            battery_capacity = 50.0  # Wh
            energy_consumed = state.power_consumption * dt / 3600.0  # Wh
            soc_decrease = energy_consumed / battery_capacity
            
            state.battery_soc = max(0.0, state.battery_soc - soc_decrease)
            
            # Update global energy tracking
            self.global_state.total_energy_consumed += energy_consumed
    
    def get_agent_state(self, agent_id: int) -> Dict[str, Any]:
        """Get state for a specific agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent state dictionary
        """
        if agent_id not in self.agent_states:
            raise ValueError(f"Agent {agent_id} not found")
        
        state = self.agent_states[agent_id]
        
        return {
            "position": state.position.copy(),
            "velocity": state.velocity.copy(),
            "acceleration": state.acceleration.copy(),
            "orientation": state.orientation.copy(),
            "angular_velocity": state.angular_velocity.copy(),
            "battery_soc": state.battery_soc,
            "power_consumption": state.power_consumption,
            "motor_speeds": state.motor_speeds.copy(),
            "collision": state.collision,
            "min_separation": state.min_separation
        }
    
    def get_positions(self) -> Dict[int, np.ndarray]:
        """Get all agent positions
        
        Returns:
            Dictionary of positions
        """
        return {
            agent_id: state.position.copy()
            for agent_id, state in self.agent_states.items()
        }
    
    def get_global_state(self) -> Dict[str, Any]:
        """Get global environment state
        
        Returns:
            Global state dictionary
        """
        return self.global_state.to_dict()
    
    def get_constraint_violations(self, agent_id: int) -> Dict[str, bool]:
        """Get constraint violations for an agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Dictionary of constraint violations
        """
        return self.constraint_violations.get(agent_id, {})
    
    def add_obstacle(self, obstacle: Dict[str, Any]):
        """Add an obstacle to the environment
        
        Args:
            obstacle: Obstacle specification
        """
        self.obstacles.append(obstacle)
    
    def get_obstacles(self) -> List[Dict[str, Any]]:
        """Get list of obstacles
        
        Returns:
            List of obstacles
        """
        return self.obstacles.copy()
    
    def set_wind(self, wind_velocity: np.ndarray):
        """Set wind velocity
        
        Args:
            wind_velocity: Wind velocity vector
        """
        self.global_state.wind_velocity = wind_velocity.copy()
    
    def set_weather(self, temperature: float, air_density: float):
        """Set weather conditions
        
        Args:
            temperature: Temperature in Celsius
            air_density: Air density in kg/m³
        """
        self.global_state.temperature = temperature
        self.global_state.air_density = air_density