"""
Drone Swarm Package Delivery Domain
Real-world implementation with Crazyflie quadcopters and Gazebo integration
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pybullet as p
from dataclasses import dataclass
import logging
from scipy.spatial.transform import Rotation

from .base_real_world import RealWorldDomain, HardwareConfig

@dataclass
class DroneSwarmConfig:
    """Configuration for drone swarm environment"""
    operation_area: Tuple[float, float, float] = (100.0, 100.0, 50.0)  # meters (x, y, z)
    num_drones: int = 5
    num_delivery_locations: int = 10
    num_packages: int = 15
    drone_mass: float = 0.5  # kg (Crazyflie with payload)
    max_payload: float = 0.1  # kg
    max_velocity: float = 2.0  # m/s
    max_acceleration: float = 2.0  # m/s^2
    max_angular_velocity: float = 2.0  # rad/s
    battery_capacity: float = 600.0  # seconds of flight
    min_separation: float = 2.0  # meters
    communication_range: float = 50.0  # meters
    wind_speed_max: float = 5.0  # m/s

class QuadcopterPhysics:
    """Physics model for quadcopter drones"""
    
    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        
        # Physical parameters
        self.mass = 0.5  # kg
        self.inertia = np.diag([0.01, 0.01, 0.02])  # kg*m^2
        self.arm_length = 0.1  # meters
        self.rotor_thrust_coefficient = 1e-6
        self.rotor_drag_coefficient = 1e-8
        
        # Aerodynamic parameters
        self.drag_coefficient = 0.1
        self.cross_section_area = 0.01  # m^2
        
        # Battery model
        self.battery_capacity = 600.0  # seconds
        self.battery_remaining = 600.0
        self.hover_power = 15.0  # W
        self.max_power = 30.0  # W
        
        # State
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([0, 0, 0, 1])  # quaternion
        self.angular_velocity = np.zeros(3)
        
    def compute_dynamics(self, 
                        rotor_speeds: np.ndarray,
                        wind_velocity: np.ndarray,
                        dt: float = 0.01) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Compute quadcopter dynamics with aerodynamics
        
        Args:
            rotor_speeds: 4D array of rotor angular velocities (rad/s)
            wind_velocity: 3D wind velocity vector (m/s)
            dt: time step
            
        Returns:
            (state_update, power_consumed)
        """
        # Compute thrust from each rotor
        thrusts = self.rotor_thrust_coefficient * rotor_speeds**2
        total_thrust = np.sum(thrusts)
        
        # Compute torques
        # Assuming rotors arranged in X configuration
        torque_x = self.arm_length * (thrusts[1] + thrusts[3] - thrusts[0] - thrusts[2])
        torque_y = self.arm_length * (thrusts[0] + thrusts[1] - thrusts[2] - thrusts[3])
        torque_z = self.rotor_drag_coefficient * (
            rotor_speeds[0]**2 - rotor_speeds[1]**2 + 
            rotor_speeds[2]**2 - rotor_speeds[3]**2
        )
        torques = np.array([torque_x, torque_y, torque_z])
        
        # Convert quaternion to rotation matrix
        rotation = Rotation.from_quat(self.orientation)
        R = rotation.as_matrix()
        
        # Compute forces in body frame
        thrust_body = np.array([0, 0, total_thrust])
        thrust_world = R @ thrust_body
        
        # Gravity
        gravity = np.array([0, 0, -9.81 * self.mass])
        
        # Aerodynamic drag (including wind)
        relative_velocity = self.velocity - wind_velocity
        drag_magnitude = 0.5 * 1.225 * self.drag_coefficient * \
                        self.cross_section_area * np.linalg.norm(relative_velocity)**2
        
        if np.linalg.norm(relative_velocity) > 0:
            drag_force = -drag_magnitude * relative_velocity / np.linalg.norm(relative_velocity)
        else:
            drag_force = np.zeros(3)
        
        # Total force
        total_force = thrust_world + gravity + drag_force
        
        # Linear dynamics
        acceleration = total_force / self.mass
        new_velocity = self.velocity + acceleration * dt
        new_position = self.position + self.velocity * dt + 0.5 * acceleration * dt**2
        
        # Angular dynamics
        angular_acceleration = np.linalg.inv(self.inertia) @ (
            torques - np.cross(self.angular_velocity, self.inertia @ self.angular_velocity)
        )
        new_angular_velocity = self.angular_velocity + angular_acceleration * dt
        
        # Update orientation
        rotation_update = Rotation.from_rotvec(self.angular_velocity * dt)
        new_rotation = rotation_update * rotation
        new_orientation = new_rotation.as_quat()
        
        # Power consumption
        rotor_power = np.sum(self.rotor_thrust_coefficient * rotor_speeds**3) / 1000  # Convert to watts
        power = np.clip(rotor_power, self.hover_power, self.max_power)
        
        # Update state
        state_update = {
            'position': new_position,
            'velocity': new_velocity,
            'orientation': new_orientation,
            'angular_velocity': new_angular_velocity,
            'acceleration': acceleration,
            'power': power
        }
        
        return state_update, power * dt

class DroneSwarmDelivery(RealWorldDomain):
    """
    Drone swarm package delivery with real Crazyflie quadcopters
    
    Task: Coordinated package delivery to multiple locations
    Physics: Aerodynamics, wind effects, payload limits, energy conservation
    """
    
    def __init__(self,
                 sim_mode: bool = True,
                 hardware_interface: Optional[Any] = None,
                 swarm_config: Optional[DroneSwarmConfig] = None):
        
        self.swarm_config = swarm_config or DroneSwarmConfig()
        
        # Hardware configuration for Crazyflie
        hardware_config = HardwareConfig(
            robot_type="Crazyflie",
            communication_protocol="CrazyradioPA",
            control_frequency=100.0,  # 100Hz for quadcopters
            safety_timeout=0.1,
            emergency_stop_enabled=True,
            data_logging_enabled=True
        )
        
        super().__init__(sim_mode, hardware_interface, hardware_config)
        
        # Swarm-specific attributes
        self.drones = {}
        self.packages = {}
        self.delivery_locations = {}
        self.wind_field = None
        
        # Task tracking
        self.delivered_packages = 0
        self.failed_deliveries = 0
        self.active_deliveries = {}
        
        # Formation control
        self.formation_graph = np.ones((self.swarm_config.num_drones,
                                      self.swarm_config.num_drones))
        self.desired_formation = None
    
    def _initialize_domain(self):
        """Initialize drone swarm domain"""
        if self.sim_mode:
            self._initialize_simulation()
        else:
            self._initialize_hardware()
        
        self._generate_delivery_locations()
        self._initialize_drones()
        self._initialize_packages()
        self._initialize_wind_field()
    
    def _initialize_simulation(self):
        """Initialize PyBullet simulation for drones"""
        self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / self.config.control_frequency)
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
        # Load buildings/obstacles
        self._create_urban_environment()
    
    def _initialize_hardware(self):
        """Initialize connection to real Crazyflie drones"""
        if self.hardware_interface is None:
            raise ValueError("Hardware interface required for real-world mode")
        
        # Initialize Crazyflie connections
        self.drone_connections = {}
        for i in range(self.swarm_config.num_drones):
            drone_uri = f"radio://0/80/2M/E7E7E7E7{i:02d}"
            self.drone_connections[i] = self.hardware_interface.connect_crazyflie(
                uri=drone_uri,
                drone_id=i
            )
    
    def _initialize_drones(self):
        """Initialize drone physics models and positions"""
        for i in range(self.swarm_config.num_drones):
            # Create physics model
            physics_model = QuadcopterPhysics(i)
            
            # Set initial position (distributed launch points)
            initial_pos = self._get_initial_drone_position(i)
            physics_model.position = initial_pos
            
            self.drones[i] = {
                'physics': physics_model,
                'position': initial_pos,
                'velocity': np.zeros(3),
                'orientation': np.array([0, 0, 0, 1]),
                'angular_velocity': np.zeros(3),
                'battery': self.swarm_config.battery_capacity,
                'carrying_package': None,
                'target_location': None,
                'rotor_speeds': np.ones(4) * 5000  # Initial hover RPM
            }
            
            if self.sim_mode:
                # Create visual representation
                drone_id = p.loadURDF(
                    "crazyflie2.urdf",
                    initial_pos,
                    self.drones[i]['orientation']
                )
                self.drones[i]['sim_id'] = drone_id
    
    def get_physics_constraints(self) -> Dict[str, Any]:
        """Return drone swarm physics constraints"""
        return {
            'max_velocity': self.swarm_config.max_velocity,
            'max_acceleration': self.swarm_config.max_acceleration,
            'max_angular_velocity': self.swarm_config.max_angular_velocity,
            'max_payload': self.swarm_config.max_payload,
            'min_separation': self.swarm_config.min_separation,
            'battery_constraints': {
                'capacity': self.swarm_config.battery_capacity,
                'critical_level': 60.0,  # 1 minute
                'return_home_level': 120.0  # 2 minutes
            },
            'aerodynamics': {
                'drag_coefficient': 0.1,
                'max_wind_speed': self.swarm_config.wind_speed_max
            },
            'communication_range': self.swarm_config.communication_range
        }
    
    def validate_action_physics(self, action: np.ndarray, agent_id: int) -> Tuple[bool, Optional[str]]:
        """
        Validate drone action against physics constraints
        
        Action format: [thrust, roll_rate, pitch_rate, yaw_rate]
        """
        if len(action) != 4:
            return False, "Invalid action dimension"
        
        thrust, roll_rate, pitch_rate, yaw_rate = action
        
        # Check thrust limits (normalized 0-1)
        if thrust < 0 or thrust > 1:
            return False, f"Thrust {thrust} out of bounds [0, 1]"
        
        # Check angular rate limits
        angular_rates = np.array([roll_rate, pitch_rate, yaw_rate])
        if np.any(np.abs(angular_rates) > self.swarm_config.max_angular_velocity):
            return False, f"Angular rates exceed maximum {self.swarm_config.max_angular_velocity}"
        
        # Check battery
        drone = self.drones[agent_id]
        if drone['battery'] < 60.0:  # Critical battery
            if thrust > 0.5:  # Only allow low thrust
                return False, "Battery critical - reduce thrust"
        
        # Check payload limits
        if drone['carrying_package'] is not None:
            package = self.packages[drone['carrying_package']]
            if package['weight'] > self.swarm_config.max_payload:
                return False, f"Package weight {package['weight']} exceeds max payload"
            
            # Reduce agility when carrying package
            if np.any(np.abs(angular_rates) > self.swarm_config.max_angular_velocity * 0.7):
                return False, "Reduce angular rates when carrying package"
        
        # Check separation from other drones
        future_pos = self._predict_drone_position(agent_id, action, dt=0.5)
        for other_id, other_drone in self.drones.items():
            if other_id == agent_id:
                continue
            
            separation = np.linalg.norm(future_pos - other_drone['position'])
            if separation < self.swarm_config.min_separation:
                return False, f"Collision risk with drone {other_id} (separation: {separation:.1f}m)"
        
        # Check altitude limits
        if future_pos[2] < 1.0:  # Minimum altitude
            return False, "Altitude too low"
        elif future_pos[2] > self.swarm_config.operation_area[2]:
            return False, f"Altitude exceeds maximum {self.swarm_config.operation_area[2]}m"
        
        # Check operation area boundaries
        if (future_pos[0] < 0 or future_pos[0] > self.swarm_config.operation_area[0] or
            future_pos[1] < 0 or future_pos[1] > self.swarm_config.operation_area[1]):
            return False, "Position outside operation area"
        
        return True, None
    
    def _predict_drone_position(self, agent_id: int, action: np.ndarray, dt: float) -> np.ndarray:
        """Predict future drone position given action"""
        drone = self.drones[agent_id]
        
        # Simple prediction based on current velocity and commanded thrust
        thrust_world = action[0] * 10.0 * np.array([0, 0, 1])  # Simplified
        acceleration = thrust_world / drone['physics'].mass - np.array([0, 0, 9.81])
        
        future_velocity = drone['velocity'] + acceleration * dt
        future_position = drone['position'] + drone['velocity'] * dt + 0.5 * acceleration * dt**2
        
        return future_position
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Execute one step in the drone swarm environment
        
        Returns:
            (observations, rewards, dones, info)
        """
        # Get current wind conditions
        current_wind = self._get_wind_at_positions()
        
        # Validate and execute actions
        safe_actions = {}
        for agent_id, action in actions.items():
            is_valid, violation = self.validate_action_physics(action, agent_id)
            if is_valid:
                safe_actions[agent_id] = action
            else:
                # Use safe hover action
                safe_actions[agent_id] = self._get_safe_action(
                    self._get_drone_state(agent_id)
                )
                self.metrics.safety_violations += 1
                self.logger.warning(f"Drone {agent_id}: {violation}")
        
        # Convert actions to rotor speeds and execute
        for agent_id, action in safe_actions.items():
            rotor_speeds = self._action_to_rotor_speeds(action)
            self._execute_drone_action(agent_id, rotor_speeds, current_wind[agent_id])
        
        # Update physics simulation
        if self.sim_mode:
            p.stepSimulation()
        
        # Check collisions and separations
        self._check_drone_collisions()
        
        # Update package pickups and deliveries
        self._update_package_status()
        
        # Update formation
        self._update_formation_graph()
        
        # Get observations
        observations = self._get_observations()
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Check termination
        dones = self._check_termination()
        
        # Additional info
        info = {
            'delivered_packages': self.delivered_packages,
            'failed_deliveries': self.failed_deliveries,
            'battery_levels': {i: d['battery'] for i, d in self.drones.items()},
            'wind_speeds': {i: np.linalg.norm(w) for i, w in current_wind.items()},
            'formation_quality': self._calculate_formation_quality()
        }
        
        return observations, rewards, dones, info
    
    def _action_to_rotor_speeds(self, action: np.ndarray) -> np.ndarray:
        """Convert high-level action to rotor speeds"""
        thrust, roll_rate, pitch_rate, yaw_rate = action
        
        # Base thrust for hover (simplified)
        hover_speed = 5000  # RPM
        thrust_component = thrust * 2000  # Additional RPM for thrust
        
        # Mixing matrix for X configuration
        # [Front-Right, Front-Left, Rear-Left, Rear-Right]
        rotor_speeds = np.array([
            hover_speed + thrust_component + roll_rate * 500 + pitch_rate * 500 - yaw_rate * 200,
            hover_speed + thrust_component - roll_rate * 500 + pitch_rate * 500 + yaw_rate * 200,
            hover_speed + thrust_component - roll_rate * 500 - pitch_rate * 500 - yaw_rate * 200,
            hover_speed + thrust_component + roll_rate * 500 - pitch_rate * 500 + yaw_rate * 200
        ])
        
        # Clip to physical limits
        return np.clip(rotor_speeds, 0, 10000)
    
    def _execute_drone_action(self, agent_id: int, rotor_speeds: np.ndarray, wind: np.ndarray):
        """Execute drone action with physics simulation"""
        drone = self.drones[agent_id]
        physics = drone['physics']
        
        # Update physics state
        physics.position = drone['position']
        physics.velocity = drone['velocity']
        physics.orientation = drone['orientation']
        physics.angular_velocity = drone['angular_velocity']
        
        # Compute dynamics
        state_update, power_consumed = physics.compute_dynamics(
            rotor_speeds, wind, dt=0.01
        )
        
        # Update drone state
        drone['position'] = state_update['position']
        drone['velocity'] = state_update['velocity']
        drone['orientation'] = state_update['orientation']
        drone['angular_velocity'] = state_update['angular_velocity']
        drone['rotor_speeds'] = rotor_speeds
        drone['battery'] -= power_consumed
        
        # Send to hardware if in real mode
        if not self.sim_mode:
            self._send_hardware_command(np.concatenate([
                [state_update['power']/self.swarm_config.max_power],
                state_update['angular_velocity']
            ]))
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all drones"""
        observations = {}
        
        for agent_id in range(self.swarm_config.num_drones):
            drone = self.drones[agent_id]
            
            # Own state (normalized)
            own_state = np.array([
                drone['position'][0] / self.swarm_config.operation_area[0],
                drone['position'][1] / self.swarm_config.operation_area[1],
                drone['position'][2] / self.swarm_config.operation_area[2],
                drone['velocity'][0] / self.swarm_config.max_velocity,
                drone['velocity'][1] / self.swarm_config.max_velocity,
                drone['velocity'][2] / self.swarm_config.max_velocity,
                drone['orientation'][0],  # Quaternion components
                drone['orientation'][1],
                drone['orientation'][2],
                drone['orientation'][3],
                drone['battery'] / self.swarm_config.battery_capacity,
                float(drone['carrying_package'] is not None)
            ])
            
            # Nearby drones (based on communication range)
            nearby_drones = []
            for other_id in range(self.swarm_config.num_drones):
                if other_id != agent_id:
                    distance = np.linalg.norm(
                        self.drones[other_id]['position'] - drone['position']
                    )
                    if distance < self.swarm_config.communication_range:
                        relative_pos = self.drones[other_id]['position'] - drone['position']
                        nearby_drones.extend([
                            relative_pos[0] / self.swarm_config.communication_range,
                            relative_pos[1] / self.swarm_config.communication_range,
                            relative_pos[2] / self.swarm_config.communication_range,
                            self.drones[other_id]['velocity'][0] / self.swarm_config.max_velocity,
                            self.drones[other_id]['velocity'][1] / self.swarm_config.max_velocity,
                            self.drones[other_id]['velocity'][2] / self.swarm_config.max_velocity
                        ])
            
            # Pad if needed
            max_nearby = 4
            while len(nearby_drones) < max_nearby * 6:
                nearby_drones.extend([0, 0, 0, 0, 0, 0])
            
            # Target information
            if drone['target_location'] is not None:
                target_relative = drone['target_location'] - drone['position']
                target_info = target_relative / np.linalg.norm(target_relative + 1e-6)
                target_distance = np.linalg.norm(target_relative) / 100.0  # Normalize
            else:
                target_info = np.zeros(3)
                target_distance = 0.0
            
            # Wind information at current position
            wind = self._get_wind_at_position(drone['position'])
            wind_info = wind / self.swarm_config.wind_speed_max
            
            observations[agent_id] = np.concatenate([
                own_state,
                nearby_drones[:max_nearby * 6],
                target_info,
                [target_distance],
                wind_info
            ])
        
        return observations
    
    def _calculate_rewards(self) -> Dict[int, float]:
        """Calculate rewards for each drone"""
        rewards = {}
        
        for agent_id in range(self.swarm_config.num_drones):
            drone = self.drones[agent_id]
            reward = 0.0
            
            # Package delivery rewards
            if drone['carrying_package'] is not None:
                package = self.packages[drone['carrying_package']]
                delivery_loc = self.delivery_locations[package['destination']]
                
                # Distance to delivery location
                dist_to_delivery = np.linalg.norm(
                    drone['position'] - delivery_loc['position']
                )
                
                # Progress reward
                reward -= 0.01 * dist_to_delivery
                
                # Delivery completion
                if dist_to_delivery < 2.0 and drone['position'][2] < 3.0:
                    reward += 50.0 * package['priority']  # Priority-weighted reward
                    self.delivered_packages += 1
                    drone['carrying_package'] = None
                    package['delivered'] = True
                
                # Time penalty for urgent packages
                if package['priority'] > 2:
                    reward -= 0.1
            
            else:
                # Find nearest undelivered package
                nearest_package = self._find_nearest_package(agent_id)
                if nearest_package is not None:
                    package = self.packages[nearest_package]
                    dist_to_package = np.linalg.norm(
                        drone['position'] - package['position']
                    )
                    reward -= 0.005 * dist_to_package
                    
                    # Pickup reward
                    if dist_to_package < 1.0 and drone['position'][2] < 5.0:
                        if package['weight'] <= self.swarm_config.max_payload:
                            reward += 20.0
                            drone['carrying_package'] = nearest_package
                            package['picked_up'] = True
                            drone['target_location'] = self.delivery_locations[
                                package['destination']
                            ]['position']
            
            # Energy efficiency
            power_usage = drone['physics'].hover_power  # Simplified
            reward -= 0.0001 * power_usage
            
            # Formation keeping bonus
            formation_error = self._calculate_formation_error(agent_id)
            if formation_error < 5.0:
                reward += 0.1 * (5.0 - formation_error)
            
            # Collision/separation penalty
            for other_id, other_drone in self.drones.items():
                if other_id != agent_id:
                    separation = np.linalg.norm(
                        drone['position'] - other_drone['position']
                    )
                    if separation < self.swarm_config.min_separation:
                        reward -= 10.0 * (self.swarm_config.min_separation - separation)
            
            # Battery management
            if drone['battery'] < 120.0:  # Low battery
                # Encourage return to base
                dist_to_base = np.linalg.norm(drone['position'] - self._get_base_position())
                reward -= 0.1 * dist_to_base
            
            # Altitude penalty (encourage efficient flight paths)
            if drone['position'][2] > 30.0:
                reward -= 0.01 * (drone['position'][2] - 30.0)
            
            rewards[agent_id] = reward
        
        return rewards
    
    def _check_drone_collisions(self):
        """Check for drone collisions"""
        collision_threshold = 0.5  # meters
        
        for i in range(self.swarm_config.num_drones):
            for j in range(i + 1, self.swarm_config.num_drones):
                distance = np.linalg.norm(
                    self.drones[i]['position'] - self.drones[j]['position']
                )
                if distance < collision_threshold:
                    self.logger.error(f"COLLISION between drones {i} and {j}!")
                    self.metrics.collision_count += 1
                    
                    # Apply collision physics (simplified)
                    self.drones[i]['velocity'] *= -0.5
                    self.drones[j]['velocity'] *= -0.5
    
    def _update_formation_graph(self):
        """Update formation based on drone positions"""
        for i in range(self.swarm_config.num_drones):
            for j in range(self.swarm_config.num_drones):
                if i == j:
                    self.formation_graph[i, j] = 1.0
                else:
                    distance = np.linalg.norm(
                        self.drones[i]['position'] - self.drones[j]['position']
                    )
                    if distance < self.swarm_config.communication_range:
                        self.formation_graph[i, j] = 1.0 - (
                            distance / self.swarm_config.communication_range
                        )
                    else:
                        self.formation_graph[i, j] = 0.0
    
    def _calculate_formation_quality(self) -> float:
        """Calculate quality of current formation"""
        if self.desired_formation is None:
            return 1.0
        
        total_error = 0.0
        for i in range(self.swarm_config.num_drones):
            error = self._calculate_formation_error(i)
            total_error += error
        
        avg_error = total_error / self.swarm_config.num_drones
        quality = np.exp(-avg_error / 10.0)  # Exponential decay
        
        return quality
    
    def _calculate_formation_error(self, agent_id: int) -> float:
        """Calculate formation error for a specific drone"""
        if self.desired_formation is None:
            return 0.0
        
        actual_pos = self.drones[agent_id]['position']
        desired_pos = self.desired_formation[agent_id]
        
        return np.linalg.norm(actual_pos - desired_pos)
    
    def _check_termination(self) -> Dict[int, bool]:
        """Check termination conditions"""
        dones = {}
        
        # All packages delivered
        all_delivered = all(
            p['delivered'] for p in self.packages.values()
        )
        
        # Any drone crashed or out of battery
        any_crashed = any(
            d['position'][2] < 0.5 or d['battery'] <= 0
            for d in self.drones.values()
        )
        
        for agent_id in range(self.swarm_config.num_drones):
            dones[agent_id] = all_delivered or any_crashed
        
        dones['__all__'] = all_delivered or any_crashed
        
        return dones
    
    # Real-world specific methods
    def _get_current_state(self) -> np.ndarray:
        """Get current state from simulation or hardware"""
        if self.sim_mode:
            states = []
            for i in range(self.swarm_config.num_drones):
                drone = self.drones[i]
                states.extend(drone['position'])
                states.extend(drone['velocity'])
                states.extend(drone['orientation'])
                states.append(drone['battery'])
            return np.array(states)
        else:
            # Get from Crazyflie hardware
            states = []
            for i in range(self.swarm_config.num_drones):
                state = self.hardware_interface.get_drone_state(i)
                states.extend(state)
            return np.array(states)
    
    def _compute_action(self, state: np.ndarray) -> np.ndarray:
        """Compute action using trained policy"""
        # Simplified hover action
        return np.array([0.5, 0.0, 0.0, 0.0])  # 50% thrust, no rotation
    
    def _get_safe_action(self, state: np.ndarray) -> np.ndarray:
        """Get safe hover action"""
        return np.array([0.45, 0.0, 0.0, 0.0])  # Gentle hover
    
    def _send_hardware_command(self, action: np.ndarray):
        """Send command to Crazyflie"""
        if not self.sim_mode and self.hardware_interface:
            # Convert to Crazyflie command format
            thrust = int(action[0] * 65535)  # 16-bit thrust
            roll_rate = action[1] * 180 / np.pi  # Convert to degrees/s
            pitch_rate = action[2] * 180 / np.pi
            yaw_rate = action[3] * 180 / np.pi
            
            command = {
                'thrust': thrust,
                'roll': roll_rate,
                'pitch': pitch_rate,
                'yaw': yaw_rate
            }
            
            self.hardware_interface.send_control_command(command)
    
    # Safety check implementations
    def _check_communication(self) -> bool:
        """Check Crazyflie radio communication"""
        if self.sim_mode:
            return True
        
        for i in range(self.swarm_config.num_drones):
            if not self.drone_connections[i].is_connected():
                return False
        return True
    
    def _check_sensors(self) -> bool:
        """Check drone sensors (IMU, barometer, flow deck)"""
        if self.sim_mode:
            return True
        
        for i in range(self.swarm_config.num_drones):
            sensor_status = self.hardware_interface.check_sensors(i)
            if not all(sensor_status.values()):
                return False
        return True
    
    def _check_actuators(self) -> bool:
        """Check drone motors"""
        if self.sim_mode:
            return True
        
        for i in range(self.swarm_config.num_drones):
            motor_status = self.hardware_interface.check_motors(i)
            if not all(motor_status):
                return False
        return True
    
    def _check_emergency_stop(self) -> bool:
        """Check emergency stop system"""
        return not self.emergency_stop
    
    def _check_workspace(self) -> bool:
        """Check if airspace is clear"""
        if self.sim_mode:
            return True
        
        # Check for no-fly zones or obstacles
        return self.hardware_interface.check_airspace_clear()
    
    def _detect_collision(self) -> bool:
        """Detect if collision occurred"""
        collision_threshold = 0.5  # meters
        
        for i in range(self.swarm_config.num_drones):
            # Check ground collision
            if self.drones[i]['position'][2] < 0.5:
                return True
            
            # Check drone-to-drone collision
            for j in range(i + 1, self.swarm_config.num_drones):
                distance = np.linalg.norm(
                    self.drones[i]['position'] - self.drones[j]['position']
                )
                if distance < collision_threshold:
                    return True
        
        return False
    
    def _check_communication_health(self) -> bool:
        """Check if communication is healthy"""
        if self.sim_mode:
            return True
        
        for i in range(self.swarm_config.num_drones):
            rssi = self.hardware_interface.get_radio_strength(i)
            if rssi < -80:  # dBm threshold
                return False
        return True
    
    def _detect_hardware_failure(self) -> bool:
        """Detect hardware failures"""
        if self.sim_mode:
            return False
        
        for i in range(self.swarm_config.num_drones):
            # Check for motor failures
            motor_health = self.hardware_interface.get_motor_health(i)
            if any(h < 0.8 for h in motor_health):
                return True
            
            # Check IMU health
            imu_variance = self.hardware_interface.get_imu_variance(i)
            if imu_variance > 0.1:  # High noise indicates problem
                return True
        
        return False
    
    # Helper methods
    def _generate_delivery_locations(self):
        """Generate delivery locations in urban environment"""
        for i in range(self.swarm_config.num_delivery_locations):
            # Random positions avoiding obstacles
            while True:
                x = np.random.uniform(10, self.swarm_config.operation_area[0] - 10)
                y = np.random.uniform(10, self.swarm_config.operation_area[1] - 10)
                z = 0.0  # Ground level
                
                # Check if location is accessible
                if self._is_location_accessible(np.array([x, y, z])):
                    self.delivery_locations[i] = {
                        'position': np.array([x, y, z]),
                        'name': f"Location_{i}",
                        'demand': np.random.randint(1, 5)
                    }
                    break
    
    def _initialize_packages(self):
        """Initialize packages to be delivered"""
        for i in range(self.swarm_config.num_packages):
            # Random pickup location (warehouse or distribution center)
            pickup_x = np.random.uniform(40, 60)  # Center area
            pickup_y = np.random.uniform(40, 60)
            
            self.packages[i] = {
                'position': np.array([pickup_x, pickup_y, 0]),
                'destination': np.random.randint(0, self.swarm_config.num_delivery_locations),
                'weight': np.random.uniform(0.02, self.swarm_config.max_payload),
                'priority': np.random.randint(1, 4),  # 1=low, 3=urgent
                'picked_up': False,
                'delivered': False
            }
    
    def _initialize_wind_field(self):
        """Initialize wind field for the environment"""
        # Simple wind model with spatial variation
        self.wind_base_direction = np.random.uniform(0, 2 * np.pi)
        self.wind_base_speed = np.random.uniform(0, self.swarm_config.wind_speed_max * 0.5)
        
        # Turbulence parameters
        self.wind_turbulence_scale = 20.0  # meters
        self.wind_turbulence_strength = 2.0  # m/s
    
    def _get_wind_at_position(self, position: np.ndarray) -> np.ndarray:
        """Get wind velocity at specific position"""
        # Base wind
        wind_x = self.wind_base_speed * np.cos(self.wind_base_direction)
        wind_y = self.wind_base_speed * np.sin(self.wind_base_direction)
        
        # Add turbulence based on position
        turb_x = self.wind_turbulence_strength * np.sin(
            position[0] / self.wind_turbulence_scale
        )
        turb_y = self.wind_turbulence_strength * np.sin(
            position[1] / self.wind_turbulence_scale
        )
        
        # Altitude effect (wind increases with height)
        altitude_factor = 1.0 + position[2] / 100.0
        
        return np.array([
            (wind_x + turb_x) * altitude_factor,
            (wind_y + turb_y) * altitude_factor,
            0.0  # No vertical wind component for simplicity
        ])
    
    def _get_wind_at_positions(self) -> Dict[int, np.ndarray]:
        """Get wind at all drone positions"""
        winds = {}
        for i, drone in self.drones.items():
            winds[i] = self._get_wind_at_position(drone['position'])
        return winds
    
    def _get_initial_drone_position(self, drone_id: int) -> np.ndarray:
        """Get initial position for drone"""
        # Launch from base station
        base_x, base_y = 50, 50  # Center of operation area
        
        # Circular formation around base
        angle = (drone_id / self.swarm_config.num_drones) * 2 * np.pi
        radius = 5.0
        
        x = base_x + radius * np.cos(angle)
        y = base_y + radius * np.sin(angle)
        z = 10.0  # Initial altitude
        
        return np.array([x, y, z])
    
    def _get_base_position(self) -> np.ndarray:
        """Get base station position"""
        return np.array([50, 50, 0])
    
    def _create_urban_environment(self):
        """Create buildings and obstacles for urban delivery"""
        if not self.sim_mode:
            return
        
        # Create buildings
        num_buildings = 15
        for i in range(num_buildings):
            # Random building parameters
            width = np.random.uniform(5, 15)
            length = np.random.uniform(5, 15)
            height = np.random.uniform(10, 30)
            
            # Random position avoiding drone paths
            while True:
                x = np.random.uniform(10, self.swarm_config.operation_area[0] - 10)
                y = np.random.uniform(10, self.swarm_config.operation_area[1] - 10)
                
                # Check if not in center area (base station)
                if np.linalg.norm([x - 50, y - 50]) > 20:
                    building_visual = p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=[width/2, length/2, height/2],
                        rgbaColor=[0.6, 0.6, 0.6, 1]
                    )
                    building_collision = p.createCollisionShape(
                        p.GEOM_BOX,
                        halfExtents=[width/2, length/2, height/2]
                    )
                    p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=building_visual,
                        baseCollisionShapeIndex=building_collision,
                        basePosition=[x, y, height/2]
                    )
                    break
    
    def _is_location_accessible(self, position: np.ndarray) -> bool:
        """Check if location is accessible (not inside building)"""
        # Simplified check - would use collision detection in practice
        # Avoid center area (base station)
        if np.linalg.norm(position[:2] - np.array([50, 50])) < 15:
            return False
        
        return True
    
    def _find_nearest_package(self, agent_id: int) -> Optional[int]:
        """Find nearest undelivered package"""
        drone_pos = self.drones[agent_id]['position']
        nearest_id = None
        min_distance = float('inf')
        
        for package_id, package in self.packages.items():
            if not package['picked_up'] and not package['delivered']:
                distance = np.linalg.norm(drone_pos - package['position'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_id = package_id
        
        return nearest_id
    
    def _update_package_status(self):
        """Update package pickup and delivery status"""
        # This is handled in the reward calculation for simplicity
        pass
    
    def _get_drone_state(self, agent_id: int) -> np.ndarray:
        """Get state for specific drone"""
        drone = self.drones[agent_id]
        return np.concatenate([
            drone['position'],
            drone['velocity'],
            drone['orientation'],
            [drone['battery']]
        ])