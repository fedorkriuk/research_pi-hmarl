"""
Multi-Robot Warehouse Coordination Domain
Real-world implementation with TurtleBot3 robots and ROS2 integration
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pybullet as p
from dataclasses import dataclass
import logging

from .base_real_world import RealWorldDomain, HardwareConfig

@dataclass
class WarehouseConfig:
    """Configuration for warehouse environment"""
    warehouse_size: Tuple[float, float] = (20.0, 20.0)  # meters
    num_robots: int = 6
    num_shelves: int = 12
    num_items: int = 20
    robot_radius: float = 0.2  # meters
    max_velocity: float = 0.5  # m/s
    max_angular_velocity: float = 1.0  # rad/s
    battery_capacity: float = 3600.0  # seconds of operation
    collision_threshold: float = 0.4  # meters
    communication_range: float = 10.0  # meters

class TurtleBot3Physics:
    """Physics model for TurtleBot3 robots"""
    
    def __init__(self, robot_id: int):
        self.robot_id = robot_id
        self.mass = 2.8  # kg (with payload capacity)
        self.radius = 0.2  # meters
        self.max_force = 10.0  # N
        self.max_torque = 2.0  # Nm
        self.friction_coefficient = 0.1
        
        # Battery model
        self.battery_capacity = 3600.0  # seconds
        self.battery_remaining = 3600.0
        self.base_power_consumption = 10.0  # W
        self.motion_power_factor = 5.0  # W per m/s
        
    def compute_dynamics(self, current_vel: np.ndarray, 
                        desired_vel: np.ndarray, 
                        dt: float = 0.1) -> Tuple[np.ndarray, float]:
        """
        Compute robot dynamics with physics constraints
        
        Returns:
            (actual_velocity, power_consumed)
        """
        # Acceleration limits based on max force
        max_accel = self.max_force / self.mass
        desired_accel = (desired_vel - current_vel) / dt
        
        # Clip acceleration
        accel_magnitude = np.linalg.norm(desired_accel)
        if accel_magnitude > max_accel:
            desired_accel = desired_accel / accel_magnitude * max_accel
        
        # Apply friction
        friction_decel = self.friction_coefficient * 9.81  # m/s^2
        if np.linalg.norm(current_vel) > 0:
            friction_force = -friction_decel * current_vel / np.linalg.norm(current_vel)
            desired_accel += friction_force
        
        # Compute new velocity
        new_vel = current_vel + desired_accel * dt
        
        # Compute power consumption
        speed = np.linalg.norm(new_vel)
        power = self.base_power_consumption + self.motion_power_factor * speed
        
        return new_vel, power * dt

class MultiRobotWarehouse(RealWorldDomain):
    """
    Multi-robot warehouse coordination with real TurtleBot3 robots
    
    Task: Collaborative item retrieval and sorting
    Physics: Collision avoidance, momentum conservation, battery constraints
    """
    
    def __init__(self, 
                 sim_mode: bool = True,
                 hardware_interface: Optional[Any] = None,
                 warehouse_config: Optional[WarehouseConfig] = None):
        
        self.warehouse_config = warehouse_config or WarehouseConfig()
        
        # Hardware configuration for TurtleBot3
        hardware_config = HardwareConfig(
            robot_type="TurtleBot3",
            communication_protocol="ROS2",
            control_frequency=10.0,
            safety_timeout=0.5,
            emergency_stop_enabled=True,
            data_logging_enabled=True
        )
        
        super().__init__(sim_mode, hardware_interface, hardware_config)
        
        # Warehouse-specific attributes
        self.robots = {}
        self.items = {}
        self.shelves = {}
        self.collision_pairs = []
        
        # Task tracking
        self.completed_deliveries = 0
        self.active_tasks = {}
        self.task_queue = []
        
        # Communication graph
        self.communication_graph = np.ones((self.warehouse_config.num_robots, 
                                          self.warehouse_config.num_robots))
    
    def _initialize_domain(self):
        """Initialize warehouse domain"""
        if self.sim_mode:
            self._initialize_simulation()
        else:
            self._initialize_hardware()
        
        self._generate_warehouse_layout()
        self._initialize_robots()
        self._initialize_task_queue()
    
    def _initialize_simulation(self):
        """Initialize PyBullet simulation"""
        self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / self.config.control_frequency)
        
        # Load warehouse floor
        p.loadURDF("plane.urdf")
        
        # Load walls
        self._create_warehouse_walls()
    
    def _initialize_hardware(self):
        """Initialize connection to real TurtleBot3 robots"""
        if self.hardware_interface is None:
            raise ValueError("Hardware interface required for real-world mode")
        
        # Initialize ROS2 nodes for each robot
        self.robot_nodes = {}
        for i in range(self.warehouse_config.num_robots):
            robot_name = f"turtlebot_{i}"
            self.robot_nodes[robot_name] = self.hardware_interface.create_robot_node(
                robot_name=robot_name,
                namespace=f"/warehouse/robot_{i}"
            )
    
    def _initialize_robots(self):
        """Initialize robot physics models and positions"""
        for i in range(self.warehouse_config.num_robots):
            # Create physics model
            self.robots[i] = {
                'physics': TurtleBot3Physics(i),
                'position': self._get_initial_robot_position(i),
                'velocity': np.zeros(2),
                'orientation': 0.0,
                'battery': 3600.0,
                'carrying_item': None,
                'task': None
            }
            
            if self.sim_mode:
                # Create visual representation in simulation
                robot_id = p.loadURDF(
                    "turtlebot3_burger.urdf",
                    self.robots[i]['position'] + [0.1],
                    p.getQuaternionFromEuler([0, 0, self.robots[i]['orientation']])
                )
                self.robots[i]['sim_id'] = robot_id
    
    def get_physics_constraints(self) -> Dict[str, Any]:
        """Return warehouse-specific physics constraints"""
        return {
            'max_velocity': self.warehouse_config.max_velocity,
            'max_angular_velocity': self.warehouse_config.max_angular_velocity,
            'collision_threshold': self.warehouse_config.collision_threshold,
            'battery_constraints': {
                'capacity': self.warehouse_config.battery_capacity,
                'critical_level': 300.0  # 5 minutes
            },
            'dynamics_model': 'differential_drive',
            'communication_range': self.warehouse_config.communication_range
        }
    
    def validate_action_physics(self, action: np.ndarray, agent_id: int) -> Tuple[bool, Optional[str]]:
        """
        Validate robot action against physics constraints
        
        Action format: [linear_velocity, angular_velocity]
        """
        if len(action) != 2:
            return False, "Invalid action dimension"
        
        linear_vel, angular_vel = action
        
        # Check velocity limits
        if abs(linear_vel) > self.warehouse_config.max_velocity:
            return False, f"Linear velocity {linear_vel} exceeds max {self.warehouse_config.max_velocity}"
        
        if abs(angular_vel) > self.warehouse_config.max_angular_velocity:
            return False, f"Angular velocity {angular_vel} exceeds max {self.warehouse_config.max_angular_velocity}"
        
        # Check battery
        robot = self.robots[agent_id]
        if robot['battery'] < 60.0:  # 1 minute remaining
            if linear_vel != 0 or angular_vel != 0:
                return False, "Battery too low for movement"
        
        # Check collision prediction
        future_pos = self._predict_position(agent_id, action, dt=0.5)
        for other_id, other_robot in self.robots.items():
            if other_id == agent_id:
                continue
            
            distance = np.linalg.norm(future_pos - other_robot['position'][:2])
            if distance < self.warehouse_config.collision_threshold:
                return False, f"Collision predicted with robot {other_id}"
        
        # Check carrying capacity
        if robot['carrying_item'] is not None:
            # Reduce max speed when carrying items
            if abs(linear_vel) > self.warehouse_config.max_velocity * 0.7:
                return False, "Velocity too high while carrying item"
        
        return True, None
    
    def _predict_position(self, agent_id: int, action: np.ndarray, dt: float) -> np.ndarray:
        """Predict future position given action"""
        robot = self.robots[agent_id]
        linear_vel, angular_vel = action
        
        # Simple forward prediction
        new_orientation = robot['orientation'] + angular_vel * dt
        dx = linear_vel * np.cos(new_orientation) * dt
        dy = linear_vel * np.sin(new_orientation) * dt
        
        return robot['position'][:2] + np.array([dx, dy])
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Execute one step in the warehouse
        
        Returns:
            (observations, rewards, dones, info)
        """
        # Validate all actions
        safe_actions = {}
        for agent_id, action in actions.items():
            is_valid, violation = self.validate_action_physics(action, agent_id)
            if is_valid:
                safe_actions[agent_id] = action
            else:
                # Use safe fallback action
                safe_actions[agent_id] = self._get_safe_action(
                    self._get_robot_state(agent_id)
                )
                self.metrics.safety_violations += 1
        
        # Execute actions
        for agent_id, action in safe_actions.items():
            self._execute_robot_action(agent_id, action)
        
        # Update physics simulation
        if self.sim_mode:
            p.stepSimulation()
        
        # Check collisions
        self._check_collisions()
        
        # Update tasks
        self._update_task_progress()
        
        # Update communication graph based on distances
        self._update_communication_graph()
        
        # Get observations
        observations = self._get_observations()
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Check termination
        dones = self._check_termination()
        
        # Additional info
        info = {
            'collisions': self.collision_pairs,
            'completed_deliveries': self.completed_deliveries,
            'battery_levels': {i: r['battery'] for i, r in self.robots.items()},
            'communication_graph': self.communication_graph
        }
        
        return observations, rewards, dones, info
    
    def _execute_robot_action(self, agent_id: int, action: np.ndarray):
        """Execute robot action with physics simulation"""
        robot = self.robots[agent_id]
        linear_vel, angular_vel = action
        
        # Update physics
        current_vel = robot['velocity']
        desired_vel = np.array([
            linear_vel * np.cos(robot['orientation']),
            linear_vel * np.sin(robot['orientation'])
        ])
        
        new_vel, power_consumed = robot['physics'].compute_dynamics(
            current_vel, desired_vel, dt=0.1
        )
        
        # Update robot state
        robot['velocity'] = new_vel
        robot['orientation'] += angular_vel * 0.1
        robot['position'][:2] += new_vel * 0.1
        robot['battery'] -= power_consumed
        
        # Send to hardware if in real mode
        if not self.sim_mode:
            self._send_hardware_command(action)
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all robots"""
        observations = {}
        
        for agent_id in range(self.warehouse_config.num_robots):
            # Own state
            robot = self.robots[agent_id]
            own_state = np.array([
                robot['position'][0] / self.warehouse_config.warehouse_size[0],
                robot['position'][1] / self.warehouse_config.warehouse_size[1],
                np.cos(robot['orientation']),
                np.sin(robot['orientation']),
                robot['velocity'][0] / self.warehouse_config.max_velocity,
                robot['velocity'][1] / self.warehouse_config.max_velocity,
                robot['battery'] / self.warehouse_config.battery_capacity,
                float(robot['carrying_item'] is not None)
            ])
            
            # Nearby robots (based on communication graph)
            nearby_robots = []
            for other_id in range(self.warehouse_config.num_robots):
                if other_id != agent_id and self.communication_graph[agent_id, other_id] > 0:
                    other = self.robots[other_id]
                    relative_pos = other['position'][:2] - robot['position'][:2]
                    nearby_robots.extend([
                        relative_pos[0] / self.warehouse_config.communication_range,
                        relative_pos[1] / self.warehouse_config.communication_range,
                        other['velocity'][0] / self.warehouse_config.max_velocity,
                        other['velocity'][1] / self.warehouse_config.max_velocity
                    ])
            
            # Pad if needed
            max_nearby = 5
            while len(nearby_robots) < max_nearby * 4:
                nearby_robots.extend([0, 0, 0, 0])
            
            # Task information
            if robot['task'] is not None:
                task_info = np.array([
                    robot['task']['pickup_location'][0] / self.warehouse_config.warehouse_size[0],
                    robot['task']['pickup_location'][1] / self.warehouse_config.warehouse_size[1],
                    robot['task']['delivery_location'][0] / self.warehouse_config.warehouse_size[0],
                    robot['task']['delivery_location'][1] / self.warehouse_config.warehouse_size[1]
                ])
            else:
                task_info = np.zeros(4)
            
            observations[agent_id] = np.concatenate([
                own_state,
                nearby_robots[:max_nearby * 4],
                task_info
            ])
        
        return observations
    
    def _calculate_rewards(self) -> Dict[int, float]:
        """Calculate rewards for each robot"""
        rewards = {}
        
        for agent_id in range(self.warehouse_config.num_robots):
            robot = self.robots[agent_id]
            reward = 0.0
            
            # Task progress reward
            if robot['task'] is not None:
                if robot['carrying_item'] is None:
                    # Moving toward pickup
                    dist_to_pickup = np.linalg.norm(
                        robot['position'][:2] - robot['task']['pickup_location']
                    )
                    reward -= 0.01 * dist_to_pickup
                    
                    # Pickup reward
                    if dist_to_pickup < 0.5:
                        reward += 10.0
                        robot['carrying_item'] = robot['task']['item_id']
                else:
                    # Moving toward delivery
                    dist_to_delivery = np.linalg.norm(
                        robot['position'][:2] - robot['task']['delivery_location']
                    )
                    reward -= 0.01 * dist_to_delivery
                    
                    # Delivery reward
                    if dist_to_delivery < 0.5:
                        reward += 20.0
                        self.completed_deliveries += 1
                        robot['carrying_item'] = None
                        robot['task'] = None
            
            # Energy efficiency penalty
            power_usage = robot['physics'].base_power_consumption + \
                         robot['physics'].motion_power_factor * np.linalg.norm(robot['velocity'])
            reward -= 0.001 * power_usage
            
            # Collision penalty
            if agent_id in [pair[0] for pair in self.collision_pairs] or \
               agent_id in [pair[1] for pair in self.collision_pairs]:
                reward -= 5.0
            
            # Low battery penalty
            if robot['battery'] < 300.0:  # 5 minutes
                reward -= 0.1
            
            # Coordination bonus (robots helping each other)
            for other_id, other_robot in self.robots.items():
                if other_id != agent_id and \
                   self.communication_graph[agent_id, other_id] > 0 and \
                   other_robot['task'] is not None and robot['task'] is not None:
                    # Bonus for being near robots with related tasks
                    if np.linalg.norm(robot['task']['pickup_location'] - 
                                    other_robot['task']['pickup_location']) < 2.0:
                        reward += 0.5
            
            rewards[agent_id] = reward
        
        return rewards
    
    def _check_collisions(self):
        """Check for robot collisions"""
        self.collision_pairs = []
        
        for i in range(self.warehouse_config.num_robots):
            for j in range(i + 1, self.warehouse_config.num_robots):
                dist = np.linalg.norm(
                    self.robots[i]['position'][:2] - self.robots[j]['position'][:2]
                )
                if dist < self.warehouse_config.collision_threshold:
                    self.collision_pairs.append((i, j))
                    self.metrics.collision_count += 1
    
    def _update_communication_graph(self):
        """Update communication based on robot distances"""
        for i in range(self.warehouse_config.num_robots):
            for j in range(self.warehouse_config.num_robots):
                if i == j:
                    self.communication_graph[i, j] = 1.0
                else:
                    dist = np.linalg.norm(
                        self.robots[i]['position'][:2] - self.robots[j]['position'][:2]
                    )
                    if dist < self.warehouse_config.communication_range:
                        # Signal strength decreases with distance
                        self.communication_graph[i, j] = 1.0 - (dist / self.warehouse_config.communication_range)
                    else:
                        self.communication_graph[i, j] = 0.0
    
    def _check_termination(self) -> Dict[int, bool]:
        """Check termination conditions"""
        dones = {}
        
        # Check if all tasks completed
        all_tasks_done = len(self.task_queue) == 0 and \
                        all(r['task'] is None for r in self.robots.values())
        
        # Check battery depletion
        any_battery_dead = any(r['battery'] <= 0 for r in self.robots.values())
        
        for agent_id in range(self.warehouse_config.num_robots):
            dones[agent_id] = all_tasks_done or any_battery_dead
        
        dones['__all__'] = all_tasks_done or any_battery_dead
        
        return dones
    
    # Real-world specific methods
    def _get_current_state(self) -> np.ndarray:
        """Get current state from simulation or hardware"""
        if self.sim_mode:
            # Get from simulation
            states = []
            for i in range(self.warehouse_config.num_robots):
                robot = self.robots[i]
                states.extend([
                    robot['position'][0],
                    robot['position'][1],
                    robot['orientation'],
                    robot['velocity'][0],
                    robot['velocity'][1],
                    robot['battery']
                ])
            return np.array(states)
        else:
            # Get from hardware
            states = []
            for i in range(self.warehouse_config.num_robots):
                robot_state = self.hardware_interface.get_robot_state(f"turtlebot_{i}")
                states.extend(robot_state)
            return np.array(states)
    
    def _compute_action(self, state: np.ndarray) -> np.ndarray:
        """Compute action using trained policy"""
        # This would use the trained PI-HMARL policy
        # For now, return a simple action
        return np.array([0.2, 0.0])  # Move forward slowly
    
    def _get_safe_action(self, state: np.ndarray) -> np.ndarray:
        """Get safe fallback action"""
        # Stop moving
        return np.array([0.0, 0.0])
    
    def _send_hardware_command(self, action: np.ndarray):
        """Send velocity command to TurtleBot3"""
        if not self.sim_mode and self.hardware_interface:
            linear_vel, angular_vel = action
            
            # Create Twist message for ROS2
            twist_msg = {
                'linear': {'x': linear_vel, 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': angular_vel}
            }
            
            self.hardware_interface.publish_velocity_command(twist_msg)
    
    # Safety check implementations
    def _check_communication(self) -> bool:
        """Check ROS2 communication"""
        if self.sim_mode:
            return True
        
        return self.hardware_interface.check_connection_status()
    
    def _check_sensors(self) -> bool:
        """Check robot sensors (lidar, cameras, etc.)"""
        if self.sim_mode:
            return True
        
        for i in range(self.warehouse_config.num_robots):
            if not self.hardware_interface.check_sensor_status(f"turtlebot_{i}"):
                return False
        return True
    
    def _check_actuators(self) -> bool:
        """Check robot motors"""
        if self.sim_mode:
            return True
        
        for i in range(self.warehouse_config.num_robots):
            if not self.hardware_interface.check_motor_status(f"turtlebot_{i}"):
                return False
        return True
    
    def _check_emergency_stop(self) -> bool:
        """Check emergency stop system"""
        return not self.emergency_stop
    
    def _check_workspace(self) -> bool:
        """Check if workspace is clear"""
        if self.sim_mode:
            return True
        
        # In real mode, this would check for obstacles using sensors
        return self.hardware_interface.check_workspace_clear()
    
    def _detect_collision(self) -> bool:
        """Detect if collision occurred"""
        return len(self.collision_pairs) > 0
    
    def _check_communication_health(self) -> bool:
        """Check if communication is healthy"""
        if self.sim_mode:
            return True
        
        return self.hardware_interface.get_latency() < 100  # ms
    
    def _detect_hardware_failure(self) -> bool:
        """Detect hardware failures"""
        if self.sim_mode:
            return False
        
        for i in range(self.warehouse_config.num_robots):
            if self.hardware_interface.get_error_count(f"turtlebot_{i}") > 10:
                return True
        return False
    
    # Helper methods
    def _generate_warehouse_layout(self):
        """Generate warehouse shelves and layout"""
        # Create grid of shelves
        rows, cols = 3, 4
        shelf_spacing_x = self.warehouse_config.warehouse_size[0] / (cols + 1)
        shelf_spacing_y = self.warehouse_config.warehouse_size[1] / (rows + 1)
        
        shelf_id = 0
        for row in range(rows):
            for col in range(cols):
                x = (col + 1) * shelf_spacing_x
                y = (row + 1) * shelf_spacing_y
                self.shelves[shelf_id] = {
                    'position': np.array([x, y, 0]),
                    'items': []
                }
                shelf_id += 1
                
                if self.sim_mode:
                    # Create visual shelf in simulation
                    shelf_size = [1.0, 1.0, 2.0]
                    shelf_visual = p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=[s/2 for s in shelf_size]
                    )
                    shelf_collision = p.createCollisionShape(
                        p.GEOM_BOX,
                        halfExtents=[s/2 for s in shelf_size]
                    )
                    p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=shelf_visual,
                        baseCollisionShapeIndex=shelf_collision,
                        basePosition=[x, y, 1.0]
                    )
    
    def _get_initial_robot_position(self, robot_id: int) -> np.ndarray:
        """Get initial position for robot"""
        # Distribute robots around the warehouse perimeter
        angle = (robot_id / self.warehouse_config.num_robots) * 2 * np.pi
        radius = min(self.warehouse_config.warehouse_size) * 0.4
        
        x = self.warehouse_config.warehouse_size[0] / 2 + radius * np.cos(angle)
        y = self.warehouse_config.warehouse_size[1] / 2 + radius * np.sin(angle)
        
        return np.array([x, y, 0])
    
    def _initialize_task_queue(self):
        """Initialize delivery tasks"""
        # Create random delivery tasks
        for i in range(self.warehouse_config.num_items):
            pickup_shelf = np.random.randint(0, len(self.shelves))
            delivery_shelf = np.random.randint(0, len(self.shelves))
            
            while delivery_shelf == pickup_shelf:
                delivery_shelf = np.random.randint(0, len(self.shelves))
            
            task = {
                'item_id': i,
                'pickup_location': self.shelves[pickup_shelf]['position'][:2],
                'delivery_location': self.shelves[delivery_shelf]['position'][:2],
                'priority': np.random.randint(1, 4),
                'deadline': np.random.uniform(300, 600)  # seconds
            }
            
            self.task_queue.append(task)
        
        # Sort by priority
        self.task_queue.sort(key=lambda x: x['priority'], reverse=True)
    
    def _create_warehouse_walls(self):
        """Create warehouse walls in simulation"""
        if not self.sim_mode:
            return
        
        wall_thickness = 0.1
        wall_height = 3.0
        
        # Create walls
        walls = [
            # North wall
            ([self.warehouse_config.warehouse_size[0]/2, 0, wall_height/2],
             [self.warehouse_config.warehouse_size[0]/2, wall_thickness/2, wall_height/2]),
            # South wall  
            ([self.warehouse_config.warehouse_size[0]/2, self.warehouse_config.warehouse_size[1], wall_height/2],
             [self.warehouse_config.warehouse_size[0]/2, wall_thickness/2, wall_height/2]),
            # East wall
            ([self.warehouse_config.warehouse_size[0], self.warehouse_config.warehouse_size[1]/2, wall_height/2],
             [wall_thickness/2, self.warehouse_config.warehouse_size[1]/2, wall_height/2]),
            # West wall
            ([0, self.warehouse_config.warehouse_size[1]/2, wall_height/2],
             [wall_thickness/2, self.warehouse_config.warehouse_size[1]/2, wall_height/2])
        ]
        
        for position, half_extents in walls:
            wall_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=[0.8, 0.8, 0.8, 1]
            )
            wall_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=wall_visual,
                baseCollisionShapeIndex=wall_collision,
                basePosition=position
            )
    
    def _update_task_progress(self):
        """Update task assignments and progress"""
        # Assign tasks to idle robots
        for robot_id, robot in self.robots.items():
            if robot['task'] is None and len(self.task_queue) > 0:
                # Assign next task
                robot['task'] = self.task_queue.pop(0)
                self.active_tasks[robot_id] = robot['task']
    
    def _get_robot_state(self, agent_id: int) -> np.ndarray:
        """Get state for a specific robot"""
        robot = self.robots[agent_id]
        return np.array([
            robot['position'][0],
            robot['position'][1],
            robot['orientation'],
            robot['velocity'][0],
            robot['velocity'][1],
            robot['battery']
        ])