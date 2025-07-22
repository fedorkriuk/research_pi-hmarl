"""
Gazebo Connector for Crazyflie drone swarm simulation
Provides sim-to-real pipeline for quadcopter experiments
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import threading
import time
from dataclasses import dataclass
import socket
import json

# Note: In actual deployment, these would be real Gazebo imports
try:
    import rospy
    from gazebo_msgs.msg import ModelState, ModelStates
    from gazebo_msgs.srv import SetModelState, GetModelState
    from geometry_msgs.msg import Pose, Twist, Wrench
    from std_msgs.msg import Empty
    GAZEBO_AVAILABLE = True
except ImportError:
    GAZEBO_AVAILABLE = False

@dataclass 
class DroneState:
    """State information for a single drone"""
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray  # Quaternion
    angular_velocity: np.ndarray
    motor_speeds: np.ndarray  # 4 motors
    battery_level: float
    rssi: float  # Radio signal strength
    last_update: float

class GazeboConnector:
    """
    Gazebo connector for Crazyflie swarm simulation
    Enables high-fidelity simulation before hardware deployment
    """
    
    def __init__(self, num_drones: int, world_name: str = "drone_delivery"):
        """
        Initialize Gazebo connector
        
        Args:
            num_drones: Number of drones in swarm
            world_name: Gazebo world file name
        """
        self.num_drones = num_drones
        self.world_name = world_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Drone tracking
        self.drone_states = {}
        for i in range(num_drones):
            self.drone_states[i] = DroneState(
                position=np.zeros(3),
                velocity=np.zeros(3),
                orientation=np.array([0, 0, 0, 1]),
                angular_velocity=np.zeros(3),
                motor_speeds=np.zeros(4),
                battery_level=100.0,
                rssi=-40.0,  # Good signal
                last_update=time.time()
            )
        
        # Crazyflie connections (simulated)
        self.cf_connections = {}
        self.control_queues = {i: [] for i in range(num_drones)}
        
        # Gazebo communication
        self.gazebo_thread = None
        self.running = False
        
        if GAZEBO_AVAILABLE:
            self._initialize_gazebo()
        else:
            self.logger.warning("Gazebo not available - running in mock mode")
            self._initialize_mock()
    
    def _initialize_gazebo(self):
        """Initialize Gazebo connection"""
        rospy.init_node('pi_hmarl_gazebo_connector')
        
        # Service clients
        self.set_model_state = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState
        )
        self.get_model_state = rospy.ServiceProxy(
            '/gazebo/get_model_state', GetModelState
        )
        
        # Subscribers
        self.model_states_sub = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, self._model_states_callback
        )
        
        # Publishers for drone control
        self.drone_publishers = {}
        for i in range(self.num_drones):
            drone_name = f"crazyflie_{i}"
            
            # Motor command publishers
            for motor in range(4):
                topic = f"/{drone_name}/motor_{motor}/command"
                pub = rospy.Publisher(topic, Wrench, queue_size=1)
                self.drone_publishers[f"{drone_name}_motor_{motor}"] = pub
        
        # Start Gazebo update thread
        self.running = True
        self.gazebo_thread = threading.Thread(target=self._gazebo_loop, daemon=True)
        self.gazebo_thread.start()
        
        self.logger.info(f"Connected to Gazebo with {self.num_drones} drones")
    
    def _initialize_mock(self):
        """Initialize mock simulation for testing"""
        self.logger.info("Initializing mock Gazebo connector")
        
        # Start mock physics thread
        self.running = True
        self.gazebo_thread = threading.Thread(target=self._mock_physics_loop, daemon=True)
        self.gazebo_thread.start()
    
    def connect_crazyflie(self, uri: str, drone_id: int) -> 'CrazyflieConnection':
        """
        Create connection to Crazyflie (simulated in Gazebo)
        
        Args:
            uri: Crazyflie URI (e.g., "radio://0/80/2M/E7E7E7E7F1")
            drone_id: Drone identifier
            
        Returns:
            Connection object
        """
        if drone_id in self.cf_connections:
            self.logger.warning(f"Drone {drone_id} already connected")
            return self.cf_connections[drone_id]
        
        connection = CrazyflieConnection(self, drone_id, uri)
        self.cf_connections[drone_id] = connection
        
        # In Gazebo, spawn or reset drone model
        if GAZEBO_AVAILABLE:
            self._spawn_drone_model(drone_id)
        
        return connection
    
    def send_control_command(self, drone_id: int, command: Dict[str, float]):
        """
        Send control command to drone
        
        Args:
            drone_id: Drone identifier
            command: Dict with 'thrust', 'roll', 'pitch', 'yaw' (rates)
        """
        if drone_id not in self.drone_states:
            self.logger.error(f"Unknown drone: {drone_id}")
            return
        
        # Convert high-level command to motor speeds
        motor_speeds = self._command_to_motor_speeds(command)
        
        # Update drone state
        self.drone_states[drone_id].motor_speeds = motor_speeds
        
        # Send to Gazebo
        if GAZEBO_AVAILABLE:
            self._send_motor_commands_gazebo(drone_id, motor_speeds)
        
        # Queue command for processing
        self.control_queues[drone_id].append({
            'command': command,
            'motor_speeds': motor_speeds,
            'timestamp': time.time()
        })
    
    def get_drone_state(self, drone_id: int) -> np.ndarray:
        """
        Get current drone state
        
        Returns:
            State vector [x, y, z, vx, vy, vz, qw, qx, qy, qz, battery]
        """
        if drone_id not in self.drone_states:
            return np.zeros(11)
        
        state = self.drone_states[drone_id]
        
        return np.concatenate([
            state.position,
            state.velocity,
            state.orientation,
            [state.battery_level]
        ])
    
    def check_sensors(self, drone_id: int) -> Dict[str, bool]:
        """Check drone sensor status"""
        if drone_id not in self.drone_states:
            return {'imu': False, 'baro': False, 'flow': False}
        
        # In simulation, sensors always work
        return {
            'imu': True,
            'baro': True,
            'flow': True,
            'battery': True
        }
    
    def check_motors(self, drone_id: int) -> List[bool]:
        """Check motor status"""
        if drone_id not in self.drone_states:
            return [False] * 4
        
        # Check if motors are responding
        motor_speeds = self.drone_states[drone_id].motor_speeds
        return [speed > 0 for speed in motor_speeds]
    
    def check_airspace_clear(self) -> bool:
        """Check if airspace is clear for flight"""
        # In simulation, check for obstacles or no-fly zones
        # For now, always return True
        return True
    
    def get_radio_strength(self, drone_id: int) -> float:
        """Get radio signal strength in dBm"""
        if drone_id not in self.drone_states:
            return -100.0  # Very weak
        
        return self.drone_states[drone_id].rssi
    
    def get_motor_health(self, drone_id: int) -> List[float]:
        """Get motor health status (0-1)"""
        if drone_id not in self.drone_states:
            return [0.0] * 4
        
        # In simulation, motors are always healthy
        # Could add wear simulation later
        return [1.0] * 4
    
    def get_imu_variance(self, drone_id: int) -> float:
        """Get IMU noise level"""
        # In simulation, return low noise
        return 0.01
    
    def _command_to_motor_speeds(self, command: Dict[str, float]) -> np.ndarray:
        """
        Convert high-level command to motor speeds
        
        Uses simplified mixing for X-configuration quadcopter
        """
        thrust = command.get('thrust', 0) / 65535.0  # Normalize from 16-bit
        roll = command.get('roll', 0) * np.pi / 180  # Convert to radians
        pitch = command.get('pitch', 0) * np.pi / 180
        yaw = command.get('yaw', 0) * np.pi / 180
        
        # Base thrust
        base_speed = 5000 + thrust * 3000  # 5000-8000 RPM range
        
        # Mixing matrix for X configuration
        # Motors: [Front-Right, Front-Left, Rear-Left, Rear-Right]
        motor_speeds = np.array([
            base_speed + roll * 500 + pitch * 500 - yaw * 200,
            base_speed - roll * 500 + pitch * 500 + yaw * 200,
            base_speed - roll * 500 - pitch * 500 - yaw * 200,
            base_speed + roll * 500 - pitch * 500 + yaw * 200
        ])
        
        return np.clip(motor_speeds, 0, 10000)
    
    def _spawn_drone_model(self, drone_id: int):
        """Spawn drone model in Gazebo"""
        drone_name = f"crazyflie_{drone_id}"
        
        # Initial position (distributed in a circle)
        angle = (drone_id / self.num_drones) * 2 * np.pi
        radius = 2.0
        x = 50 + radius * np.cos(angle)  # Center at (50, 50)
        y = 50 + radius * np.sin(angle)
        z = 1.0  # 1m initial height
        
        # Create model state
        model_state = ModelState()
        model_state.model_name = drone_name
        model_state.pose.position.x = x
        model_state.pose.position.y = y  
        model_state.pose.position.z = z
        model_state.pose.orientation.w = 1.0
        
        # Set in Gazebo
        try:
            self.set_model_state(model_state)
            self.logger.info(f"Spawned {drone_name} at ({x:.1f}, {y:.1f}, {z:.1f})")
        except Exception as e:
            self.logger.error(f"Failed to spawn {drone_name}: {e}")
    
    def _send_motor_commands_gazebo(self, drone_id: int, motor_speeds: np.ndarray):
        """Send motor commands to Gazebo"""
        drone_name = f"crazyflie_{drone_id}"
        
        for motor in range(4):
            # Convert RPM to force (simplified model)
            # F = k * omega^2
            k_thrust = 1e-6  # Thrust coefficient
            force = k_thrust * (motor_speeds[motor] ** 2)
            
            # Create wrench message
            wrench = Wrench()
            wrench.force.z = force
            
            # Publish
            pub_key = f"{drone_name}_motor_{motor}"
            if pub_key in self.drone_publishers:
                self.drone_publishers[pub_key].publish(wrench)
    
    def _model_states_callback(self, msg):
        """Handle model states from Gazebo"""
        for i, name in enumerate(msg.name):
            # Check if it's one of our drones
            if name.startswith("crazyflie_"):
                try:
                    drone_id = int(name.split("_")[1])
                    if drone_id in self.drone_states:
                        state = self.drone_states[drone_id]
                        
                        # Update position
                        pose = msg.pose[i]
                        state.position = np.array([
                            pose.position.x,
                            pose.position.y,
                            pose.position.z
                        ])
                        
                        # Update orientation
                        state.orientation = np.array([
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w
                        ])
                        
                        # Update velocities
                        twist = msg.twist[i]
                        state.velocity = np.array([
                            twist.linear.x,
                            twist.linear.y,
                            twist.linear.z
                        ])
                        state.angular_velocity = np.array([
                            twist.angular.x,
                            twist.angular.y,
                            twist.angular.z
                        ])
                        
                        state.last_update = time.time()
                        
                except (ValueError, IndexError):
                    pass
    
    def _gazebo_loop(self):
        """Main Gazebo update loop"""
        rate = rospy.Rate(100)  # 100 Hz
        
        while self.running and not rospy.is_shutdown():
            # Process control commands
            for drone_id, queue in self.control_queues.items():
                if queue:
                    # Get latest command
                    latest_cmd = queue[-1]
                    self.control_queues[drone_id] = []  # Clear queue
                    
                    # Apply physics constraints
                    self._apply_physics_constraints(drone_id, latest_cmd)
            
            # Update battery simulation
            self._update_battery_simulation()
            
            rate.sleep()
    
    def _mock_physics_loop(self):
        """Mock physics simulation when Gazebo not available"""
        dt = 0.01  # 100Hz
        
        while self.running:
            for drone_id, state in self.drone_states.items():
                # Simple physics update
                if drone_id in self.control_queues and self.control_queues[drone_id]:
                    # Get latest command
                    cmd = self.control_queues[drone_id][-1]
                    motor_speeds = cmd['motor_speeds']
                    
                    # Compute thrust
                    total_thrust = np.sum(motor_speeds**2) * 1e-6
                    
                    # Update velocity (simplified)
                    acceleration = np.array([0, 0, total_thrust / 0.5]) - np.array([0, 0, 9.81])
                    state.velocity += acceleration * dt
                    
                    # Update position
                    state.position += state.velocity * dt
                    
                    # Keep drone above ground
                    if state.position[2] < 0:
                        state.position[2] = 0
                        state.velocity[2] = 0
                
                # Update battery
                power = np.sum(state.motor_speeds) / 1000  # Simplified
                state.battery_level -= power * dt / 36  # Drain in 600s at full power
                state.battery_level = max(0, state.battery_level)
            
            time.sleep(dt)
    
    def _apply_physics_constraints(self, drone_id: int, command: Dict):
        """Apply physics constraints to commands"""
        state = self.drone_states[drone_id]
        
        # Check battery
        if state.battery_level < 10:
            # Force landing
            self.logger.warning(f"Drone {drone_id} battery critical - forcing landing")
            command['motor_speeds'] = np.ones(4) * 3000  # Gentle descent
        
        # Check altitude limits
        if state.position[2] > 50:  # Max altitude
            self.logger.warning(f"Drone {drone_id} exceeding altitude limit")
            command['motor_speeds'] *= 0.8  # Reduce thrust
    
    def _update_battery_simulation(self):
        """Update battery levels based on motor usage"""
        for drone_id, state in self.drone_states.items():
            # Power consumption model
            motor_power = np.sum(state.motor_speeds) / 1000  # W
            current_draw = motor_power / 11.1  # 3S LiPo voltage
            
            # Drain battery (600s flight time at hover)
            drain_rate = current_draw / 600
            state.battery_level -= drain_rate * 0.01  # 100Hz update
            state.battery_level = max(0, state.battery_level)
            
            # Simulate RSSI based on distance from base
            base_pos = np.array([50, 50, 0])
            distance = np.linalg.norm(state.position - base_pos)
            state.rssi = -40 - distance  # Simple distance-based model
    
    def shutdown(self):
        """Shutdown Gazebo connector"""
        self.logger.info("Shutting down Gazebo connector")
        
        self.running = False
        if self.gazebo_thread:
            self.gazebo_thread.join(timeout=2.0)
        
        if GAZEBO_AVAILABLE:
            # Land all drones
            for drone_id in range(self.num_drones):
                self.send_control_command(drone_id, {
                    'thrust': 0,
                    'roll': 0,
                    'pitch': 0,
                    'yaw': 0
                })

class CrazyflieConnection:
    """Simulated Crazyflie connection for Gazebo"""
    
    def __init__(self, connector: GazeboConnector, drone_id: int, uri: str):
        self.connector = connector
        self.drone_id = drone_id
        self.uri = uri
        self._connected = True
    
    def is_connected(self) -> bool:
        """Check connection status"""
        return self._connected and self.connector.running
    
    def send_setpoint(self, thrust: int, roll: float, pitch: float, yaw_rate: float):
        """Send control setpoint"""
        command = {
            'thrust': thrust,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw_rate
        }
        self.connector.send_control_command(self.drone_id, command)
    
    def get_state(self) -> Dict[str, float]:
        """Get drone state"""
        state_array = self.connector.get_drone_state(self.drone_id)
        return {
            'x': state_array[0],
            'y': state_array[1], 
            'z': state_array[2],
            'vx': state_array[3],
            'vy': state_array[4],
            'vz': state_array[5],
            'battery': state_array[10]
        }
    
    def emergency_stop(self):
        """Emergency stop"""
        self.send_setpoint(0, 0, 0, 0)