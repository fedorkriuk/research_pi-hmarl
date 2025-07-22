"""
ROS2 Bridge for TurtleBot3 integration
Provides interface between PI-HMARL and real robots
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
import logging
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty

# Note: In actual deployment, these would be real ROS2 imports
# For now, we'll create mock interfaces that match ROS2 API
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
    from sensor_msgs.msg import LaserScan, BatteryState
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Bool, Float32
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    # Mock classes for development
    class Node:
        pass
    class Twist:
        pass

@dataclass
class RobotState:
    """State information for a single robot"""
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray  # Quaternion
    battery_level: float
    laser_scan: Optional[np.ndarray] = None
    is_connected: bool = True
    last_update: float = 0.0

class ROS2Bridge:
    """
    ROS2 Bridge for multi-robot coordination
    Handles communication between PI-HMARL and TurtleBot3 fleet
    """
    
    def __init__(self, robot_names: List[str], namespace: str = "/warehouse"):
        """
        Initialize ROS2 bridge
        
        Args:
            robot_names: List of robot identifiers
            namespace: ROS2 namespace for the fleet
        """
        self.robot_names = robot_names
        self.namespace = namespace
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Robot state tracking
        self.robot_states = {name: RobotState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
            battery_level=100.0
        ) for name in robot_names}
        
        # Communication tracking
        self.latencies = {name: [] for name in robot_names}
        self.message_queues = {name: Queue(maxsize=100) for name in robot_names}
        
        # ROS2 setup
        self.ros_nodes = {}
        self.publishers = {}
        self.subscribers = {}
        self.ros_thread = None
        self.running = False
        
        if ROS2_AVAILABLE:
            self._initialize_ros2()
        else:
            self.logger.warning("ROS2 not available - running in mock mode")
    
    def _initialize_ros2(self):
        """Initialize ROS2 nodes and topics"""
        rclpy.init()
        
        # Create node for each robot
        for robot_name in self.robot_names:
            node_name = f"pi_hmarl_{robot_name}"
            node = Node(node_name)
            self.ros_nodes[robot_name] = node
            
            # Publishers
            self.publishers[robot_name] = {
                'cmd_vel': node.create_publisher(
                    Twist, f"{self.namespace}/{robot_name}/cmd_vel", 10
                ),
                'emergency_stop': node.create_publisher(
                    Bool, f"{self.namespace}/{robot_name}/emergency_stop", 10
                )
            }
            
            # Subscribers
            self.subscribers[robot_name] = {
                'odom': node.create_subscription(
                    Odometry,
                    f"{self.namespace}/{robot_name}/odom",
                    lambda msg, r=robot_name: self._odom_callback(r, msg),
                    10
                ),
                'scan': node.create_subscription(
                    LaserScan,
                    f"{self.namespace}/{robot_name}/scan",
                    lambda msg, r=robot_name: self._scan_callback(r, msg),
                    10
                ),
                'battery': node.create_subscription(
                    BatteryState,
                    f"{self.namespace}/{robot_name}/battery_state",
                    lambda msg, r=robot_name: self._battery_callback(r, msg),
                    10
                )
            }
        
        # Start ROS2 spinning in separate thread
        self.running = True
        self.ros_thread = threading.Thread(target=self._ros_spin, daemon=True)
        self.ros_thread.start()
    
    def _ros_spin(self):
        """ROS2 spinning thread"""
        executor = rclpy.executors.MultiThreadedExecutor()
        for node in self.ros_nodes.values():
            executor.add_node(node)
        
        while self.running and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    
    def create_robot_node(self, robot_name: str, namespace: str) -> 'RobotNode':
        """Create a robot node for individual robot control"""
        return RobotNode(self, robot_name, namespace)
    
    def publish_velocity_command(self, robot_name: str, linear: np.ndarray, angular: np.ndarray):
        """
        Publish velocity command to robot
        
        Args:
            robot_name: Robot identifier
            linear: Linear velocity [x, y, z] in m/s
            angular: Angular velocity [x, y, z] in rad/s
        """
        if not ROS2_AVAILABLE:
            # Mock mode - just update state
            self.robot_states[robot_name].velocity = linear
            return
        
        if robot_name not in self.publishers:
            self.logger.error(f"Unknown robot: {robot_name}")
            return
        
        # Create and publish Twist message
        twist = Twist()
        twist.linear.x = float(linear[0])
        twist.linear.y = float(linear[1])
        twist.linear.z = float(linear[2])
        twist.angular.x = float(angular[0])
        twist.angular.y = float(angular[1])
        twist.angular.z = float(angular[2])
        
        # Track latency
        start_time = time.time()
        self.publishers[robot_name]['cmd_vel'].publish(twist)
        latency = (time.time() - start_time) * 1000  # ms
        
        self.latencies[robot_name].append(latency)
        if len(self.latencies[robot_name]) > 100:
            self.latencies[robot_name].pop(0)
    
    def emergency_stop_all(self):
        """Send emergency stop to all robots"""
        self.logger.critical("EMERGENCY STOP - All robots")
        
        for robot_name in self.robot_names:
            if ROS2_AVAILABLE and robot_name in self.publishers:
                stop_msg = Bool()
                stop_msg.data = True
                self.publishers[robot_name]['emergency_stop'].publish(stop_msg)
            
            # Also send zero velocity
            self.publish_velocity_command(
                robot_name,
                np.zeros(3),
                np.zeros(3)
            )
    
    def get_robot_state(self, robot_name: str) -> np.ndarray:
        """
        Get current state of robot
        
        Returns:
            State vector [x, y, theta, vx, vy, vtheta, battery]
        """
        if robot_name not in self.robot_states:
            self.logger.error(f"Unknown robot: {robot_name}")
            return np.zeros(7)
        
        state = self.robot_states[robot_name]
        
        # Convert quaternion to euler (just yaw for 2D robots)
        yaw = self._quaternion_to_yaw(state.orientation)
        
        return np.array([
            state.position[0],
            state.position[1],
            yaw,
            state.velocity[0],
            state.velocity[1],
            state.velocity[2],  # Angular velocity z
            state.battery_level
        ])
    
    def check_connection_status(self) -> bool:
        """Check if all robots are connected"""
        if not ROS2_AVAILABLE:
            return True  # Mock mode always connected
        
        all_connected = all(
            state.is_connected and 
            (time.time() - state.last_update) < 1.0  # 1 second timeout
            for state in self.robot_states.values()
        )
        
        return all_connected
    
    def check_sensor_status(self, robot_name: str) -> bool:
        """Check if robot sensors are working"""
        if robot_name not in self.robot_states:
            return False
        
        state = self.robot_states[robot_name]
        
        # Check if we're receiving sensor data
        has_scan = state.laser_scan is not None and len(state.laser_scan) > 0
        recent_update = (time.time() - state.last_update) < 0.5
        
        return has_scan and recent_update
    
    def check_motor_status(self, robot_name: str) -> bool:
        """Check motor status"""
        # In real implementation, would check motor diagnostics
        return self.robot_states[robot_name].is_connected
    
    def get_latency(self) -> float:
        """Get average communication latency in ms"""
        all_latencies = []
        for robot_latencies in self.latencies.values():
            all_latencies.extend(robot_latencies)
        
        if not all_latencies:
            return 0.0
        
        return np.mean(all_latencies)
    
    def get_error_count(self, robot_name: str) -> int:
        """Get error count for robot"""
        # In real implementation, would track actual errors
        if robot_name not in self.robot_states:
            return 999
        
        # Check for stale data
        if (time.time() - self.robot_states[robot_name].last_update) > 2.0:
            return 10  # High error count for stale data
        
        return 0
    
    def check_workspace_clear(self) -> bool:
        """Check if workspace is clear using laser scans"""
        for robot_name, state in self.robot_states.items():
            if state.laser_scan is not None:
                # Check for close obstacles
                min_distance = np.min(state.laser_scan)
                if min_distance < 0.5:  # 0.5m safety threshold
                    self.logger.warning(
                        f"Obstacle detected near {robot_name}: {min_distance:.2f}m"
                    )
                    return False
        
        return True
    
    # ROS2 Callbacks
    def _odom_callback(self, robot_name: str, msg):
        """Handle odometry updates"""
        state = self.robot_states[robot_name]
        
        # Update position
        state.position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        # Update orientation
        state.orientation = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
        
        # Update velocity
        state.velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.angular.z
        ])
        
        state.last_update = time.time()
        state.is_connected = True
    
    def _scan_callback(self, robot_name: str, msg):
        """Handle laser scan updates"""
        state = self.robot_states[robot_name]
        
        # Convert laser scan to numpy array
        ranges = np.array(msg.ranges)
        
        # Replace inf values with max range
        ranges[np.isinf(ranges)] = msg.range_max
        
        state.laser_scan = ranges
    
    def _battery_callback(self, robot_name: str, msg):
        """Handle battery state updates"""
        state = self.robot_states[robot_name]
        
        # Convert battery percentage
        if msg.percentage >= 0:
            state.battery_level = msg.percentage
        else:
            # Calculate from voltage if percentage not available
            # Assuming 12V nominal, 10V empty, 12.6V full
            voltage_percent = (msg.voltage - 10.0) / 2.6 * 100
            state.battery_level = np.clip(voltage_percent, 0, 100)
    
    def _quaternion_to_yaw(self, quaternion: np.ndarray) -> float:
        """Convert quaternion to yaw angle"""
        x, y, z, w = quaternion
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return yaw
    
    def shutdown(self):
        """Shutdown ROS2 bridge"""
        self.logger.info("Shutting down ROS2 bridge")
        
        # Stop all robots
        self.emergency_stop_all()
        
        # Stop ROS2
        self.running = False
        if self.ros_thread:
            self.ros_thread.join(timeout=2.0)
        
        if ROS2_AVAILABLE:
            for node in self.ros_nodes.values():
                node.destroy_node()
            rclpy.shutdown()

class RobotNode:
    """Individual robot node for simplified control"""
    
    def __init__(self, bridge: ROS2Bridge, robot_name: str, namespace: str):
        self.bridge = bridge
        self.robot_name = robot_name
        self.namespace = namespace
    
    def publish_velocity(self, linear_x: float, angular_z: float):
        """Publish simple differential drive command"""
        linear = np.array([linear_x, 0, 0])
        angular = np.array([0, 0, angular_z])
        self.bridge.publish_velocity_command(self.robot_name, linear, angular)
    
    def get_state(self) -> Dict[str, Any]:
        """Get robot state as dictionary"""
        state_array = self.bridge.get_robot_state(self.robot_name)
        return {
            'x': state_array[0],
            'y': state_array[1],
            'theta': state_array[2],
            'vx': state_array[3],
            'vy': state_array[4],
            'vtheta': state_array[5],
            'battery': state_array[6]
        }
    
    def emergency_stop(self):
        """Emergency stop this robot"""
        self.publish_velocity(0.0, 0.0)