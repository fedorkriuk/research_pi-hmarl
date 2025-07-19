"""Hardware Integration Module

This module provides interfaces and implementations for integrating
the PI-HMARL system with real hardware platforms.
"""

from .drone_interface import (
    DroneInterface, PX4Interface, ArduPilotInterface,
    DJIInterface, DroneController
)
from .robot_interface import (
    RobotInterface, ROSInterface, ROS2Interface,
    TurtleBotInterface, RobotController
)
from .sensor_interface import (
    SensorInterface, CameraInterface, LiDARInterface,
    IMUInterface, GPSInterface, SensorFusion
)
from .actuator_interface import (
    ActuatorInterface, MotorController, ServoController,
    GimbalController, ActuatorManager
)
from .communication_interface import (
    HardwareCommunication, MAVLinkProtocol, ROSBridge,
    SerialInterface, NetworkInterface
)

__all__ = [
    # Drone interfaces
    'DroneInterface', 'PX4Interface', 'ArduPilotInterface',
    'DJIInterface', 'DroneController',
    
    # Robot interfaces
    'RobotInterface', 'ROSInterface', 'ROS2Interface',
    'TurtleBotInterface', 'RobotController',
    
    # Sensor interfaces
    'SensorInterface', 'CameraInterface', 'LiDARInterface',
    'IMUInterface', 'GPSInterface', 'SensorFusion',
    
    # Actuator interfaces
    'ActuatorInterface', 'MotorController', 'ServoController',
    'GimbalController', 'ActuatorManager',
    
    # Communication interfaces
    'HardwareCommunication', 'MAVLinkProtocol', 'ROSBridge',
    'SerialInterface', 'NetworkInterface'
]