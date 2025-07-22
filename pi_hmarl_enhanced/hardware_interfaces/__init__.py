"""
Hardware interfaces for real-world robot integration
"""

from .ros2_bridge import ROS2Bridge
from .gazebo_connector import GazeboConnector

__all__ = ['ROS2Bridge', 'GazeboConnector']