"""Physics Engine Module for PI-HMARL

This module provides realistic physics simulation for multi-agent systems
using real-world parameters from actual drones and environmental conditions.
"""

from .base_physics import PhysicsEngine
from .pybullet_engine import PyBulletEngine
from .vehicle_dynamics import VehicleDynamics, DroneParameters
from .battery_model import BatteryModel, BatteryParameters
from .collision_detection import CollisionDetector
from .environmental_factors import EnvironmentalFactors
from .physics_validator import PhysicsValidator
from .physics_utils import (
    quaternion_to_euler,
    euler_to_quaternion,
    calculate_drag_force,
    calculate_lift_force
)

__all__ = [
    "PhysicsEngine",
    "PyBulletEngine", 
    "VehicleDynamics",
    "DroneParameters",
    "BatteryModel",
    "BatteryParameters",
    "CollisionDetector",
    "EnvironmentalFactors",
    "PhysicsValidator",
    "quaternion_to_euler",
    "euler_to_quaternion",
    "calculate_drag_force",
    "calculate_lift_force"
]