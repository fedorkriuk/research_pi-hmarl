"""Physics Utility Functions

This module provides utility functions for physics calculations,
coordinate transformations, and common physics operations.
"""

import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles (roll, pitch, yaw)
    
    Args:
        q: Quaternion [x, y, z, w]
        
    Returns:
        Euler angles [roll, pitch, yaw] in radians
    """
    x, y, z, w = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles to quaternion
    
    Args:
        roll: Roll angle (radians)
        pitch: Pitch angle (radians)
        yaw: Yaw angle (radians)
        
    Returns:
        Quaternion [x, y, z, w]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([x, y, z, w])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions
    
    Args:
        q1: First quaternion [x, y, z, w]
        q2: Second quaternion [x, y, z, w]
        
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


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Get quaternion conjugate
    
    Args:
        q: Quaternion [x, y, z, w]
        
    Returns:
        Conjugate quaternion
    """
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix
    
    Args:
        q: Quaternion [x, y, z, w]
        
    Returns:
        3x3 rotation matrix
    """
    x, y, z, w = q
    
    # Normalize quaternion
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
        x, y, z, w = q
    
    # Build rotation matrix
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    return R


def rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate vector by quaternion
    
    Args:
        v: Vector to rotate
        q: Rotation quaternion
        
    Returns:
        Rotated vector
    """
    # Convert vector to quaternion form
    v_quat = np.array([v[0], v[1], v[2], 0])
    
    # Rotate: v' = q * v * q^*
    q_conj = quaternion_conjugate(q)
    v_rotated = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    
    return v_rotated[:3]


def calculate_drag_force(
    velocity: np.ndarray,
    air_density: float,
    drag_coefficient: float,
    reference_area: float
) -> np.ndarray:
    """Calculate aerodynamic drag force
    
    Args:
        velocity: Velocity vector (m/s)
        air_density: Air density (kg/m³)
        drag_coefficient: Drag coefficient
        reference_area: Reference area (m²)
        
    Returns:
        Drag force vector (N)
    """
    v_mag = np.linalg.norm(velocity)
    
    if v_mag < 0.01:  # Nearly stationary
        return np.zeros(3)
    
    # Drag force: F = -0.5 * ρ * Cd * A * v²
    drag_magnitude = 0.5 * air_density * drag_coefficient * reference_area * v_mag**2
    drag_direction = -velocity / v_mag
    
    return drag_magnitude * drag_direction


def calculate_lift_force(
    velocity: np.ndarray,
    air_density: float,
    lift_coefficient: float,
    reference_area: float,
    angle_of_attack: float
) -> np.ndarray:
    """Calculate aerodynamic lift force
    
    Args:
        velocity: Velocity vector (m/s)
        air_density: Air density (kg/m³)
        lift_coefficient: Lift coefficient
        reference_area: Reference area (m²)
        angle_of_attack: Angle of attack (radians)
        
    Returns:
        Lift force vector (N)
    """
    v_mag = np.linalg.norm(velocity)
    
    if v_mag < 0.01:
        return np.zeros(3)
    
    # Lift coefficient varies with angle of attack
    cl = lift_coefficient * np.sin(2 * angle_of_attack)
    
    # Lift magnitude: L = 0.5 * ρ * Cl * A * v²
    lift_magnitude = 0.5 * air_density * cl * reference_area * v_mag**2
    
    # Lift direction (perpendicular to velocity, typically upward)
    velocity_normalized = velocity / v_mag
    
    # Assume lift is perpendicular to velocity in the vertical plane
    lift_direction = np.cross(velocity_normalized, np.array([0, 0, 1]))
    if np.linalg.norm(lift_direction) > 0:
        lift_direction = lift_direction / np.linalg.norm(lift_direction)
        lift_direction = np.cross(lift_direction, velocity_normalized)
    else:
        # Velocity is vertical, no lift
        return np.zeros(3)
    
    return lift_magnitude * lift_direction


def calculate_thrust_from_motors(
    motor_speeds: np.ndarray,
    thrust_constant: float,
    motor_positions: np.ndarray,
    motor_directions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate total thrust force and torque from motors
    
    Args:
        motor_speeds: Motor speeds (rad/s)
        thrust_constant: Thrust constant (N/(rad/s)²)
        motor_positions: Motor positions relative to COM
        motor_directions: Motor thrust directions (unit vectors)
        
    Returns:
        Total thrust force and torque vectors
    """
    total_force = np.zeros(3)
    total_torque = np.zeros(3)
    
    for i, speed in enumerate(motor_speeds):
        # Thrust from motor
        thrust_magnitude = thrust_constant * speed**2
        thrust_force = thrust_magnitude * motor_directions[i]
        
        # Add to total force
        total_force += thrust_force
        
        # Torque from thrust (r × F)
        torque = np.cross(motor_positions[i], thrust_force)
        total_torque += torque
        
        # Add motor reaction torque (alternating direction)
        reaction_torque = (-1)**i * 0.01 * speed**2  # Simplified model
        total_torque[2] += reaction_torque  # Yaw torque
    
    return total_force, total_torque


def calculate_moment_of_inertia(
    mass: float,
    shape: str,
    dimensions: List[float]
) -> np.ndarray:
    """Calculate moment of inertia for common shapes
    
    Args:
        mass: Object mass (kg)
        shape: Shape type (box, cylinder, sphere)
        dimensions: Shape dimensions
        
    Returns:
        Inertia tensor diagonal elements [Ixx, Iyy, Izz]
    """
    if shape == "box":
        # Box with dimensions [length, width, height]
        l, w, h = dimensions
        Ixx = mass * (w**2 + h**2) / 12
        Iyy = mass * (l**2 + h**2) / 12
        Izz = mass * (l**2 + w**2) / 12
        
    elif shape == "cylinder":
        # Cylinder with [radius, height]
        r, h = dimensions
        Ixx = mass * (3 * r**2 + h**2) / 12
        Iyy = Ixx
        Izz = mass * r**2 / 2
        
    elif shape == "sphere":
        # Sphere with radius
        r = dimensions[0]
        I = 2 * mass * r**2 / 5
        Ixx = Iyy = Izz = I
        
    else:
        # Default to sphere approximation
        r = dimensions[0] if dimensions else 0.1
        I = 2 * mass * r**2 / 5
        Ixx = Iyy = Izz = I
    
    return np.array([Ixx, Iyy, Izz])


def world_to_body_frame(
    vector_world: np.ndarray,
    orientation: np.ndarray
) -> np.ndarray:
    """Transform vector from world frame to body frame
    
    Args:
        vector_world: Vector in world frame
        orientation: Body orientation quaternion
        
    Returns:
        Vector in body frame
    """
    # Rotate by inverse (conjugate) of orientation
    q_inv = quaternion_conjugate(orientation)
    return rotate_vector_by_quaternion(vector_world, q_inv)


def body_to_world_frame(
    vector_body: np.ndarray,
    orientation: np.ndarray
) -> np.ndarray:
    """Transform vector from body frame to world frame
    
    Args:
        vector_body: Vector in body frame
        orientation: Body orientation quaternion
        
    Returns:
        Vector in world frame
    """
    return rotate_vector_by_quaternion(vector_body, orientation)


def calculate_reynolds_number(
    velocity: float,
    characteristic_length: float,
    kinematic_viscosity: float = 1.5e-5  # Air at 20°C
) -> float:
    """Calculate Reynolds number
    
    Args:
        velocity: Flow velocity (m/s)
        characteristic_length: Characteristic length (m)
        kinematic_viscosity: Kinematic viscosity (m²/s)
        
    Returns:
        Reynolds number
    """
    return velocity * characteristic_length / kinematic_viscosity


def interpolate_trajectory(
    waypoints: List[np.ndarray],
    times: List[float],
    t: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate position and velocity along trajectory
    
    Args:
        waypoints: List of position waypoints
        times: Time at each waypoint
        t: Query time
        
    Returns:
        Interpolated position and velocity
    """
    if t <= times[0]:
        return waypoints[0], np.zeros(3)
    
    if t >= times[-1]:
        return waypoints[-1], np.zeros(3)
    
    # Find surrounding waypoints
    for i in range(len(times) - 1):
        if times[i] <= t <= times[i + 1]:
            # Linear interpolation
            dt = times[i + 1] - times[i]
            alpha = (t - times[i]) / dt
            
            position = (1 - alpha) * waypoints[i] + alpha * waypoints[i + 1]
            velocity = (waypoints[i + 1] - waypoints[i]) / dt
            
            return position, velocity
    
    return waypoints[-1], np.zeros(3)


def check_line_of_sight(
    pos1: np.ndarray,
    pos2: np.ndarray,
    obstacles: List[Dict[str, np.ndarray]]
) -> bool:
    """Check if there's line of sight between two positions
    
    Args:
        pos1: First position
        pos2: Second position
        obstacles: List of obstacles with 'position' and 'size'
        
    Returns:
        True if line of sight exists
    """
    ray_origin = pos1
    ray_direction = pos2 - pos1
    ray_length = np.linalg.norm(ray_direction)
    
    if ray_length < 0.001:
        return True
    
    ray_direction = ray_direction / ray_length
    
    for obstacle in obstacles:
        obs_pos = obstacle['position']
        obs_size = obstacle.get('size', np.array([1, 1, 1]))
        
        # Simple sphere approximation for obstacle
        obs_radius = np.max(obs_size) / 2
        
        # Ray-sphere intersection
        oc = ray_origin - obs_pos
        b = 2 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - obs_radius**2
        discriminant = b**2 - 4*c
        
        if discriminant >= 0:
            # Check if intersection is within ray length
            t1 = (-b - np.sqrt(discriminant)) / 2
            t2 = (-b + np.sqrt(discriminant)) / 2
            
            if 0 < t1 < ray_length or 0 < t2 < ray_length:
                return False
    
    return True