"""Vehicle Dynamics Model

This module implements realistic vehicle dynamics for drones using
actual specifications from real drone models.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DroneParameters:
    """Real drone parameters based on actual models"""
    # Basic properties
    name: str
    mass: float  # kg
    arm_length: float  # m (distance from center to motor)
    
    # Inertia tensor components (kg*m^2)
    inertia_xx: float
    inertia_yy: float
    inertia_zz: float
    
    # Motor properties
    num_rotors: int
    max_rpm: float
    motor_constant: float  # N/(rad/s)^2
    moment_constant: float  # Nm/(rad/s)^2
    rotor_drag_coefficient: float
    rolling_moment_coefficient: float
    
    # Aerodynamic properties
    drag_coefficient_x: float
    drag_coefficient_y: float
    drag_coefficient_z: float
    reference_area: float  # m^2
    
    # Performance limits
    max_velocity: float  # m/s
    max_acceleration: float  # m/s^2
    max_angular_velocity: float  # rad/s
    max_angular_acceleration: float  # rad/s^2
    max_tilt_angle: float  # rad
    
    # Battery and power
    battery_capacity: float  # Wh
    hover_power: float  # W
    max_power: float  # W
    
    @classmethod
    def dji_mavic_3(cls) -> "DroneParameters":
        """DJI Mavic 3 parameters"""
        return cls(
            name="DJI Mavic 3",
            mass=0.895,  # 895g
            arm_length=0.15,  # Estimated
            inertia_xx=0.01,
            inertia_yy=0.01,
            inertia_zz=0.02,
            num_rotors=4,
            max_rpm=15000,
            motor_constant=8.54858e-06,
            moment_constant=0.016,
            rotor_drag_coefficient=0.0001,
            rolling_moment_coefficient=1e-6,
            drag_coefficient_x=0.5,
            drag_coefficient_y=0.5,
            drag_coefficient_z=0.7,
            reference_area=0.05,
            max_velocity=21.0,  # 21 m/s (75.6 km/h)
            max_acceleration=10.0,
            max_angular_velocity=3.14,  # ~180 deg/s
            max_angular_acceleration=6.28,
            max_tilt_angle=0.785,  # 45 degrees
            battery_capacity=77.0,  # 77 Wh
            hover_power=150.0,  # Estimated
            max_power=500.0  # Estimated
        )
    
    @classmethod
    def parrot_anafi(cls) -> "DroneParameters":
        """Parrot Anafi parameters"""
        return cls(
            name="Parrot Anafi",
            mass=0.320,  # 320g
            arm_length=0.10,
            inertia_xx=0.003,
            inertia_yy=0.003,
            inertia_zz=0.006,
            num_rotors=4,
            max_rpm=12000,
            motor_constant=8.54858e-06,
            moment_constant=0.012,
            rotor_drag_coefficient=0.0001,
            rolling_moment_coefficient=1e-6,
            drag_coefficient_x=0.4,
            drag_coefficient_y=0.4,
            drag_coefficient_z=0.6,
            reference_area=0.03,
            max_velocity=16.0,  # 16 m/s (57.6 km/h)
            max_acceleration=8.0,
            max_angular_velocity=2.5,
            max_angular_acceleration=5.0,
            max_tilt_angle=0.698,  # 40 degrees
            battery_capacity=42.0,  # 42 Wh
            hover_power=80.0,
            max_power=250.0
        )


class VehicleDynamics:
    """Realistic vehicle dynamics model for drones"""
    
    def __init__(self, parameters: DroneParameters):
        """Initialize vehicle dynamics
        
        Args:
            parameters: Drone parameters
        """
        self.params = parameters
        
        # State variables
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([0, 0, 0, 1])  # Quaternion [x,y,z,w]
        self.angular_velocity = np.zeros(3)
        
        # Motor states
        self.motor_speeds = np.zeros(parameters.num_rotors)  # rad/s
        self.motor_commands = np.zeros(parameters.num_rotors)  # 0-1
        
        # Forces and torques
        self.thrust_force = np.zeros(3)
        self.drag_force = np.zeros(3)
        self.torques = np.zeros(3)
        
        # Power tracking
        self.current_power = 0.0
        
        # Motor mixing matrix for quadcopter
        if parameters.num_rotors == 4:
            self._setup_quadcopter_mixing()
        
        logger.info(f"Initialized VehicleDynamics for {parameters.name}")
    
    def _setup_quadcopter_mixing(self):
        """Setup motor mixing matrix for quadcopter configuration"""
        # Standard X configuration
        # Motors: 0=front-right, 1=rear-left, 2=front-left, 3=rear-right
        # Spinning: 0,2 CCW, 1,3 CW
        
        L = self.params.arm_length
        
        # Mixing matrix: [thrust, roll, pitch, yaw] -> motor commands
        self.mixing_matrix = np.array([
            [0.25,  L/4,  L/4, -0.25],  # Motor 0
            [0.25, -L/4, -L/4, -0.25],  # Motor 1
            [0.25, -L/4,  L/4,  0.25],  # Motor 2
            [0.25,  L/4, -L/4,  0.25]   # Motor 3
        ])
        
        # Inverse mixing matrix
        self.mixing_matrix_inv = np.linalg.pinv(self.mixing_matrix)
    
    def set_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation: np.ndarray,
        angular_velocity: np.ndarray
    ):
        """Set vehicle state
        
        Args:
            position: Position [x, y, z]
            velocity: Velocity [vx, vy, vz]
            orientation: Quaternion [x, y, z, w]
            angular_velocity: Angular velocity [wx, wy, wz]
        """
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.orientation = orientation.copy()
        self.angular_velocity = angular_velocity.copy()
    
    def set_motor_commands(self, commands: np.ndarray):
        """Set motor commands
        
        Args:
            commands: Motor commands (0-1) for each motor
        """
        self.motor_commands = np.clip(commands, 0, 1)
    
    def set_control_inputs(
        self,
        thrust: float,
        roll_torque: float,
        pitch_torque: float,
        yaw_torque: float
    ):
        """Set control inputs using thrust and torques
        
        Args:
            thrust: Total thrust (N)
            roll_torque: Roll torque (Nm)
            pitch_torque: Pitch torque (Nm)
            yaw_torque: Yaw torque (Nm)
        """
        if self.params.num_rotors == 4:
            # Convert to motor commands using mixing matrix
            control_vector = np.array([thrust, roll_torque, pitch_torque, yaw_torque])
            motor_forces = self.mixing_matrix_inv @ control_vector
            
            # Convert forces to motor commands (simplified)
            max_thrust_per_motor = self.params.max_power / (4 * 100)  # Rough estimate
            commands = motor_forces / max_thrust_per_motor
            
            self.set_motor_commands(commands)
    
    def update(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Update vehicle dynamics
        
        Args:
            dt: Time step
            
        Returns:
            Total force and torque vectors
        """
        # Update motor speeds (first-order dynamics)
        motor_time_constant = 0.1  # seconds
        target_speeds = self.motor_commands * self.params.max_rpm * (2 * np.pi / 60)
        self.motor_speeds += (target_speeds - self.motor_speeds) * dt / motor_time_constant
        
        # Calculate thrust and torques
        total_force, total_torque = self._calculate_forces_and_torques()
        
        # Add drag forces
        self._calculate_drag_forces()
        total_force += self.drag_force
        
        # Calculate power consumption
        self._calculate_power_consumption()
        
        return total_force, total_torque
    
    def _calculate_forces_and_torques(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate forces and torques from motor speeds"""
        # Thrust force for each motor
        motor_thrusts = self.params.motor_constant * self.motor_speeds**2
        
        # Total thrust in body frame (pointing up)
        total_thrust = np.sum(motor_thrusts)
        self.thrust_force = np.array([0, 0, total_thrust])
        
        # Convert to world frame using orientation
        # Simplified rotation (assuming small angles)
        R = self._quaternion_to_rotation_matrix(self.orientation)
        thrust_world = R @ self.thrust_force
        
        # Calculate torques
        torques = np.zeros(3)
        
        if self.params.num_rotors == 4:
            # Roll torque (difference between left and right motors)
            L = self.params.arm_length
            torques[0] = L * (motor_thrusts[0] + motor_thrusts[3] - 
                             motor_thrusts[1] - motor_thrusts[2])
            
            # Pitch torque (difference between front and back motors)
            torques[1] = L * (motor_thrusts[0] + motor_thrusts[2] - 
                             motor_thrusts[1] - motor_thrusts[3])
            
            # Yaw torque (reaction torques from propellers)
            motor_torques = self.params.moment_constant * self.motor_speeds**2
            torques[2] = -motor_torques[0] - motor_torques[1] + \
                         motor_torques[2] + motor_torques[3]
        
        self.torques = torques
        
        return thrust_world, torques
    
    def _calculate_drag_forces(self):
        """Calculate aerodynamic drag forces"""
        # Velocity in body frame
        R = self._quaternion_to_rotation_matrix(self.orientation)
        v_body = R.T @ self.velocity
        
        # Drag force components
        drag_x = -0.5 * 1.225 * self.params.drag_coefficient_x * \
                 self.params.reference_area * v_body[0] * abs(v_body[0])
        drag_y = -0.5 * 1.225 * self.params.drag_coefficient_y * \
                 self.params.reference_area * v_body[1] * abs(v_body[1])
        drag_z = -0.5 * 1.225 * self.params.drag_coefficient_z * \
                 self.params.reference_area * v_body[2] * abs(v_body[2])
        
        # Convert back to world frame
        drag_body = np.array([drag_x, drag_y, drag_z])
        self.drag_force = R @ drag_body
    
    def _calculate_power_consumption(self):
        """Calculate current power consumption"""
        # Mechanical power for each motor
        motor_powers = self.params.motor_constant * self.motor_speeds**3
        
        # Electrical power (accounting for motor efficiency ~0.8)
        electrical_power = np.sum(motor_powers) / 0.8
        
        # Add base power consumption
        base_power = 20.0  # Electronics, sensors, etc.
        
        self.current_power = electrical_power + base_power
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        x, y, z, w = q
        
        R = np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
        
        return R
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get current state as dictionary"""
        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "orientation": self.orientation.copy(),
            "angular_velocity": self.angular_velocity.copy(),
            "motor_speeds": self.motor_speeds.copy(),
            "thrust_force": self.thrust_force.copy(),
            "drag_force": self.drag_force.copy(),
            "torques": self.torques.copy(),
            "power_consumption": self.current_power
        }
    
    def check_limits(self) -> Dict[str, bool]:
        """Check if vehicle is within operational limits"""
        violations = {}
        
        # Velocity limit
        speed = np.linalg.norm(self.velocity)
        violations["velocity"] = speed > self.params.max_velocity
        
        # Angular velocity limit
        angular_speed = np.linalg.norm(self.angular_velocity)
        violations["angular_velocity"] = angular_speed > self.params.max_angular_velocity
        
        # Tilt angle limit
        R = self._quaternion_to_rotation_matrix(self.orientation)
        tilt_angle = np.arccos(np.clip(R[2, 2], -1, 1))
        violations["tilt_angle"] = tilt_angle > self.params.max_tilt_angle
        
        # Power limit
        violations["power"] = self.current_power > self.params.max_power
        
        return violations