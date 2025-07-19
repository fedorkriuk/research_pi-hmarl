"""Actuator Hardware Interface

This module provides interfaces for controlling various actuators including
motors, servos, and gimbals.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from enum import Enum
import time
import threading
from queue import Queue

logger = logging.getLogger(__name__)


class ActuatorType(Enum):
    """Types of actuators"""
    DC_MOTOR = "dc_motor"
    SERVO = "servo"
    STEPPER = "stepper"
    BRUSHLESS = "brushless"
    LINEAR = "linear"
    GIMBAL = "gimbal"


class ControlMode(Enum):
    """Actuator control modes"""
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    CURRENT = "current"
    PWM = "pwm"


@dataclass
class ActuatorStatus:
    """Actuator status information"""
    actuator_id: str
    actuator_type: ActuatorType
    position: float  # radians or meters
    velocity: float  # rad/s or m/s
    torque: float  # Nm
    current: float  # Amps
    temperature: float  # Celsius
    enabled: bool
    fault: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class ActuatorCommand:
    """Command for actuator control"""
    mode: ControlMode
    target: float
    feed_forward: Optional[float] = None
    duration: Optional[float] = None  # For timed movements


class ActuatorInterface(ABC):
    """Abstract base class for actuator interfaces"""
    
    def __init__(self, actuator_id: str, actuator_type: ActuatorType):
        """Initialize actuator interface
        
        Args:
            actuator_id: Unique actuator identifier
            actuator_type: Type of actuator
        """
        self.actuator_id = actuator_id
        self.actuator_type = actuator_type
        self.is_connected = False
        self.is_enabled = False
        self.control_mode = ControlMode.POSITION
        
        # Limits
        self.position_limits = (-np.pi, np.pi)
        self.velocity_limit = 10.0  # rad/s
        self.torque_limit = 5.0  # Nm
        self.current_limit = 10.0  # Amps
        
        self._callbacks = {}
        
    @abstractmethod
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to actuator
        
        Args:
            config: Actuator configuration
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from actuator"""
        pass
    
    @abstractmethod
    async def enable(self) -> bool:
        """Enable actuator
        
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def disable(self) -> bool:
        """Disable actuator
        
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def send_command(self, command: ActuatorCommand) -> bool:
        """Send control command
        
        Args:
            command: Actuator command
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def get_status(self) -> ActuatorStatus:
        """Get current actuator status
        
        Returns:
            Current status
        """
        pass
    
    @abstractmethod
    async def home(self) -> bool:
        """Home the actuator
        
        Returns:
            Success status
        """
        pass
    
    def set_limits(
        self,
        position_limits: Optional[Tuple[float, float]] = None,
        velocity_limit: Optional[float] = None,
        torque_limit: Optional[float] = None,
        current_limit: Optional[float] = None
    ):
        """Set actuator limits
        
        Args:
            position_limits: Min and max position
            velocity_limit: Max velocity
            torque_limit: Max torque
            current_limit: Max current
        """
        if position_limits:
            self.position_limits = position_limits
        if velocity_limit:
            self.velocity_limit = velocity_limit
        if torque_limit:
            self.torque_limit = torque_limit
        if current_limit:
            self.current_limit = current_limit
    
    def register_callback(self, event: str, callback: Callable):
        """Register event callback
        
        Args:
            event: Event name
            callback: Callback function
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, data: Any):
        """Trigger event callbacks
        
        Args:
            event: Event name
            data: Event data
        """
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")


class MotorController(ActuatorInterface):
    """Controller for DC/Brushless motors"""
    
    def __init__(self, actuator_id: str, motor_type: str = "brushless"):
        """Initialize motor controller
        
        Args:
            actuator_id: Motor identifier
            motor_type: Type of motor (dc, brushless)
        """
        actuator_type = (
            ActuatorType.BRUSHLESS if motor_type == "brushless"
            else ActuatorType.DC_MOTOR
        )
        super().__init__(actuator_id, actuator_type)
        
        # Motor parameters
        self.encoder_resolution = 4096  # counts per revolution
        self.gear_ratio = 1.0
        self.kv_rating = 1000  # RPM/V for brushless
        
        # Control gains
        self.position_gains = {'kp': 1.0, 'ki': 0.1, 'kd': 0.01}
        self.velocity_gains = {'kp': 0.5, 'ki': 0.05, 'kd': 0.0}
        
        # State
        self.current_position = 0.0
        self.current_velocity = 0.0
        
        logger.info(f"Initialized MotorController for {actuator_id}")
    
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to motor controller
        
        Args:
            config: Motor configuration
            
        Returns:
            Success status
        """
        try:
            # In real implementation, would connect to motor driver
            # via CAN, serial, or PWM
            
            self.encoder_resolution = config.get('encoder_resolution', 4096)
            self.gear_ratio = config.get('gear_ratio', 1.0)
            
            self.is_connected = True
            logger.info(f"Connected to motor {self.actuator_id}")
            return True
            
        except Exception as e:
            logger.error(f"Motor connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from motor"""
        await self.disable()
        self.is_connected = False
        logger.info("Disconnected from motor")
    
    async def enable(self) -> bool:
        """Enable motor
        
        Returns:
            Success status
        """
        if not self.is_connected:
            return False
        
        # Send enable command to motor driver
        self.is_enabled = True
        logger.info(f"Motor {self.actuator_id} enabled")
        return True
    
    async def disable(self) -> bool:
        """Disable motor
        
        Returns:
            Success status
        """
        # Send disable command
        self.is_enabled = False
        logger.info(f"Motor {self.actuator_id} disabled")
        return True
    
    async def send_command(self, command: ActuatorCommand) -> bool:
        """Send motor command
        
        Args:
            command: Motor command
            
        Returns:
            Success status
        """
        if not self.is_connected or not self.is_enabled:
            return False
        
        try:
            # Validate command
            validated_command = self._validate_command(command)
            
            # Send command based on mode
            if validated_command.mode == ControlMode.POSITION:
                await self._position_control(validated_command.target)
            elif validated_command.mode == ControlMode.VELOCITY:
                await self._velocity_control(validated_command.target)
            elif validated_command.mode == ControlMode.TORQUE:
                await self._torque_control(validated_command.target)
            elif validated_command.mode == ControlMode.PWM:
                await self._pwm_control(validated_command.target)
            
            return True
            
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return False
    
    async def get_status(self) -> ActuatorStatus:
        """Get motor status
        
        Returns:
            Current status
        """
        # In real implementation, would read from motor driver
        return ActuatorStatus(
            actuator_id=self.actuator_id,
            actuator_type=self.actuator_type,
            position=self.current_position,
            velocity=self.current_velocity,
            torque=0.0,  # Would read from driver
            current=0.0,  # Would read from driver
            temperature=25.0,  # Would read from driver
            enabled=self.is_enabled,
            timestamp=time.time()
        )
    
    async def home(self) -> bool:
        """Home the motor
        
        Returns:
            Success status
        """
        if not self.is_connected:
            return False
        
        logger.info(f"Homing motor {self.actuator_id}")
        
        # Move to home switch or use index pulse
        # Simplified - just reset position
        self.current_position = 0.0
        
        return True
    
    def _validate_command(self, command: ActuatorCommand) -> ActuatorCommand:
        """Validate and limit command
        
        Args:
            command: Input command
            
        Returns:
            Validated command
        """
        validated = ActuatorCommand(
            mode=command.mode,
            target=command.target,
            feed_forward=command.feed_forward
        )
        
        # Apply limits based on mode
        if command.mode == ControlMode.POSITION:
            validated.target = np.clip(
                command.target,
                self.position_limits[0],
                self.position_limits[1]
            )
        elif command.mode == ControlMode.VELOCITY:
            validated.target = np.clip(
                command.target,
                -self.velocity_limit,
                self.velocity_limit
            )
        elif command.mode == ControlMode.TORQUE:
            validated.target = np.clip(
                command.target,
                -self.torque_limit,
                self.torque_limit
            )
        
        return validated
    
    async def _position_control(self, target_position: float):
        """Position control mode
        
        Args:
            target_position: Target position in radians
        """
        # Simplified PID control
        error = target_position - self.current_position
        
        # Would implement full PID controller
        control_output = self.position_gains['kp'] * error
        
        # Send to motor driver
        await self._send_control_output(control_output)
    
    async def _velocity_control(self, target_velocity: float):
        """Velocity control mode
        
        Args:
            target_velocity: Target velocity in rad/s
        """
        # Simplified velocity control
        error = target_velocity - self.current_velocity
        
        control_output = self.velocity_gains['kp'] * error
        
        await self._send_control_output(control_output)
    
    async def _torque_control(self, target_torque: float):
        """Torque control mode
        
        Args:
            target_torque: Target torque in Nm
        """
        # Convert torque to current
        target_current = target_torque / self.gear_ratio  # Simplified
        
        await self._send_control_output(target_current)
    
    async def _pwm_control(self, duty_cycle: float):
        """Direct PWM control
        
        Args:
            duty_cycle: PWM duty cycle (-1 to 1)
        """
        # Send PWM command to driver
        await self._send_control_output(duty_cycle)
    
    async def _send_control_output(self, output: float):
        """Send control output to motor driver
        
        Args:
            output: Control output value
        """
        # In real implementation, would send via communication interface
        pass
    
    def set_control_gains(
        self,
        mode: ControlMode,
        kp: float,
        ki: float,
        kd: float
    ):
        """Set control gains
        
        Args:
            mode: Control mode
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        if mode == ControlMode.POSITION:
            self.position_gains = {'kp': kp, 'ki': ki, 'kd': kd}
        elif mode == ControlMode.VELOCITY:
            self.velocity_gains = {'kp': kp, 'ki': ki, 'kd': kd}


class ServoController(ActuatorInterface):
    """Controller for servo motors"""
    
    def __init__(self, actuator_id: str):
        """Initialize servo controller
        
        Args:
            actuator_id: Servo identifier
        """
        super().__init__(actuator_id, ActuatorType.SERVO)
        
        # Servo parameters
        self.pwm_min = 1000  # microseconds
        self.pwm_max = 2000  # microseconds
        self.angle_range = 180  # degrees
        self.current_angle = 90  # degrees
        
        logger.info(f"Initialized ServoController for {actuator_id}")
    
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to servo
        
        Args:
            config: Servo configuration
            
        Returns:
            Success status
        """
        try:
            # In real implementation, would initialize PWM
            self.pwm_min = config.get('pwm_min', 1000)
            self.pwm_max = config.get('pwm_max', 2000)
            self.angle_range = config.get('angle_range', 180)
            
            # Set position limits in radians
            self.position_limits = (
                0,
                np.radians(self.angle_range)
            )
            
            self.is_connected = True
            logger.info(f"Connected to servo {self.actuator_id}")
            return True
            
        except Exception as e:
            logger.error(f"Servo connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from servo"""
        self.is_connected = False
        logger.info("Disconnected from servo")
    
    async def enable(self) -> bool:
        """Enable servo"""
        self.is_enabled = True
        return True
    
    async def disable(self) -> bool:
        """Disable servo"""
        self.is_enabled = False
        return True
    
    async def send_command(self, command: ActuatorCommand) -> bool:
        """Send servo command
        
        Args:
            command: Servo command
            
        Returns:
            Success status
        """
        if not self.is_connected:
            return False
        
        try:
            if command.mode == ControlMode.POSITION:
                # Convert radians to degrees
                angle_deg = np.degrees(command.target)
                
                # Clamp to range
                angle_deg = np.clip(angle_deg, 0, self.angle_range)
                
                # Convert to PWM
                pwm_value = self._angle_to_pwm(angle_deg)
                
                # Send PWM command
                await self._set_pwm(pwm_value)
                
                self.current_angle = angle_deg
                
            return True
            
        except Exception as e:
            logger.error(f"Servo command failed: {e}")
            return False
    
    async def get_status(self) -> ActuatorStatus:
        """Get servo status"""
        return ActuatorStatus(
            actuator_id=self.actuator_id,
            actuator_type=self.actuator_type,
            position=np.radians(self.current_angle),
            velocity=0.0,  # Servos typically don't report velocity
            torque=0.0,
            current=0.0,
            temperature=25.0,
            enabled=self.is_enabled,
            timestamp=time.time()
        )
    
    async def home(self) -> bool:
        """Home servo to center position"""
        center_position = np.radians(self.angle_range / 2)
        command = ActuatorCommand(
            mode=ControlMode.POSITION,
            target=center_position
        )
        return await self.send_command(command)
    
    def _angle_to_pwm(self, angle_deg: float) -> int:
        """Convert angle to PWM microseconds
        
        Args:
            angle_deg: Angle in degrees
            
        Returns:
            PWM value in microseconds
        """
        pwm_range = self.pwm_max - self.pwm_min
        pwm_per_degree = pwm_range / self.angle_range
        
        return int(self.pwm_min + angle_deg * pwm_per_degree)
    
    async def _set_pwm(self, pwm_us: int):
        """Set PWM value
        
        Args:
            pwm_us: PWM in microseconds
        """
        # In real implementation, would set PWM pin
        pass


class GimbalController:
    """Controller for multi-axis gimbals"""
    
    def __init__(self, gimbal_id: str, num_axes: int = 2):
        """Initialize gimbal controller
        
        Args:
            gimbal_id: Gimbal identifier
            num_axes: Number of gimbal axes
        """
        self.gimbal_id = gimbal_id
        self.num_axes = num_axes
        
        # Create servo controllers for each axis
        self.axes = {}
        self.axes['pitch'] = ServoController(f"{gimbal_id}_pitch")
        if num_axes >= 2:
            self.axes['roll'] = ServoController(f"{gimbal_id}_roll")
        if num_axes >= 3:
            self.axes['yaw'] = ServoController(f"{gimbal_id}_yaw")
        
        # Gimbal limits
        self.pitch_limits = (-45, 45)  # degrees
        self.roll_limits = (-45, 45)
        self.yaw_limits = (-180, 180)
        
        # Stabilization
        self.stabilization_enabled = False
        self.imu_data = None
        
        logger.info(f"Initialized {num_axes}-axis GimbalController for {gimbal_id}")
    
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to gimbal
        
        Args:
            config: Gimbal configuration
            
        Returns:
            Success status
        """
        success = True
        
        # Connect each axis
        for axis_name, controller in self.axes.items():
            axis_config = config.get(axis_name, {})
            if not await controller.connect(axis_config):
                success = False
                logger.error(f"Failed to connect {axis_name} axis")
        
        return success
    
    async def disconnect(self):
        """Disconnect from gimbal"""
        for controller in self.axes.values():
            await controller.disconnect()
    
    async def set_orientation(
        self,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
        yaw: Optional[float] = None
    ) -> bool:
        """Set gimbal orientation
        
        Args:
            pitch: Pitch angle in radians
            roll: Roll angle in radians
            yaw: Yaw angle in radians
            
        Returns:
            Success status
        """
        success = True
        
        if pitch is not None and 'pitch' in self.axes:
            command = ActuatorCommand(
                mode=ControlMode.POSITION,
                target=pitch
            )
            if not await self.axes['pitch'].send_command(command):
                success = False
        
        if roll is not None and 'roll' in self.axes:
            command = ActuatorCommand(
                mode=ControlMode.POSITION,
                target=roll
            )
            if not await self.axes['roll'].send_command(command):
                success = False
        
        if yaw is not None and 'yaw' in self.axes:
            command = ActuatorCommand(
                mode=ControlMode.POSITION,
                target=yaw
            )
            if not await self.axes['yaw'].send_command(command):
                success = False
        
        return success
    
    async def look_at(self, target_vector: np.ndarray) -> bool:
        """Point gimbal at target direction
        
        Args:
            target_vector: Target direction vector [x, y, z]
            
        Returns:
            Success status
        """
        # Calculate required angles
        target_norm = target_vector / np.linalg.norm(target_vector)
        
        pitch = np.arcsin(-target_norm[2])
        yaw = np.arctan2(target_norm[1], target_norm[0])
        
        return await self.set_orientation(pitch=pitch, yaw=yaw)
    
    def enable_stabilization(self, imu_callback: Callable):
        """Enable gimbal stabilization
        
        Args:
            imu_callback: Callback to get IMU data
        """
        self.stabilization_enabled = True
        self.imu_callback = imu_callback
        
        # Start stabilization thread
        threading.Thread(
            target=self._stabilization_loop,
            daemon=True
        ).start()
        
        logger.info("Gimbal stabilization enabled")
    
    def disable_stabilization(self):
        """Disable gimbal stabilization"""
        self.stabilization_enabled = False
        logger.info("Gimbal stabilization disabled")
    
    def _stabilization_loop(self):
        """Stabilization control loop"""
        while self.stabilization_enabled:
            try:
                # Get IMU data
                imu_data = self.imu_callback()
                
                if imu_data:
                    # Simple stabilization - compensate for platform motion
                    # In real implementation, would use proper control algorithm
                    
                    # Extract platform orientation
                    roll, pitch, _ = self._quaternion_to_euler(
                        imu_data.orientation
                    )
                    
                    # Compensate
                    asyncio.run(self.set_orientation(
                        pitch=-pitch,
                        roll=-roll
                    ))
                
                time.sleep(0.01)  # 100Hz stabilization
                
            except Exception as e:
                logger.error(f"Stabilization error: {e}")
    
    def _quaternion_to_euler(self, q: np.ndarray) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            Roll, pitch, yaw in radians
        """
        w, x, y, z = q
        
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(2 * (w * y - z * x))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        
        return roll, pitch, yaw


class ActuatorManager:
    """Manages multiple actuators"""
    
    def __init__(self):
        """Initialize actuator manager"""
        self.actuators: Dict[str, ActuatorInterface] = {}
        self.groups: Dict[str, List[str]] = {}
        
        logger.info("Initialized ActuatorManager")
    
    def add_actuator(self, actuator: ActuatorInterface):
        """Add actuator to manager
        
        Args:
            actuator: Actuator interface
        """
        self.actuators[actuator.actuator_id] = actuator
        logger.info(f"Added actuator {actuator.actuator_id}")
    
    def create_group(self, group_name: str, actuator_ids: List[str]):
        """Create actuator group
        
        Args:
            group_name: Name of group
            actuator_ids: List of actuator IDs
        """
        self.groups[group_name] = actuator_ids
        logger.info(f"Created actuator group {group_name}")
    
    async def send_group_command(
        self,
        group_name: str,
        command: ActuatorCommand
    ) -> bool:
        """Send command to actuator group
        
        Args:
            group_name: Group name
            command: Command to send
            
        Returns:
            Success status
        """
        if group_name not in self.groups:
            logger.error(f"Unknown group {group_name}")
            return False
        
        success = True
        
        for actuator_id in self.groups[group_name]:
            if actuator_id in self.actuators:
                if not await self.actuators[actuator_id].send_command(command):
                    success = False
        
        return success
    
    async def enable_all(self) -> bool:
        """Enable all actuators
        
        Returns:
            Success status
        """
        success = True
        
        for actuator in self.actuators.values():
            if not await actuator.enable():
                success = False
        
        return success
    
    async def disable_all(self) -> bool:
        """Disable all actuators
        
        Returns:
            Success status
        """
        success = True
        
        for actuator in self.actuators.values():
            if not await actuator.disable():
                success = False
        
        return success
    
    async def emergency_stop(self):
        """Emergency stop all actuators"""
        logger.warning("Emergency stop activated")
        
        # Disable all actuators
        await self.disable_all()
        
        # Send zero commands
        zero_command = ActuatorCommand(
            mode=ControlMode.VELOCITY,
            target=0.0
        )
        
        for actuator in self.actuators.values():
            try:
                await actuator.send_command(zero_command)
            except Exception as e:
                logger.error(f"Failed to stop {actuator.actuator_id}: {e}")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary of all actuators
        
        Returns:
            Status summary
        """
        summary = {
            'total_actuators': len(self.actuators),
            'enabled_count': 0,
            'fault_count': 0,
            'actuators': {}
        }
        
        for actuator_id, actuator in self.actuators.items():
            try:
                status = asyncio.run(actuator.get_status())
                
                summary['actuators'][actuator_id] = {
                    'type': actuator.actuator_type.value,
                    'enabled': status.enabled,
                    'position': status.position,
                    'temperature': status.temperature,
                    'fault': status.fault
                }
                
                if status.enabled:
                    summary['enabled_count'] += 1
                
                if status.fault:
                    summary['fault_count'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to get status for {actuator_id}: {e}")
        
        return summary


# Example usage
async def example_actuator_control():
    """Example of actuator control"""
    
    # Create actuator manager
    manager = ActuatorManager()
    
    # Create motors for quadruped robot
    for i in range(4):  # 4 legs
        for j in range(3):  # 3 joints per leg
            motor = MotorController(f"leg{i}_joint{j}")
            await motor.connect({'encoder_resolution': 4096})
            manager.add_actuator(motor)
    
    # Create groups
    manager.create_group("front_legs", [
        "leg0_joint0", "leg0_joint1", "leg0_joint2",
        "leg1_joint0", "leg1_joint1", "leg1_joint2"
    ])
    
    # Enable all actuators
    await manager.enable_all()
    
    # Send group command
    stand_command = ActuatorCommand(
        mode=ControlMode.POSITION,
        target=np.radians(45)
    )
    
    await manager.send_group_command("front_legs", stand_command)
    
    # Get status
    status = manager.get_status_summary()
    logger.info(f"Actuator status: {status}")
    
    # Cleanup
    await manager.disable_all()