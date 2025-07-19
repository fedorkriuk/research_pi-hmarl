"""Physics Validation System

This module validates physics constraints and ensures realistic behavior
in the simulation by checking against real-world physics limits.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of physics validation"""
    valid: bool
    violations: List[str]
    warnings: List[str]
    metrics: Dict[str, float]
    
    def __str__(self):
        status = "VALID" if self.valid else "INVALID"
        msg = f"Validation {status}"
        if self.violations:
            msg += f"\nViolations: {', '.join(self.violations)}"
        if self.warnings:
            msg += f"\nWarnings: {', '.join(self.warnings)}"
        return msg


class PhysicsValidator:
    """Validates physics simulation against real-world constraints"""
    
    def __init__(self):
        """Initialize physics validator"""
        # Physical limits
        self.max_velocity = 50.0  # m/s (180 km/h)
        self.max_acceleration = 20.0  # m/s² (~2g)
        self.max_angular_velocity = 10.0  # rad/s
        self.max_angular_acceleration = 20.0  # rad/s²
        
        # Energy limits
        self.max_power_density = 1000.0  # W/kg
        self.min_efficiency = 0.5  # Minimum system efficiency
        
        # Force limits
        self.max_thrust_to_weight = 3.0  # Maximum T/W ratio
        self.max_torque_to_inertia = 50.0  # Maximum torque/inertia ratio
        
        # Tolerances
        self.energy_tolerance = 0.01  # 1% energy conservation tolerance
        self.momentum_tolerance = 0.01  # 1% momentum conservation tolerance
        
        # History for validation
        self.state_history = []
        self.max_history_length = 100
        
        logger.info("Initialized PhysicsValidator")
    
    def validate_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        orientation: np.ndarray,
        angular_velocity: np.ndarray,
        angular_acceleration: np.ndarray,
        mass: float,
        forces: np.ndarray,
        torques: np.ndarray,
        inertia: np.ndarray
    ) -> ValidationResult:
        """Validate a single physics state
        
        Args:
            position: Position vector
            velocity: Velocity vector
            acceleration: Acceleration vector
            orientation: Orientation quaternion
            angular_velocity: Angular velocity vector
            angular_acceleration: Angular acceleration vector
            mass: Object mass
            forces: Applied forces
            torques: Applied torques
            inertia: Inertia tensor
            
        Returns:
            Validation result
        """
        violations = []
        warnings = []
        metrics = {}
        
        # Kinematic limits
        speed = np.linalg.norm(velocity)
        metrics["speed"] = speed
        if speed > self.max_velocity:
            violations.append(f"Velocity exceeds limit: {speed:.1f} > {self.max_velocity} m/s")
        
        accel_mag = np.linalg.norm(acceleration)
        metrics["acceleration_magnitude"] = accel_mag
        if accel_mag > self.max_acceleration:
            violations.append(f"Acceleration exceeds limit: {accel_mag:.1f} > {self.max_acceleration} m/s²")
        
        angular_speed = np.linalg.norm(angular_velocity)
        metrics["angular_speed"] = angular_speed
        if angular_speed > self.max_angular_velocity:
            violations.append(f"Angular velocity exceeds limit: {angular_speed:.1f} > {self.max_angular_velocity} rad/s")
        
        angular_accel_mag = np.linalg.norm(angular_acceleration)
        metrics["angular_acceleration_magnitude"] = angular_accel_mag
        if angular_accel_mag > self.max_angular_acceleration:
            violations.append(f"Angular acceleration exceeds limit: {angular_accel_mag:.1f} > {self.max_angular_acceleration} rad/s²")
        
        # Newton's second law validation
        expected_acceleration = forces / mass
        accel_error = np.linalg.norm(acceleration - expected_acceleration)
        metrics["newton_2nd_law_error"] = accel_error
        if accel_error > 0.1:
            warnings.append(f"Newton's 2nd law violation: error = {accel_error:.3f} m/s²")
        
        # Torque validation (simplified - assumes diagonal inertia)
        if inertia.shape == (3, 3):
            inertia_diag = np.diag(inertia)
        else:
            inertia_diag = inertia
        
        expected_angular_accel = torques / inertia_diag
        angular_accel_error = np.linalg.norm(angular_acceleration - expected_angular_accel)
        metrics["torque_equation_error"] = angular_accel_error
        if angular_accel_error > 0.1:
            warnings.append(f"Torque equation violation: error = {angular_accel_error:.3f} rad/s²")
        
        # Force limits
        force_mag = np.linalg.norm(forces)
        weight = mass * 9.81
        thrust_to_weight = force_mag / weight
        metrics["thrust_to_weight_ratio"] = thrust_to_weight
        if thrust_to_weight > self.max_thrust_to_weight:
            violations.append(f"Thrust/weight ratio exceeds limit: {thrust_to_weight:.1f} > {self.max_thrust_to_weight}")
        
        # Torque limits
        torque_mag = np.linalg.norm(torques)
        avg_inertia = np.mean(inertia_diag)
        torque_to_inertia = torque_mag / avg_inertia
        metrics["torque_to_inertia_ratio"] = torque_to_inertia
        if torque_to_inertia > self.max_torque_to_inertia:
            warnings.append(f"Torque/inertia ratio high: {torque_to_inertia:.1f}")
        
        # Quaternion validation
        quat_norm = np.linalg.norm(orientation)
        metrics["quaternion_norm"] = quat_norm
        if abs(quat_norm - 1.0) > 0.01:
            violations.append(f"Quaternion not normalized: |q| = {quat_norm:.3f}")
        
        # Determine overall validity
        valid = len(violations) == 0
        
        return ValidationResult(valid, violations, warnings, metrics)
    
    def validate_trajectory(
        self,
        states: List[Dict[str, np.ndarray]],
        dt: float
    ) -> ValidationResult:
        """Validate a trajectory for physical consistency
        
        Args:
            states: List of state dictionaries
            dt: Time step between states
            
        Returns:
            Validation result
        """
        if len(states) < 2:
            return ValidationResult(True, [], ["Insufficient states for trajectory validation"], {})
        
        violations = []
        warnings = []
        metrics = {}
        
        # Check continuity
        max_position_jump = 0.0
        max_velocity_jump = 0.0
        
        for i in range(1, len(states)):
            prev_state = states[i-1]
            curr_state = states[i]
            
            # Position continuity
            expected_pos = prev_state["position"] + prev_state["velocity"] * dt
            pos_error = np.linalg.norm(curr_state["position"] - expected_pos)
            max_position_jump = max(max_position_jump, pos_error)
            
            # Velocity continuity
            if "acceleration" in prev_state:
                expected_vel = prev_state["velocity"] + prev_state["acceleration"] * dt
                vel_error = np.linalg.norm(curr_state["velocity"] - expected_vel)
                max_velocity_jump = max(max_velocity_jump, vel_error)
        
        metrics["max_position_discontinuity"] = max_position_jump
        metrics["max_velocity_discontinuity"] = max_velocity_jump
        
        if max_position_jump > 0.1:
            violations.append(f"Position discontinuity: {max_position_jump:.3f} m")
        
        if max_velocity_jump > 1.0:
            violations.append(f"Velocity discontinuity: {max_velocity_jump:.3f} m/s")
        
        # Energy consistency check
        if self._check_energy_conservation(states, dt, metrics):
            warnings.append("Energy not conserved within tolerance")
        
        # Momentum consistency check
        if self._check_momentum_conservation(states, metrics):
            warnings.append("Momentum not conserved within tolerance")
        
        valid = len(violations) == 0
        
        return ValidationResult(valid, violations, warnings, metrics)
    
    def validate_collision(
        self,
        mass1: float,
        mass2: float,
        vel1_before: np.ndarray,
        vel2_before: np.ndarray,
        vel1_after: np.ndarray,
        vel2_after: np.ndarray,
        restitution: float = 0.8
    ) -> ValidationResult:
        """Validate collision physics
        
        Args:
            mass1, mass2: Object masses
            vel1_before, vel2_before: Velocities before collision
            vel1_after, vel2_after: Velocities after collision
            restitution: Coefficient of restitution
            
        Returns:
            Validation result
        """
        violations = []
        warnings = []
        metrics = {}
        
        # Momentum conservation
        momentum_before = mass1 * vel1_before + mass2 * vel2_before
        momentum_after = mass1 * vel1_after + mass2 * vel2_after
        momentum_error = np.linalg.norm(momentum_after - momentum_before)
        
        metrics["momentum_error"] = momentum_error
        if momentum_error > 0.1 * np.linalg.norm(momentum_before):
            violations.append(f"Momentum not conserved: error = {momentum_error:.3f} kg⋅m/s")
        
        # Energy check
        ke_before = 0.5 * mass1 * np.dot(vel1_before, vel1_before) + \
                   0.5 * mass2 * np.dot(vel2_before, vel2_before)
        ke_after = 0.5 * mass1 * np.dot(vel1_after, vel1_after) + \
                  0.5 * mass2 * np.dot(vel2_after, vel2_after)
        
        metrics["kinetic_energy_before"] = ke_before
        metrics["kinetic_energy_after"] = ke_after
        metrics["energy_ratio"] = ke_after / ke_before if ke_before > 0 else 1.0
        
        # Energy should decrease (or stay same for elastic collision)
        if ke_after > ke_before * 1.01:
            violations.append(f"Energy increased in collision: {ke_after:.1f} > {ke_before:.1f} J")
        
        # Check restitution
        relative_vel_before = vel1_before - vel2_before
        relative_vel_after = vel1_after - vel2_after
        
        # For head-on collision, check coefficient of restitution
        if np.dot(relative_vel_before, relative_vel_after) < 0:
            actual_restitution = -np.dot(relative_vel_after, relative_vel_before) / \
                               np.dot(relative_vel_before, relative_vel_before)
            metrics["actual_restitution"] = actual_restitution
            
            if actual_restitution > restitution * 1.1:
                warnings.append(f"Restitution exceeds expected: {actual_restitution:.2f} > {restitution}")
        
        valid = len(violations) == 0
        
        return ValidationResult(valid, violations, warnings, metrics)
    
    def validate_motor_physics(
        self,
        motor_speeds: np.ndarray,
        motor_commands: np.ndarray,
        thrust_produced: float,
        power_consumed: float,
        motor_params: Dict[str, float]
    ) -> ValidationResult:
        """Validate motor and propeller physics
        
        Args:
            motor_speeds: Motor speeds (rad/s)
            motor_commands: Motor commands (0-1)
            thrust_produced: Total thrust (N)
            power_consumed: Power consumption (W)
            motor_params: Motor parameters
            
        Returns:
            Validation result
        """
        violations = []
        warnings = []
        metrics = {}
        
        # Check motor speed limits
        max_rpm = motor_params.get("max_rpm", 20000)
        max_speed_rad = max_rpm * 2 * np.pi / 60
        
        for i, speed in enumerate(motor_speeds):
            if speed > max_speed_rad:
                violations.append(f"Motor {i} exceeds max speed: {speed:.0f} > {max_speed_rad:.0f} rad/s")
        
        # Thrust calculation check
        thrust_constant = motor_params.get("thrust_constant", 8.54858e-06)
        expected_thrust = thrust_constant * np.sum(motor_speeds**2)
        thrust_error = abs(thrust_produced - expected_thrust) / expected_thrust
        
        metrics["thrust_error"] = thrust_error
        if thrust_error > 0.1:
            warnings.append(f"Thrust calculation error: {thrust_error:.1%}")
        
        # Power check
        # Mechanical power
        mech_power = thrust_constant * np.sum(motor_speeds**3)
        
        # Electrical power (with efficiency)
        efficiency = motor_params.get("efficiency", 0.8)
        expected_power = mech_power / efficiency
        power_error = abs(power_consumed - expected_power) / expected_power
        
        metrics["power_error"] = power_error
        metrics["efficiency"] = mech_power / power_consumed if power_consumed > 0 else 0
        
        if power_error > 0.2:
            warnings.append(f"Power calculation error: {power_error:.1%}")
        
        valid = len(violations) == 0
        
        return ValidationResult(valid, violations, warnings, metrics)
    
    def _check_energy_conservation(
        self,
        states: List[Dict[str, np.ndarray]],
        dt: float,
        metrics: Dict[str, float]
    ) -> bool:
        """Check energy conservation in trajectory"""
        if len(states) < 2:
            return False
        
        # Calculate total energy at start and end
        initial_state = states[0]
        final_state = states[-1]
        
        # Kinetic energy
        if "mass" in initial_state:
            mass = initial_state["mass"]
            ke_initial = 0.5 * mass * np.dot(initial_state["velocity"], initial_state["velocity"])
            ke_final = 0.5 * mass * np.dot(final_state["velocity"], final_state["velocity"])
            
            # Potential energy (gravitational)
            pe_initial = mass * 9.81 * initial_state["position"][2]
            pe_final = mass * 9.81 * final_state["position"][2]
            
            total_initial = ke_initial + pe_initial
            total_final = ke_final + pe_final
            
            # Account for work done (if forces provided)
            work_done = 0.0
            if "forces" in states[0]:
                for i in range(1, len(states)):
                    displacement = states[i]["position"] - states[i-1]["position"]
                    avg_force = (states[i]["forces"] + states[i-1]["forces"]) / 2
                    work_done += np.dot(avg_force, displacement)
            
            expected_final = total_initial + work_done
            energy_error = abs(total_final - expected_final) / total_initial
            
            metrics["energy_conservation_error"] = energy_error
            
            return energy_error > self.energy_tolerance
        
        return False
    
    def _check_momentum_conservation(
        self,
        states: List[Dict[str, np.ndarray]],
        metrics: Dict[str, float]
    ) -> bool:
        """Check momentum conservation in trajectory"""
        if len(states) < 2 or "mass" not in states[0]:
            return False
        
        # Calculate total momentum at start and end
        initial_state = states[0]
        final_state = states[-1]
        
        mass = initial_state["mass"]
        momentum_initial = mass * initial_state["velocity"]
        momentum_final = mass * final_state["velocity"]
        
        # Account for impulses (if forces provided)
        total_impulse = np.zeros(3)
        if "forces" in states[0]:
            dt = 0.01  # Assumed timestep
            for state in states:
                total_impulse += state["forces"] * dt
        
        expected_momentum_final = momentum_initial + total_impulse
        momentum_error = np.linalg.norm(momentum_final - expected_momentum_final) / \
                        np.linalg.norm(momentum_initial)
        
        metrics["momentum_conservation_error"] = momentum_error
        
        return momentum_error > self.momentum_tolerance