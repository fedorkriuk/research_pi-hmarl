"""Tests for Physics Module

This module tests the physics engine implementation including
vehicle dynamics, battery models, and environmental factors.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from physics import (
    PyBulletEngine,
    VehicleDynamics,
    DroneParameters,
    BatteryModel,
    BatteryParameters,
    CollisionDetector,
    CollisionShape,
    EnvironmentalFactors,
    WindField,
    WeatherConditions,
    PhysicsValidator,
    quaternion_to_euler,
    euler_to_quaternion,
    calculate_drag_force,
    calculate_thrust_from_motors
)


class TestPyBulletEngine:
    """Test PyBullet physics engine"""
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        engine = PyBulletEngine(timestep=0.01, use_gui=False)
        
        assert engine.timestep == 0.01
        assert engine.enable_collision == True
        assert engine.enable_aerodynamics == True
        
        info = engine.get_simulation_info()
        assert info["timestep"] == 0.01
        assert info["num_objects"] == 0
        
        # Cleanup
        del engine
    
    def test_object_management(self):
        """Test adding and removing objects"""
        engine = PyBulletEngine(use_gui=False)
        
        # Add object
        success = engine.add_object(
            object_id=1,
            position=np.array([0, 0, 1]),
            orientation=np.array([0, 0, 0, 1]),
            mass=1.0,
            inertia=np.array([0.1, 0.1, 0.1]),
            shape="sphere",
            size=[0.5]
        )
        assert success
        
        # Get object state
        state = engine.get_object_state(1)
        assert np.allclose(state["position"], [0, 0, 1])
        assert np.allclose(state["orientation"], [0, 0, 0, 1])
        
        # Remove object
        success = engine.remove_object(1)
        assert success
        
        # Cleanup
        del engine
    
    def test_physics_simulation(self):
        """Test physics simulation step"""
        engine = PyBulletEngine(use_gui=False)
        
        # Add falling object
        engine.add_object(
            object_id=1,
            position=np.array([0, 0, 5]),
            orientation=np.array([0, 0, 0, 1]),
            mass=1.0,
            inertia=np.array([0.1, 0.1, 0.1]),
            shape="sphere",
            size=[0.5]
        )
        
        # Simulate for 1 second
        initial_z = 5.0
        for _ in range(100):  # 100 steps at 0.01s each
            states = engine.step()
        
        # Check that object has fallen
        final_state = engine.get_object_state(1)
        assert final_state["position"][2] < initial_z
        
        # Cleanup
        del engine


class TestVehicleDynamics:
    """Test vehicle dynamics model"""
    
    def test_drone_parameters(self):
        """Test drone parameter definitions"""
        # DJI Mavic 3
        mavic = DroneParameters.dji_mavic_3()
        assert mavic.mass == 0.895
        assert mavic.max_velocity == 21.0
        assert mavic.battery_capacity == 77.0
        
        # Parrot Anafi
        anafi = DroneParameters.parrot_anafi()
        assert anafi.mass == 0.320
        assert anafi.max_velocity == 16.0
        assert anafi.battery_capacity == 42.0
    
    def test_vehicle_dynamics_update(self):
        """Test vehicle dynamics calculations"""
        params = DroneParameters.dji_mavic_3()
        dynamics = VehicleDynamics(params)
        
        # Set initial state
        dynamics.set_state(
            position=np.array([0, 0, 10]),
            velocity=np.array([5, 0, 0]),
            orientation=np.array([0, 0, 0, 1]),
            angular_velocity=np.array([0, 0, 0])
        )
        
        # Set hover thrust
        hover_thrust = params.mass * 9.81
        dynamics.set_control_inputs(hover_thrust, 0, 0, 0)
        
        # Update dynamics
        force, torque = dynamics.update(0.01)
        
        # Check thrust approximately balances gravity
        assert abs(force[2] - params.mass * 9.81) < 1.0
        
        # Check power consumption
        assert dynamics.current_power > 0
        assert dynamics.current_power < params.max_power
    
    def test_motor_mixing(self):
        """Test quadcopter motor mixing"""
        params = DroneParameters.dji_mavic_3()
        dynamics = VehicleDynamics(params)
        
        # Test pure thrust
        dynamics.set_control_inputs(10.0, 0, 0, 0)
        assert np.all(dynamics.motor_commands > 0)
        assert np.allclose(dynamics.motor_commands, dynamics.motor_commands[0])
        
        # Test roll command
        dynamics.set_control_inputs(10.0, 1.0, 0, 0)
        # Right motors should have more thrust
        assert dynamics.motor_commands[0] > dynamics.motor_commands[2]
    
    def test_vehicle_limits(self):
        """Test vehicle limit checking"""
        params = DroneParameters.dji_mavic_3()
        dynamics = VehicleDynamics(params)
        
        # Set excessive velocity
        dynamics.velocity = np.array([30, 0, 0])  # Exceeds max
        violations = dynamics.check_limits()
        assert violations["velocity"] == True
        
        # Set normal velocity
        dynamics.velocity = np.array([10, 0, 0])
        violations = dynamics.check_limits()
        assert violations["velocity"] == False


class TestBatteryModel:
    """Test battery model"""
    
    def test_battery_initialization(self):
        """Test battery initialization"""
        params = BatteryParameters.dji_intelligent_battery()
        battery = BatteryModel(params, initial_soc=0.8)
        
        assert battery.soc == 0.8
        assert battery.voltage > params.voltage_points[0]
        assert battery.temperature == 25.0
    
    def test_battery_discharge(self):
        """Test battery discharge"""
        params = BatteryParameters.samsung_18650()
        battery = BatteryModel(params)
        
        initial_soc = battery.soc
        initial_energy = battery.soc * params.energy_capacity
        
        # Discharge at constant power for 10 seconds
        power_draw = 30.0  # Watts
        for _ in range(1000):  # 10 seconds at 0.01s steps
            battery.update(power_draw, 0.01)
        
        # Check SOC decreased
        assert battery.soc < initial_soc
        
        # Check energy conservation (roughly)
        energy_used = (initial_soc - battery.soc) * params.energy_capacity
        expected_energy = power_draw * 10 / 3600  # Wh
        assert abs(energy_used - expected_energy) < 0.1
    
    def test_battery_temperature_effects(self):
        """Test temperature effects on battery"""
        params = BatteryParameters.dji_intelligent_battery()
        battery = BatteryModel(params, initial_temperature=0.0)  # Cold
        
        # Get effective capacity at cold temperature
        cold_capacity = battery._get_effective_capacity()
        
        # Warm up battery
        battery.temperature = 25.0
        normal_capacity = battery._get_effective_capacity()
        
        # Cold battery should have less capacity
        assert cold_capacity < normal_capacity
    
    def test_battery_safety_limits(self):
        """Test battery safety limit detection"""
        params = BatteryParameters.samsung_18650()
        battery = BatteryModel(params, initial_soc=0.15)  # Low SOC
        
        limits = battery.check_safety_limits()
        assert limits["soc_low"] == True
        assert limits["soc_critical"] == False
        
        # Drain further
        battery.soc = 0.05
        limits = battery.check_safety_limits()
        assert limits["soc_critical"] == True


class TestCollisionDetection:
    """Test collision detection system"""
    
    def test_collision_detector_initialization(self):
        """Test collision detector initialization"""
        detector = CollisionDetector(enable_spatial_hash=True)
        assert detector.enable_spatial_hash == True
        assert len(detector.objects) == 0
    
    def test_sphere_collision(self):
        """Test sphere-sphere collision"""
        detector = CollisionDetector()
        
        # Add two spheres
        detector.add_object(
            object_id=1,
            shape=CollisionShape("sphere", np.array([1.0])),
            position=np.array([0, 0, 0]),
            orientation=np.array([0, 0, 0, 1])
        )
        
        detector.add_object(
            object_id=2,
            shape=CollisionShape("sphere", np.array([1.0])),
            position=np.array([1.5, 0, 0]),  # Overlapping
            orientation=np.array([0, 0, 0, 1])
        )
        
        # Check collisions
        collisions = detector.detect_collisions()
        assert len(collisions) == 1
        assert collisions[0].penetration_depth > 0
    
    def test_no_collision(self):
        """Test no collision case"""
        detector = CollisionDetector()
        
        # Add two separated spheres
        detector.add_object(
            object_id=1,
            shape=CollisionShape("sphere", np.array([1.0])),
            position=np.array([0, 0, 0]),
            orientation=np.array([0, 0, 0, 1])
        )
        
        detector.add_object(
            object_id=2,
            shape=CollisionShape("sphere", np.array([1.0])),
            position=np.array([5, 0, 0]),  # Far apart
            orientation=np.array([0, 0, 0, 1])
        )
        
        collisions = detector.detect_collisions()
        assert len(collisions) == 0


class TestEnvironmentalFactors:
    """Test environmental factors"""
    
    def test_wind_field(self):
        """Test wind field generation"""
        wind = WindField(
            base_velocity=np.array([10, 0, 0]),
            turbulence_intensity=0.1
        )
        
        # Get wind at different positions
        wind1 = wind.get_wind_at_position(np.array([0, 0, 10]), 0.0)
        wind2 = wind.get_wind_at_position(np.array([0, 0, 20]), 0.0)
        
        # Higher altitude should have stronger wind (shear)
        assert np.linalg.norm(wind2[:2]) > np.linalg.norm(wind1[:2])
    
    def test_air_density_altitude(self):
        """Test air density calculation with altitude"""
        env = EnvironmentalFactors()
        
        # Sea level density
        density_0 = env.get_air_density(0)
        assert abs(density_0 - 1.225) < 0.1
        
        # High altitude density
        density_1000 = env.get_air_density(1000)
        assert density_1000 < density_0
    
    def test_environmental_forces(self):
        """Test environmental force calculations"""
        env = EnvironmentalFactors()
        env.set_wind_conditions(np.array([5, 0, 0]))
        
        vehicle_params = {
            "mass": 1.0,
            "drag_coefficient": 0.5,
            "reference_area": 0.1
        }
        
        forces = env.calculate_environmental_forces(
            position=np.array([0, 0, 10]),
            velocity=np.array([10, 0, 0]),
            vehicle_params=vehicle_params
        )
        
        # Should have wind drag
        assert "wind_drag" in forces
        assert forces["wind_drag"][0] < 0  # Opposing motion


class TestPhysicsValidator:
    """Test physics validation"""
    
    def test_state_validation(self):
        """Test single state validation"""
        validator = PhysicsValidator()
        
        # Valid state
        result = validator.validate_state(
            position=np.array([0, 0, 10]),
            velocity=np.array([10, 0, 0]),
            acceleration=np.array([0, 0, -9.81]),
            orientation=np.array([0, 0, 0, 1]),
            angular_velocity=np.array([0, 0, 0]),
            angular_acceleration=np.array([0, 0, 0]),
            mass=1.0,
            forces=np.array([0, 0, -9.81]),
            torques=np.array([0, 0, 0]),
            inertia=np.array([0.1, 0.1, 0.1])
        )
        
        assert result.valid == True
        assert len(result.violations) == 0
    
    def test_invalid_state(self):
        """Test invalid state detection"""
        validator = PhysicsValidator()
        
        # Excessive velocity
        result = validator.validate_state(
            position=np.array([0, 0, 10]),
            velocity=np.array([100, 0, 0]),  # Too fast
            acceleration=np.array([0, 0, 0]),
            orientation=np.array([0, 0, 0, 1]),
            angular_velocity=np.array([0, 0, 0]),
            angular_acceleration=np.array([0, 0, 0]),
            mass=1.0,
            forces=np.array([0, 0, 0]),
            torques=np.array([0, 0, 0]),
            inertia=np.array([0.1, 0.1, 0.1])
        )
        
        assert result.valid == False
        assert len(result.violations) > 0
    
    def test_collision_validation(self):
        """Test collision physics validation"""
        validator = PhysicsValidator()
        
        # Elastic collision
        result = validator.validate_collision(
            mass1=1.0,
            mass2=1.0,
            vel1_before=np.array([10, 0, 0]),
            vel2_before=np.array([-10, 0, 0]),
            vel1_after=np.array([-10, 0, 0]),
            vel2_after=np.array([10, 0, 0]),
            restitution=1.0
        )
        
        assert result.valid == True
        assert result.metrics["momentum_error"] < 0.1


class TestPhysicsUtils:
    """Test physics utility functions"""
    
    def test_quaternion_euler_conversion(self):
        """Test quaternion to Euler conversion"""
        # Identity quaternion
        q = np.array([0, 0, 0, 1])
        euler = quaternion_to_euler(q)
        assert np.allclose(euler, [0, 0, 0])
        
        # Convert back
        q_back = euler_to_quaternion(euler[0], euler[1], euler[2])
        assert np.allclose(q, q_back)
    
    def test_drag_force_calculation(self):
        """Test drag force calculation"""
        velocity = np.array([10, 0, 0])
        drag = calculate_drag_force(
            velocity=velocity,
            air_density=1.225,
            drag_coefficient=0.5,
            reference_area=0.1
        )
        
        # Drag opposes motion
        assert drag[0] < 0
        assert abs(drag[1]) < 0.001
        assert abs(drag[2]) < 0.001
    
    def test_thrust_calculation(self):
        """Test thrust from motors calculation"""
        motor_speeds = np.array([1000, 1000, 1000, 1000])  # rad/s
        motor_positions = np.array([
            [0.1, 0.1, 0],
            [-0.1, -0.1, 0],
            [-0.1, 0.1, 0],
            [0.1, -0.1, 0]
        ])
        motor_directions = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ])
        
        force, torque = calculate_thrust_from_motors(
            motor_speeds=motor_speeds,
            thrust_constant=1e-5,
            motor_positions=motor_positions,
            motor_directions=motor_directions
        )
        
        # Should produce upward thrust
        assert force[2] > 0
        # Balanced motors should produce minimal torque
        assert np.linalg.norm(torque[:2]) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])