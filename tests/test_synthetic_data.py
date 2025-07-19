"""Tests for synthetic data generation

This module tests the real-parameter synthetic data generation system.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import h5py

from src.data.real_parameter_extractor import (
    RealParameterExtractor, 
    DroneSpecifications,
    BatterySpecifications,
    CommunicationSpecifications
)
from src.data.physics_accurate_synthetic import (
    PhysicsAccurateSynthetic,
    SyntheticScenario,
    PhysicsState
)
from src.data.perfect_label_generator import (
    PerfectLabelGenerator,
    PhysicsConstraintLabel
)
from src.data.minimal_real_integrator import (
    MinimalRealDataIntegrator,
    RealDataSample
)
from src.data.sim_to_real_validator import (
    SimToRealValidator,
    ValidationMetrics
)
from src.data.data_utils import (
    DataConfig,
    PIHMARLDataset,
    compute_trajectory_metrics
)


class TestRealParameterExtractor:
    """Test real parameter extraction"""
    
    def test_initialization(self):
        """Test parameter extractor initialization"""
        extractor = RealParameterExtractor()
        
        # Check built-in specs loaded
        assert len(extractor.drone_specs) >= 2
        assert "dji_mavic_3" in extractor.drone_specs
        assert "parrot_anafi" in extractor.drone_specs
        
        assert len(extractor.battery_specs) >= 1
        assert "samsung_25r" in extractor.battery_specs
        
        assert len(extractor.communication_specs) >= 2
        assert "wifi_ac" in extractor.communication_specs
        assert "5g_nr" in extractor.communication_specs
    
    def test_drone_specifications(self):
        """Test drone specification retrieval"""
        extractor = RealParameterExtractor()
        
        # Get DJI Mavic 3 specs
        mavic = extractor.get_drone_specs("dji_mavic_3")
        assert mavic is not None
        assert mavic.mass == 0.895  # kg
        assert mavic.max_speed == 19.0  # m/s
        assert mavic.battery_capacity == 5000.0  # mAh
        assert mavic.flight_time == 46.0  # minutes
    
    def test_battery_voltage_calculation(self):
        """Test battery voltage calculations"""
        extractor = RealParameterExtractor()
        
        # Test voltage at different SoC
        voltage_100 = extractor.get_battery_voltage("samsung_25r", soc=1.0, c_rate=1.0)
        voltage_50 = extractor.get_battery_voltage("samsung_25r", soc=0.5, c_rate=1.0)
        voltage_10 = extractor.get_battery_voltage("samsung_25r", soc=0.1, c_rate=1.0)
        
        assert voltage_100 > voltage_50 > voltage_10
        assert voltage_100 <= 4.2  # Max voltage
        assert voltage_10 >= 3.0   # Min voltage
    
    def test_weather_conditions(self):
        """Test weather condition retrieval"""
        extractor = RealParameterExtractor()
        
        nominal = extractor.get_weather_conditions("nominal")
        assert nominal["wind_speed"] == 2.0
        assert nominal["temperature"] == 20.0
        
        extreme = extractor.get_weather_conditions("extreme")
        assert extreme["wind_speed"] == 20.0
        assert extreme["precipitation"] == 25.0


class TestPhysicsAccurateSynthetic:
    """Test physics-accurate synthetic data generation"""
    
    def test_initialization(self):
        """Test synthetic generator initialization"""
        generator = PhysicsAccurateSynthetic(render=False, gui=False)
        assert generator.parameter_extractor is not None
        assert generator.timestep == 0.01
        generator.cleanup()
    
    def test_drone_creation(self):
        """Test drone agent creation"""
        generator = PhysicsAccurateSynthetic(render=False, gui=False)
        
        # Create drone
        body_id = generator.create_drone_agent(
            agent_id=0,
            drone_type="dji_mavic_3",
            position=(0, 0, 5)
        )
        
        assert body_id >= 0
        assert 0 in generator.agents
        assert 0 in generator.agent_specs
        assert 0 in generator.battery_models
        
        # Check battery initialized at 100%
        assert generator.battery_models[0]["soc"] == 1.0
        
        generator.cleanup()
    
    def test_physics_state_computation(self):
        """Test physics state computation"""
        generator = PhysicsAccurateSynthetic(render=False, gui=False)
        
        # Create test scenario
        scenario = SyntheticScenario(
            name="test",
            description="Test scenario",
            num_agents=2,
            duration=1.0,
            timestep=0.01,
            world_size=(100, 100, 50),
            agent_types=["dji_mavic_3", "dji_mavic_3"],
            initial_positions=[(10, 10, 5), (20, 20, 5)]
        )
        
        # Create agents
        for i in range(scenario.num_agents):
            generator.create_drone_agent(
                i, 
                scenario.agent_types[i],
                scenario.initial_positions[i]
            )
        
        # Compute state
        state = generator.compute_physics_state(scenario)
        
        assert isinstance(state, PhysicsState)
        assert state.positions.shape == (2, 3)
        assert state.velocities.shape == (2, 3)
        assert state.battery_soc.shape == (2,)
        assert np.all(state.battery_soc == 1.0)  # Full battery
        assert state.collision_distances.shape == (2, 2)
        
        generator.cleanup()
    
    def test_scenario_generation(self):
        """Test complete scenario generation"""
        generator = PhysicsAccurateSynthetic(render=False, gui=False)
        
        # Simple hover scenario
        scenario = SyntheticScenario(
            name="test_hover",
            description="Test hover",
            num_agents=1,
            duration=2.0,
            timestep=0.1,
            world_size=(50, 50, 20),
            agent_types=["dji_mavic_3"],
            initial_positions=[(25, 25, 5)]
        )
        
        # Generate data
        states = generator.generate_scenario_data(scenario)
        
        assert len(states) >= 10  # At least 10 samples at 10Hz
        assert all(isinstance(s, PhysicsState) for s in states)
        
        # Check physics constraints
        for state in states:
            # Energy constraint
            assert np.all(state.battery_soc > 0)
            # Velocity constraint
            speeds = np.linalg.norm(state.velocities, axis=1)
            assert np.all(speeds < 20.0)  # Below max speed
        
        generator.cleanup()


class TestPerfectLabelGenerator:
    """Test perfect label generation"""
    
    def test_initialization(self):
        """Test label generator initialization"""
        physics_config = {
            "min_separation_distance": 2.0,
            "max_velocity": 20.0,
            "max_acceleration": 10.0,
            "communication_range": 50.0
        }
        
        generator = PerfectLabelGenerator(physics_config)
        assert generator.min_battery_soc == 0.1
        assert generator.min_separation_distance == 2.0
    
    def test_label_generation(self):
        """Test constraint label generation"""
        physics_config = {
            "min_separation_distance": 2.0,
            "max_velocity": 20.0,
            "max_acceleration": 10.0,
            "communication_range": 50.0
        }
        
        generator = PerfectLabelGenerator(physics_config)
        
        # Test data
        positions = np.array([
            [0, 0, 5],
            [10, 0, 5],
            [20, 0, 5]
        ])
        velocities = np.array([
            [5, 0, 0],
            [10, 0, 0],
            [25, 0, 0]  # Exceeds max velocity
        ])
        accelerations = np.array([
            [1, 0, 0],
            [5, 0, 0],
            [15, 0, 0]  # Exceeds max acceleration
        ])
        battery_soc = np.array([0.8, 0.5, 0.05])  # Last one below minimum
        power_consumption = np.array([100, 150, 200])
        masses = np.array([0.895, 0.895, 0.895])
        
        # Generate labels
        labels = generator.generate_labels(
            positions, velocities, accelerations,
            battery_soc, power_consumption, masses
        )
        
        assert len(labels) == 3
        
        # Check first agent (all constraints satisfied)
        assert labels[0].energy_feasible == True
        assert labels[0].collision_free == True
        assert labels[0].velocity_feasible == True
        assert labels[0].acceleration_feasible == True
        assert labels[0].fully_feasible == True
        
        # Check third agent (multiple violations)
        assert labels[2].energy_feasible == False  # Low battery
        assert labels[2].velocity_feasible == False  # Too fast
        assert labels[2].acceleration_feasible == False  # Too high acceleration
        assert labels[2].fully_feasible == False
    
    def test_physical_consistency_validation(self):
        """Test physical consistency validation"""
        physics_config = {"min_separation_distance": 2.0}
        generator = PerfectLabelGenerator(physics_config)
        
        # Create consistent trajectory
        dt = 0.01
        time = np.arange(0, 1, dt)
        positions = np.zeros((len(time), 1, 3))
        velocities = np.zeros((len(time), 1, 3))
        
        # Constant velocity motion
        velocity = 5.0  # m/s
        for t in range(len(time)):
            positions[t, 0, 0] = velocity * time[t]
            velocities[t, 0, 0] = velocity
        
        trajectory_data = {
            "positions": positions,
            "velocities": velocities
        }
        
        validation = generator.validate_physical_consistency(trajectory_data)
        assert validation["velocity_consistent"] == True
        assert validation["velocity_error"] < 0.1


class TestMinimalRealDataIntegrator:
    """Test minimal real data integration"""
    
    def test_initialization(self):
        """Test real data integrator initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            integrator = MinimalRealDataIntegrator(Path(tmpdir))
            assert integrator.data_dir.exists()
    
    def test_battery_validation_data(self):
        """Test battery validation data retrieval"""
        integrator = MinimalRealDataIntegrator()
        
        samples = integrator.get_battery_validation_data()
        assert len(samples) > 0
        
        sample = samples[0]
        assert sample.data_type == "battery"
        assert sample.battery_discharge_curves is not None
        assert "0.2C" in sample.battery_discharge_curves
        assert sample.battery_temperature_data is not None
    
    def test_communication_validation_data(self):
        """Test communication validation data retrieval"""
        integrator = MinimalRealDataIntegrator()
        
        samples = integrator.get_communication_validation_data()
        assert len(samples) > 0
        
        sample = samples[0]
        assert sample.data_type == "communication"
        assert sample.latency_measurements is not None
        assert sample.packet_loss_rates is not None
    
    def test_baseline_performance(self):
        """Test baseline performance metrics"""
        integrator = MinimalRealDataIntegrator()
        
        baselines = integrator.get_baseline_performance()
        assert "search_rescue" in baselines
        assert "formation_flight" in baselines
        assert "delivery" in baselines
        
        # Check search & rescue metrics
        sr_baseline = baselines["search_rescue"]
        assert "coverage_efficiency" in sr_baseline
        assert "detection_accuracy" in sr_baseline
        assert sr_baseline["coverage_efficiency"] > 0.5


class TestSimToRealValidator:
    """Test sim-to-real validation"""
    
    def test_initialization(self):
        """Test validator initialization"""
        validator = SimToRealValidator()
        assert validator.performance_threshold == 0.9
        assert validator.physics_compliance_threshold == 0.95
    
    def test_validation_metrics(self):
        """Test validation metric computation"""
        validator = SimToRealValidator()
        
        # Create synthetic results
        synthetic_results = {
            "mission_success": np.array([1, 1, 1, 1, 0]),  # 80% success
            "battery_soc": np.array([[0.8, 0.7], [0.6, 0.5], [0.4, 0.3]]),
            "min_separation": np.array([5.0, 4.0, 3.0, 2.5, 1.5]),
            "velocity": np.random.randn(10, 2, 3) * 5,
            "positions": np.random.randn(10, 2, 3) * 10,
            "power_consumption": np.random.uniform(100, 200, (10, 2))
        }
        
        # Create real results (slightly different)
        real_results = {
            "mission_success": np.array([1, 1, 1, 0, 0]),  # 60% success
            "battery_soc": np.array([[0.75, 0.65], [0.55, 0.45], [0.35, 0.25]]),
            "min_separation": np.array([4.5, 3.5, 2.5, 2.0, 1.0]),
            "velocity": np.random.randn(10, 2, 3) * 6,
            "positions": np.random.randn(10, 2, 3) * 11,
            "power_consumption": np.random.uniform(110, 210, (10, 2))
        }
        
        # Validate
        metrics = validator.validate_transfer(synthetic_results, real_results)
        
        assert isinstance(metrics, ValidationMetrics)
        assert 0 <= metrics.task_success_rate <= 1
        assert metrics.performance_gap >= 0
        assert 0 <= metrics.constraint_violation_rate <= 1
        assert metrics.min_separation_maintained >= 0
    
    def test_validation_report_generation(self):
        """Test validation report generation"""
        validator = SimToRealValidator()
        
        # Create sample metrics
        metrics = ValidationMetrics(
            task_success_rate=0.92,
            performance_gap=0.05,
            constraint_violation_rate=0.03,
            physics_realism_score=0.95,
            energy_prediction_error=0.08,
            endurance_prediction_error=2.5,
            control_smoothness=0.88,
            response_accuracy=0.91,
            safety_violation_count=0,
            min_separation_maintained=2.1,
            adaptation_iterations=50,
            domain_shift_magnitude=0.15,
            confidence_interval=(0.88, 0.96),
            p_value=0.12
        )
        
        report = validator.generate_validation_report(metrics)
        
        assert "PASSED" in report
        assert "Task Success Rate: 92.00%" in report
        assert "No safety violations" in report


class TestDataUtils:
    """Test data utility functions"""
    
    def test_data_config(self):
        """Test data configuration"""
        config = DataConfig()
        assert config.batch_size == 32
        assert config.sequence_length == 100
        assert config.train_split == 0.8
    
    def test_dataset_initialization(self):
        """Test PI-HMARL dataset initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy data file
            data_path = Path(tmpdir)
            dummy_file = data_path / "test.h5"
            
            with h5py.File(dummy_file, 'w') as f:
                # Create minimal structure
                traj = f.create_group('trajectories')
                traj.create_dataset('positions', data=np.zeros((100, 2, 3)))
                traj.create_dataset('velocities', data=np.zeros((100, 2, 3)))
            
            # Create dataset
            config = DataConfig(sequence_length=10, stride=5)
            dataset = PIHMARLDataset(data_path, config, split="train")
            
            assert len(dataset) > 0
            assert len(dataset.split_files) == 1
    
    def test_trajectory_metrics(self):
        """Test trajectory metric computation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test trajectory
            test_file = Path(tmpdir) / "test_traj.h5"
            
            with h5py.File(test_file, 'w') as f:
                traj = f.create_group('trajectories')
                
                # Simple circular motion
                t = np.linspace(0, 2*np.pi, 100)
                positions = np.stack([
                    10 * np.cos(t),
                    10 * np.sin(t),
                    5 * np.ones_like(t)
                ], axis=-1)
                positions = positions[:, np.newaxis, :]  # Add agent dimension
                
                traj.create_dataset('positions', data=positions)
                
                # Energy data
                energy = f.create_group('energy')
                battery_soc = np.linspace(1.0, 0.5, 100)[:, np.newaxis]
                energy.create_dataset('battery_soc', data=battery_soc)
            
            # Compute metrics
            metrics = compute_trajectory_metrics(test_file)
            
            assert 'duration' in metrics
            assert 'num_agents' in metrics
            assert metrics['num_agents'] == 1
            assert metrics['avg_trajectory_length'] > 0
            assert metrics['min_battery_soc'] == 0.5


def test_integration():
    """Test integration between components"""
    # Create parameter extractor
    param_extractor = RealParameterExtractor()
    
    # Create synthetic generator
    generator = PhysicsAccurateSynthetic(param_extractor, render=False)
    
    # Create simple scenario
    scenario = SyntheticScenario(
        name="integration_test",
        description="Integration test",
        num_agents=2,
        duration=1.0,
        timestep=0.1,
        world_size=(50, 50, 20),
        agent_types=["dji_mavic_3", "parrot_anafi"],
        initial_positions=[(10, 10, 5), (20, 20, 5)]
    )
    
    # Generate data
    states = generator.generate_scenario_data(scenario)
    assert len(states) > 0
    
    # Generate labels
    physics_config = {
        "min_separation_distance": 2.0,
        "max_velocity": 20.0,
        "max_acceleration": 10.0,
        "communication_range": 50.0
    }
    label_generator = PerfectLabelGenerator(physics_config)
    
    # Get first state
    state = states[0]
    labels = label_generator.generate_labels(
        state.positions,
        state.velocities,
        state.accelerations,
        state.battery_soc,
        state.power_consumption,
        np.array([param_extractor.get_drone_specs(t).mass for t in scenario.agent_types])
    )
    
    assert len(labels) == scenario.num_agents
    
    generator.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])