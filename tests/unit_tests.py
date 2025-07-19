"""Unit Tests for PI-HMARL Components

This module implements unit tests for individual components
of the PI-HMARL system.
"""

import torch
import numpy as np
from typing import Dict, List, Any
import pytest
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.physics_engine.physics_integration import PhysicsEngine, PhysicsConfig
from src.physics_engine.aerodynamics import AerodynamicsModel
from src.agents.agent import Agent, AgentConfig
from src.agents.hierarchical_agent import HierarchicalAgent
from src.communication.protocol import CommunicationProtocol, MessageType, MessagePriority
from src.communication.network_manager import NetworkManager, TopologyType
from src.task_decomposition.task_analyzer import TaskAnalyzer, TaskComplexityEstimator
from src.models.attention import MultiHeadSelfAttention, HierarchicalAttention
from src.models.pinn import PhysicsInformedNN, PINNConfig
from src.energy.energy_optimizer import EnergyOptimizer, BatteryModel

from .test_framework import TestCase, TestSuite, TestLevel, TestAssertions


class PhysicsTests:
    """Unit tests for physics components"""
    
    @staticmethod
    def create_test_suite() -> TestSuite:
        """Create physics test suite"""
        suite = TestSuite("PhysicsTests", "Unit tests for physics engine")
        
        # Add test cases
        suite.add_test(TestCase(
            name="PhysicsTests.test_drone_dynamics",
            description="Test drone dynamics calculations",
            test_func=PhysicsTests.test_drone_dynamics,
            level=TestLevel.UNIT,
            tags=["physics", "dynamics"]
        ))
        
        suite.add_test(TestCase(
            name="PhysicsTests.test_aerodynamics",
            description="Test aerodynamics model",
            test_func=PhysicsTests.test_aerodynamics,
            level=TestLevel.UNIT,
            tags=["physics", "aerodynamics"]
        ))
        
        suite.add_test(TestCase(
            name="PhysicsTests.test_collision_detection",
            description="Test collision detection",
            test_func=PhysicsTests.test_collision_detection,
            level=TestLevel.UNIT,
            tags=["physics", "collision"]
        ))
        
        suite.add_test(TestCase(
            name="PhysicsTests.test_wind_effects",
            description="Test wind effect calculations",
            test_func=PhysicsTests.test_wind_effects,
            level=TestLevel.UNIT,
            tags=["physics", "environment"]
        ))
        
        return suite
    
    @staticmethod
    def test_drone_dynamics() -> Dict[str, Any]:
        """Test drone dynamics calculations"""
        # Create physics engine
        config = PhysicsConfig(
            timestep=0.01,
            gravity=9.81,
            enable_wind=False
        )
        engine = PhysicsEngine(config)
        
        # Create test drone state
        position = torch.tensor([0.0, 0.0, 10.0])
        velocity = torch.tensor([1.0, 0.0, 0.0])
        orientation = torch.tensor([0.0, 0.0, 0.0])
        
        # Apply forces
        thrust = torch.tensor([0.0, 0.0, 10.0])
        torque = torch.tensor([0.1, 0.0, 0.0])
        
        # Step physics
        new_state = engine.step_drone_physics(
            position, velocity, orientation,
            thrust, torque, mass=1.3
        )
        
        # Verify physics
        # Check gravity effect
        expected_accel_z = (thrust[2] / 1.3) - 9.81
        actual_accel_z = (new_state['velocity'][2] - velocity[2]) / config.timestep
        
        TestAssertions.assert_near(
            actual_accel_z, expected_accel_z, tolerance=0.01,
            message="Vertical acceleration mismatch"
        )
        
        # Check position update
        expected_pos_x = position[0] + velocity[0] * config.timestep
        TestAssertions.assert_near(
            new_state['position'][0].item(), expected_pos_x, tolerance=0.001,
            message="Position update incorrect"
        )
        
        return {
            'acceleration_error': abs(actual_accel_z - expected_accel_z),
            'position_error': abs(new_state['position'][0].item() - expected_pos_x)
        }
    
    @staticmethod
    def test_aerodynamics() -> Dict[str, Any]:
        """Test aerodynamics model"""
        # Create aerodynamics model
        aero_model = AerodynamicsModel()
        
        # Test drag force
        velocity = torch.tensor([10.0, 0.0, 0.0])
        drag = aero_model.calculate_drag(velocity, air_density=1.225)
        
        # Drag should oppose velocity
        assert drag[0] < 0, "Drag should oppose velocity"
        
        # Test lift force
        lift = aero_model.calculate_lift(velocity, angle_of_attack=0.1)
        
        # Lift should be perpendicular to velocity (simplified)
        assert lift[2] > 0, "Lift should have positive z component"
        
        # Test ground effect
        altitude = 2.0  # meters
        ground_effect = aero_model.ground_effect_factor(altitude)
        
        TestAssertions.assert_in_range(
            ground_effect, 1.0, 2.0,
            message="Ground effect factor out of range"
        )
        
        return {
            'drag_magnitude': torch.norm(drag).item(),
            'lift_magnitude': torch.norm(lift).item(),
            'ground_effect': ground_effect
        }
    
    @staticmethod
    def test_collision_detection() -> Dict[str, Any]:
        """Test collision detection"""
        config = PhysicsConfig()
        engine = PhysicsEngine(config)
        
        # Test sphere-sphere collision
        pos1 = torch.tensor([0.0, 0.0, 0.0])
        pos2 = torch.tensor([1.0, 0.0, 0.0])
        radius = 0.6
        
        collision = engine.check_collision(pos1, pos2, radius, radius)
        assert collision, "Should detect collision"
        
        # Test no collision
        pos2 = torch.tensor([2.0, 0.0, 0.0])
        collision = engine.check_collision(pos1, pos2, radius, radius)
        assert not collision, "Should not detect collision"
        
        # Test multiple agents
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 2.0, 0.0]
        ])
        
        collisions = engine.check_multi_collision(positions, radius)
        expected_collisions = 2  # First two agents collide
        
        assert len(collisions) == expected_collisions, f"Expected {expected_collisions} collisions"
        
        return {
            'collision_detected': True,
            'multi_collision_count': len(collisions)
        }
    
    @staticmethod
    def test_wind_effects() -> Dict[str, Any]:
        """Test wind effect calculations"""
        config = PhysicsConfig(enable_wind=True)
        engine = PhysicsEngine(config)
        
        # Set wind conditions
        engine.set_wind(
            velocity=torch.tensor([5.0, 0.0, 0.0]),
            turbulence_intensity=0.1
        )
        
        # Get wind at position
        position = torch.tensor([100.0, 100.0, 50.0])
        wind = engine.get_wind_at_position(position)
        
        # Check wind has expected properties
        TestAssertions.assert_near(
            wind[0].item(), 5.0, tolerance=1.0,
            message="Wind x-component incorrect"
        )
        
        # Test wind force on drone
        velocity = torch.tensor([10.0, 0.0, 0.0])
        wind_force = engine.calculate_wind_force(velocity, wind)
        
        # Force should oppose relative velocity
        relative_vel = velocity - wind
        assert torch.dot(wind_force, relative_vel) < 0, "Wind force should oppose relative velocity"
        
        return {
            'wind_magnitude': torch.norm(wind).item(),
            'wind_force_magnitude': torch.norm(wind_force).item()
        }


class CommunicationTests:
    """Unit tests for communication components"""
    
    @staticmethod
    def create_test_suite() -> TestSuite:
        """Create communication test suite"""
        suite = TestSuite("CommunicationTests", "Unit tests for communication system")
        
        suite.add_test(TestCase(
            name="CommunicationTests.test_message_creation",
            description="Test message creation and serialization",
            test_func=CommunicationTests.test_message_creation,
            level=TestLevel.UNIT,
            tags=["communication", "protocol"]
        ))
        
        suite.add_test(TestCase(
            name="CommunicationTests.test_network_topology",
            description="Test network topology management",
            test_func=CommunicationTests.test_network_topology,
            level=TestLevel.UNIT,
            tags=["communication", "network"]
        ))
        
        suite.add_test(TestCase(
            name="CommunicationTests.test_routing",
            description="Test message routing",
            test_func=CommunicationTests.test_routing,
            level=TestLevel.UNIT,
            tags=["communication", "routing"]
        ))
        
        suite.add_test(TestCase(
            name="CommunicationTests.test_bandwidth_management",
            description="Test bandwidth allocation",
            test_func=CommunicationTests.test_bandwidth_management,
            level=TestLevel.UNIT,
            tags=["communication", "qos"]
        ))
        
        return suite
    
    @staticmethod
    def test_message_creation() -> Dict[str, Any]:
        """Test message creation and serialization"""
        # Create protocol
        protocol = CommunicationProtocol(agent_id=1)
        
        # Create different message types
        messages_created = 0
        
        # Heartbeat message
        heartbeat = protocol.create_message(
            msg_type=MessageType.HEARTBEAT,
            receiver_id=None,
            payload={'status': 'active'},
            priority=MessagePriority.LOW
        )
        assert heartbeat.msg_type == MessageType.HEARTBEAT
        assert heartbeat.sender_id == 1
        messages_created += 1
        
        # Emergency message
        emergency = protocol.create_message(
            msg_type=MessageType.EMERGENCY,
            receiver_id=2,
            payload={'alert': 'collision_imminent'},
            priority=MessagePriority.CRITICAL,
            requires_ack=True
        )
        assert emergency.priority == MessagePriority.CRITICAL
        assert emergency.requires_ack
        messages_created += 1
        
        # Verify checksums
        assert len(heartbeat.checksum) == 16
        assert len(emergency.checksum) == 16
        
        return {
            'messages_created': messages_created,
            'checksum_length': len(heartbeat.checksum)
        }
    
    @staticmethod
    def test_network_topology() -> Dict[str, Any]:
        """Test network topology management"""
        # Create network manager
        manager = NetworkManager(
            agent_id=1,
            topology_type=TopologyType.MESH,
            max_range=1000.0
        )
        
        # Add agents
        positions = {
            1: torch.tensor([0.0, 0.0, 0.0]),
            2: torch.tensor([500.0, 0.0, 0.0]),
            3: torch.tensor([1000.0, 0.0, 0.0]),
            4: torch.tensor([2000.0, 0.0, 0.0])  # Out of range
        }
        
        manager.update_agent_positions(positions)
        
        # Check connectivity
        neighbors = len(manager.links)
        assert neighbors == 2, f"Expected 2 neighbors, got {neighbors}"
        
        # Update link quality
        manager.update_link_quality(2, {
            'rssi': -70.0,
            'packet_loss': 0.01,
            'latency': 0.05
        })
        
        # Calculate routes
        routes = manager.calculate_routes()
        
        # Should have route to agent 3 via agent 2
        assert 3 in routes, "Should have route to agent 3"
        assert routes[3].next_hop == 2, "Should route through agent 2"
        
        return {
            'neighbors': neighbors,
            'routes': len(routes),
            'network_diameter': 2
        }
    
    @staticmethod
    def test_routing() -> Dict[str, Any]:
        """Test message routing"""
        # Setup network
        manager = NetworkManager(agent_id=1)
        
        positions = {
            1: torch.tensor([0.0, 0.0, 0.0]),
            2: torch.tensor([500.0, 0.0, 0.0]),
            3: torch.tensor([1000.0, 0.0, 0.0])
        }
        manager.update_agent_positions(positions)
        
        # Calculate routes
        routes = manager.calculate_routes()
        
        # Test direct route
        next_hop = manager.get_next_hop(2)
        assert next_hop == 2, "Should have direct route to agent 2"
        
        # Test multi-hop route
        next_hop = manager.get_next_hop(3)
        assert next_hop == 2, "Should route to agent 3 through agent 2"
        
        # Test no route
        next_hop = manager.get_next_hop(99)
        assert next_hop is None, "Should have no route to non-existent agent"
        
        return {
            'total_routes': len(routes),
            'multi_hop_routes': sum(1 for r in routes.values() if r.hop_count > 1)
        }
    
    @staticmethod
    def test_bandwidth_management() -> Dict[str, Any]:
        """Test bandwidth allocation"""
        from src.communication.bandwidth_manager import BandwidthManager, TrafficClass
        
        # Create bandwidth manager
        bandwidth_mgr = BandwidthManager(total_bandwidth=10e6)  # 10 Mbps
        
        # Allocate bandwidth for different traffic classes
        allocations = []
        
        # Control traffic
        control_bw = bandwidth_mgr.allocate_bandwidth(
            source=1, destination=2,
            traffic_class=TrafficClass.CONTROL,
            requested_bandwidth=2e6
        )
        allocations.append(control_bw)
        
        # Real-time traffic
        realtime_bw = bandwidth_mgr.allocate_bandwidth(
            source=1, destination=3,
            traffic_class=TrafficClass.REALTIME,
            requested_bandwidth=5e6
        )
        allocations.append(realtime_bw)
        
        # Verify QoS
        assert control_bw >= 2e6, "Control traffic should get minimum bandwidth"
        assert sum(allocations) <= 10e6, "Total allocation should not exceed capacity"
        
        # Test transmission
        success, time_est = bandwidth_mgr.transmit(
            source=1, destination=2,
            data_size=1024,  # 1KB
            traffic_class=TrafficClass.CONTROL
        )
        
        assert success, "Transmission should succeed"
        assert time_est > 0, "Should have positive transmission time"
        
        return {
            'control_bandwidth': control_bw / 1e6,  # Mbps
            'realtime_bandwidth': realtime_bw / 1e6,
            'transmission_time': time_est
        }


class CoordinationTests:
    """Unit tests for coordination components"""
    
    @staticmethod
    def create_test_suite() -> TestSuite:
        """Create coordination test suite"""
        suite = TestSuite("CoordinationTests", "Unit tests for coordination system")
        
        suite.add_test(TestCase(
            name="CoordinationTests.test_task_decomposition",
            description="Test task decomposition",
            test_func=CoordinationTests.test_task_decomposition,
            level=TestLevel.UNIT,
            tags=["coordination", "tasks"]
        ))
        
        suite.add_test(TestCase(
            name="CoordinationTests.test_task_assignment",
            description="Test task assignment optimization",
            test_func=CoordinationTests.test_task_assignment,
            level=TestLevel.UNIT,
            tags=["coordination", "optimization"]
        ))
        
        suite.add_test(TestCase(
            name="CoordinationTests.test_consensus",
            description="Test consensus protocol",
            test_func=CoordinationTests.test_consensus,
            level=TestLevel.UNIT,
            tags=["coordination", "consensus"]
        ))
        
        return suite
    
    @staticmethod
    def test_task_decomposition() -> Dict[str, Any]:
        """Test task decomposition"""
        # Create task analyzer
        analyzer = TaskAnalyzer()
        estimator = TaskComplexityEstimator()
        
        # Create complex task
        task = {
            'task_id': 'surveillance_001',
            'type': 'surveillance',
            'area': [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
            'duration': 3600,  # 1 hour
            'priority': 0.8,
            'requirements': ['camera', 'long_range']
        }
        
        # Estimate complexity
        team_state = {
            'num_agents': 3,
            'avg_battery': 0.8,
            'capabilities': ['camera', 'long_range', 'communication']
        }
        
        complexity = estimator.estimate(task, team_state)
        TestAssertions.assert_in_range(
            complexity, 0.0, 1.0,
            message="Complexity should be normalized"
        )
        
        # Analyze task
        analysis = analyzer.analyze_task(task, team_state)
        
        assert analysis['needs_decomposition'], "Complex task should need decomposition"
        assert 'subtasks' in analysis, "Should generate subtasks"
        assert len(analysis['subtasks']) > 1, "Should create multiple subtasks"
        
        return {
            'complexity': complexity,
            'num_subtasks': len(analysis['subtasks']),
            'decomposition_time': analysis.get('processing_time', 0)
        }
    
    @staticmethod
    def test_task_assignment() -> Dict[str, Any]:
        """Test task assignment optimization"""
        from src.task_decomposition.assignment_optimizer import AssignmentOptimizer
        
        # Create optimizer
        optimizer = AssignmentOptimizer(algorithm='hungarian')
        
        # Create tasks and agents
        tasks = [
            {'id': 't1', 'position': [0, 0], 'priority': 0.9},
            {'id': 't2', 'position': [100, 100], 'priority': 0.7},
            {'id': 't3', 'position': [200, 0], 'priority': 0.8}
        ]
        
        agent_states = {
            1: {'position': torch.tensor([0., 0., 50.]), 'battery': 0.9},
            2: {'position': torch.tensor([150., 150., 50.]), 'battery': 0.8},
            3: {'position': torch.tensor([200., 0., 50.]), 'battery': 0.7}
        }
        
        # Optimize assignment
        assignment = optimizer.optimize_assignment(tasks, agent_states)
        
        # Verify assignment
        assert len(assignment) == 3, "All agents should be assigned"
        
        # Check optimality (agent 1 should get t1, etc.)
        total_cost = optimizer.calculate_assignment_cost(assignment, tasks, agent_states)
        
        return {
            'agents_assigned': len(assignment),
            'total_cost': total_cost,
            'algorithm': 'hungarian'
        }
    
    @staticmethod
    def test_consensus() -> Dict[str, Any]:
        """Test consensus protocol"""
        from src.communication.consensus import ConsensusProtocol, ConsensusType, VoteType
        
        # Create consensus protocol
        protocol = ConsensusProtocol(
            agent_id=1,
            total_agents=5,
            consensus_type=ConsensusType.MAJORITY
        )
        
        # Create proposal
        proposal_id = protocol.propose(
            content={'action': 'change_formation', 'formation': 'v-shape'},
            min_votes=3,
            deadline=10.0
        )
        
        assert proposal_id is not None, "Should create proposal"
        
        # Cast own vote
        protocol.vote(proposal_id, VoteType.APPROVE)
        
        # Simulate other votes
        from src.communication.consensus import Vote
        
        # Add more votes
        for agent_id in [2, 3]:
            vote = Vote(
                voter_id=agent_id,
                proposal_id=proposal_id,
                vote_type=VoteType.APPROVE,
                timestamp=time.time()
            )
            protocol.receive_vote(vote)
        
        # Check consensus status
        status = protocol.get_consensus_status(proposal_id)
        assert status is not None, "Should have consensus status"
        assert status.status == 'accepted', "Should reach consensus with 3/5 votes"
        
        return {
            'proposal_created': True,
            'consensus_reached': status.status == 'accepted',
            'votes_received': len(status.votes)
        }


class EnergyTests:
    """Unit tests for energy management"""
    
    @staticmethod
    def create_test_suite() -> TestSuite:
        """Create energy test suite"""
        suite = TestSuite("EnergyTests", "Unit tests for energy management")
        
        suite.add_test(TestCase(
            name="EnergyTests.test_battery_model",
            description="Test battery model accuracy",
            test_func=EnergyTests.test_battery_model,
            level=TestLevel.UNIT,
            tags=["energy", "battery"]
        ))
        
        suite.add_test(TestCase(
            name="EnergyTests.test_power_estimation",
            description="Test power consumption estimation",
            test_func=EnergyTests.test_power_estimation,
            level=TestLevel.UNIT,
            tags=["energy", "power"]
        ))
        
        suite.add_test(TestCase(
            name="EnergyTests.test_energy_optimization",
            description="Test energy optimization",
            test_func=EnergyTests.test_energy_optimization,
            level=TestLevel.UNIT,
            tags=["energy", "optimization"]
        ))
        
        return suite
    
    @staticmethod
    def test_battery_model() -> Dict[str, Any]:
        """Test battery model accuracy"""
        # Create battery model
        battery = BatteryModel(
            capacity=18.5,  # Wh
            voltage_nominal=11.1,  # V
            internal_resistance=0.03  # Ohms
        )
        
        # Test discharge
        initial_soc = battery.soc
        power = 20.0  # Watts
        duration = 60.0  # seconds
        
        new_soc = battery.discharge(power, duration)
        
        # Calculate expected SOC change
        energy_used = power * duration / 3600  # Wh
        expected_soc_change = energy_used / battery.capacity
        actual_soc_change = initial_soc - new_soc
        
        TestAssertions.assert_near(
            actual_soc_change, expected_soc_change, tolerance=0.01,
            message="SOC change mismatch"
        )
        
        # Test voltage curve
        voltage = battery.get_voltage()
        TestAssertions.assert_in_range(
            voltage, 9.0, 12.6,
            message="Battery voltage out of range"
        )
        
        # Test efficiency
        efficiency = battery.get_efficiency(power)
        TestAssertions.assert_in_range(
            efficiency, 0.8, 1.0,
            message="Battery efficiency out of range"
        )
        
        return {
            'soc_accuracy': abs(actual_soc_change - expected_soc_change),
            'voltage': voltage,
            'efficiency': efficiency
        }
    
    @staticmethod
    def test_power_estimation() -> Dict[str, Any]:
        """Test power consumption estimation"""
        from src.energy.power_models import PowerEstimator
        
        # Create power estimator
        estimator = PowerEstimator()
        
        # Test hover power
        mass = 1.3  # kg
        hover_power = estimator.estimate_hover_power(mass)
        
        # Theoretical hover power
        thrust = mass * 9.81
        theoretical_power = thrust ** 1.5 / (2 * 1.225 * np.pi * 0.15**2) ** 0.5
        
        TestAssertions.assert_near(
            hover_power, theoretical_power, tolerance=theoretical_power * 0.2,
            message="Hover power estimation error"
        )
        
        # Test forward flight power
        velocity = 10.0  # m/s
        flight_power = estimator.estimate_flight_power(mass, velocity)
        
        assert flight_power > hover_power, "Forward flight should use more power"
        
        # Test climbing power
        climb_rate = 2.0  # m/s
        climb_power = estimator.estimate_climb_power(mass, climb_rate)
        
        additional_power = mass * 9.81 * climb_rate
        assert climb_power >= hover_power + additional_power * 0.8, "Climb power too low"
        
        return {
            'hover_power': hover_power,
            'flight_power': flight_power,
            'climb_power': climb_power
        }
    
    @staticmethod
    def test_energy_optimization() -> Dict[str, Any]:
        """Test energy optimization"""
        # Create energy optimizer
        optimizer = EnergyOptimizer(num_agents=3)
        
        # Create mission profile
        agent_states = {
            0: {'position': torch.tensor([0., 0., 50.]), 'battery_soc': 0.9},
            1: {'position': torch.tensor([100., 0., 50.]), 'battery_soc': 0.7},
            2: {'position': torch.tensor([0., 100., 50.]), 'battery_soc': 0.8}
        }
        
        tasks = [
            {'position': torch.tensor([200., 200., 50.]), 'duration': 300},
            {'position': torch.tensor([-100., 100., 50.]), 'duration': 200}
        ]
        
        # Optimize energy allocation
        optimization_result = optimizer.optimize_mission(agent_states, tasks)
        
        assert 'trajectories' in optimization_result, "Should generate trajectories"
        assert 'energy_usage' in optimization_result, "Should estimate energy usage"
        
        # Verify feasibility
        for agent_id, energy_used in optimization_result['energy_usage'].items():
            available_energy = agent_states[agent_id]['battery_soc'] * 18.5  # Wh
            assert energy_used <= available_energy, f"Agent {agent_id} energy infeasible"
        
        return {
            'total_energy': sum(optimization_result['energy_usage'].values()),
            'feasible': True,
            'optimization_time': optimization_result.get('computation_time', 0)
        }


class TaskTests:
    """Unit tests for task management"""
    
    @staticmethod
    def create_test_suite() -> TestSuite:
        """Create task test suite"""
        suite = TestSuite("TaskTests", "Unit tests for task management")
        
        suite.add_test(TestCase(
            name="TaskTests.test_skill_matching",
            description="Test skill-based task matching",
            test_func=TaskTests.test_skill_matching,
            level=TestLevel.UNIT,
            tags=["tasks", "skills"]
        ))
        
        suite.add_test(TestCase(
            name="TaskTests.test_task_monitoring",
            description="Test task execution monitoring",
            test_func=TaskTests.test_task_monitoring,
            level=TestLevel.UNIT,
            tags=["tasks", "monitoring"]
        ))
        
        return suite
    
    @staticmethod
    def test_skill_matching() -> Dict[str, Any]:
        """Test skill-based task matching"""
        from src.task_decomposition.skill_matching import SkillMatcher, TaskRequirement, AgentCapability, SkillLevel, Skill
        
        # Create skill matcher
        matcher = SkillMatcher()
        
        # Create task requirements
        task_req = TaskRequirement(
            task_id='inspection_001',
            required_skills={
                'perception': SkillLevel.ADVANCED,
                'navigation': SkillLevel.INTERMEDIATE,
                'data_analysis': SkillLevel.INTERMEDIATE
            },
            preferred_skills={},
            skill_importance={
                'perception': 0.9,
                'navigation': 0.7,
                'data_analysis': 0.5
            },
            complexity=0.7,
            flexibility=0.3
        )
        
        # Create agent capabilities
        agent_capabilities = {
            1: AgentCapability(
                agent_id=1,
                skills={
                    'perception': Skill('perception', SkillLevel.EXPERT, 100.0, 0.95, time.time(), []),
                    'navigation': Skill('navigation', SkillLevel.ADVANCED, 80.0, 0.9, time.time(), []),
                    'data_analysis': Skill('data_analysis', SkillLevel.INTERMEDIATE, 40.0, 0.85, time.time(), [])
                },
                specializations=['perception'],
                performance_history=[0.9, 0.92, 0.88],
                learning_rate=0.1,
                adaptability=0.7
            ),
            2: AgentCapability(
                agent_id=2,
                skills={
                    'perception': Skill('perception', SkillLevel.INTERMEDIATE, 50.0, 0.8, time.time(), []),
                    'navigation': Skill('navigation', SkillLevel.INTERMEDIATE, 60.0, 0.85, time.time(), []),
                    'data_analysis': Skill('data_analysis', SkillLevel.BEGINNER, 20.0, 0.7, time.time(), [])
                },
                specializations=[],
                performance_history=[0.75, 0.78, 0.8],
                learning_rate=0.15,
                adaptability=0.6
            )
        }
        
        # Match agents to task
        matches = matcher.match_agents_to_task(task_req, agent_capabilities)
        
        assert len(matches) == 2, "Should match all agents"
        assert matches[0][0] == 1, "Agent 1 should be best match"
        assert matches[0][1] > matches[1][1], "Agent 1 should have higher score"
        
        return {
            'best_match': matches[0][0],
            'match_score': matches[0][1],
            'score_difference': matches[0][1] - matches[1][1]
        }
    
    @staticmethod
    def test_task_monitoring() -> Dict[str, Any]:
        """Test task execution monitoring"""
        from src.task_decomposition.task_monitor import TaskMonitor, TaskProgress
        
        # Create task monitor
        monitor = TaskMonitor()
        
        # Create task progress
        task_progress = TaskProgress(
            task_id='delivery_001',
            start_time=time.time() - 100,
            estimated_duration=300,
            assigned_agents=[1, 2],
            checkpoints=['pickup', 'transit', 'delivery'],
            current_checkpoint=1,
            progress_percentage=40.0
        )
        
        # Update progress
        agent_updates = {
            1: {'position': [50, 50], 'status': 'active', 'battery': 0.7},
            2: {'position': [52, 48], 'status': 'active', 'battery': 0.65}
        }
        
        update_result = monitor.update_progress('delivery_001', agent_updates)
        
        assert 'estimated_completion' in update_result, "Should estimate completion"
        assert 'performance_score' in update_result, "Should calculate performance"
        
        # Check alerts
        assert 'alerts' in update_result, "Should check for alerts"
        
        return {
            'progress': task_progress.progress_percentage,
            'estimated_completion': update_result['estimated_completion'],
            'alerts_count': len(update_result['alerts'])
        }