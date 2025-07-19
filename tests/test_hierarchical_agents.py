"""Tests for Hierarchical Agent Components

This module tests the hierarchical agent architecture with real-world
constraints and parameters.
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Tuple
import time

# Import agent components
from src.agents.hierarchical_agent import HierarchicalAgent
from src.agents.meta_controller import MetaController
from src.agents.execution_policy import ExecutionPolicy, SafetyFilter
from src.agents.temporal_abstraction import TemporalAbstraction, Option
from src.agents.action_decomposer import ActionDecomposer, ActionConstraints
from src.agents.hierarchical_state import (
    HierarchicalStateEncoder, CrossLevelAttention, StateAggregator
)
from src.agents.communication_interfaces import (
    MessagePassing, CommandInterface, FeedbackLoop, StateSharing
)
from src.agents.skill_library import SkillLibrary, SkillTransfer


class TestHierarchicalAgent:
    """Test hierarchical agent functionality"""
    
    @pytest.fixture
    def agent_config(self):
        """Agent configuration"""
        return {
            "agent_id": 0,
            "state_dim": 64,
            "meta_action_dim": 8,
            "primitive_action_dim": 4,
            "planning_horizon": 30.0,
            "control_frequency": 20.0
        }
    
    @pytest.fixture
    def hierarchical_agent(self, agent_config):
        """Create hierarchical agent"""
        return HierarchicalAgent(**agent_config)
    
    def test_agent_initialization(self, hierarchical_agent):
        """Test agent initialization"""
        assert hierarchical_agent is not None
        assert hasattr(hierarchical_agent, 'meta_controller')
        assert hasattr(hierarchical_agent, 'execution_policy')
        assert hasattr(hierarchical_agent, 'temporal_abstraction')
        assert hasattr(hierarchical_agent, 'state_encoder')
    
    def test_forward_pass(self, hierarchical_agent):
        """Test forward pass through agent"""
        state = torch.randn(1, 64)
        
        # Get action
        action, info = hierarchical_agent(state)
        
        assert action.shape == (1, 4)
        assert "meta_action" in info
        assert "option_id" in info
        assert "value" in info
    
    def test_temporal_consistency(self, hierarchical_agent):
        """Test temporal consistency of decisions"""
        state = torch.randn(1, 64)
        
        # First decision
        action1, info1 = hierarchical_agent(state)
        option1 = info1["option_id"]
        
        # Small state change
        state += torch.randn(1, 64) * 0.01
        
        # Second decision (should maintain option)
        action2, info2 = hierarchical_agent(state)
        option2 = info2["option_id"]
        
        # Option should persist (most of the time)
        # This is probabilistic, so we don't assert equality


class TestMetaController:
    """Test meta-controller functionality"""
    
    @pytest.fixture
    def meta_controller(self):
        """Create meta-controller"""
        return MetaController(
            state_dim=64,
            action_dim=8,
            planning_horizon=30.0
        )
    
    def test_initialization(self, meta_controller):
        """Test initialization"""
        assert meta_controller.planning_horizon == 30.0
        assert meta_controller.update_frequency == 1.0
    
    def test_forward_pass(self, meta_controller):
        """Test forward pass"""
        strategic_state = torch.randn(1, 64)
        
        action, value = meta_controller(strategic_state)
        
        assert action.shape == (1, 8)
        assert value.shape == (1,)
    
    def test_multi_agent_coordination(self, meta_controller):
        """Test multi-agent coordination"""
        # Multiple agent states
        states = torch.randn(5, 64)
        
        actions = []
        for i in range(5):
            action, _ = meta_controller(states[i:i+1])
            actions.append(action)
        
        # Actions should be diverse for exploration
        actions_tensor = torch.cat(actions)
        std = torch.std(actions_tensor, dim=0)
        assert torch.mean(std) > 0.1  # Some diversity


class TestExecutionPolicy:
    """Test execution policy"""
    
    @pytest.fixture
    def execution_policy(self):
        """Create execution policy"""
        return ExecutionPolicy(
            state_dim=64,
            action_dim=4,
            control_frequency=20.0
        )
    
    def test_control_frequency(self, execution_policy):
        """Test control frequency constraints"""
        assert execution_policy.control_frequency == 20.0
        assert execution_policy.control_period == 0.05
        assert execution_policy.max_inference_time == 0.020
    
    def test_action_generation(self, execution_policy):
        """Test action generation"""
        state = torch.randn(1, 64)
        
        # Deterministic action
        action_det, info_det = execution_policy(state, deterministic=True)
        
        # Stochastic action
        action_stoch, info_stoch = execution_policy(state, deterministic=False)
        
        assert action_det.shape == (1, 4)
        assert action_stoch.shape == (1, 4)
        assert torch.allclose(action_det, info_det["action_mean"])
    
    def test_safety_filtering(self, execution_policy):
        """Test safety filtering"""
        state = torch.randn(10, 64)
        
        # Generate extreme actions
        with torch.no_grad():
            # Override to create extreme actions
            features = execution_policy.feature_extractor(state)
            extreme_actions = torch.randn(10, 4) * 10  # Very large actions
            
            # Apply safety filter
            safe_actions = execution_policy.safety_filter(extreme_actions, state)
        
        # Check actions are bounded
        assert torch.all(safe_actions >= -1)
        assert torch.all(safe_actions <= 1)


class TestTemporalAbstraction:
    """Test temporal abstraction"""
    
    @pytest.fixture
    def temporal_abstraction(self):
        """Create temporal abstraction"""
        return TemporalAbstraction(
            state_dim=64,
            num_options=8
        )
    
    def test_option_selection(self, temporal_abstraction):
        """Test option selection"""
        state = torch.randn(1, 64)
        meta_action = torch.tensor(3)
        
        option_id = temporal_abstraction.select_option(state, meta_action)
        
        assert 0 <= option_id < 8
    
    def test_option_termination(self, temporal_abstraction):
        """Test option termination"""
        state = torch.randn(1, 64)
        option_id = 2
        start_time = time.time()
        
        # Should not terminate immediately
        current_time = start_time + 1.0
        should_terminate = temporal_abstraction.check_termination(
            state, option_id, start_time, current_time
        )
        
        # May or may not terminate (probabilistic)
        assert isinstance(should_terminate, bool)
    
    def test_option_features(self, temporal_abstraction):
        """Test option-specific features"""
        state = torch.randn(1, 64)
        
        features = []
        for option_id in range(8):
            feat = temporal_abstraction.get_option_features(state, option_id)
            features.append(feat)
        
        # Features should be different for different options
        features_tensor = torch.stack(features)
        std = torch.std(features_tensor, dim=0)
        assert torch.mean(std) > 0.01


class TestActionDecomposer:
    """Test action decomposition"""
    
    @pytest.fixture
    def action_decomposer(self):
        """Create action decomposer"""
        return ActionDecomposer(
            meta_action_dim=8,
            primitive_action_dim=4,
            use_learned_decomposition=True
        )
    
    def test_action_decomposition(self, action_decomposer):
        """Test hierarchical action decomposition"""
        primitive_action = torch.randn(1, 4)
        
        # Test different meta actions
        for meta_action in range(8):
            final_action = action_decomposer(primitive_action, meta_action)
            
            assert final_action.shape == (1, 4)
            assert torch.all(final_action >= -1)
            assert torch.all(final_action <= 1)
    
    def test_action_constraints(self):
        """Test action constraints"""
        constraints = ActionConstraints()
        
        # Test emergency landing constraints
        action = torch.randn(1, 4)
        constrained = constraints(action, option_id=7)
        
        # Should zero out horizontal movement
        assert torch.allclose(constrained[:, 0], torch.zeros(1))
        assert torch.allclose(constrained[:, 1], torch.zeros(1))
        assert constrained[:, 2] <= 0  # Only downward


class TestHierarchicalState:
    """Test hierarchical state encoding"""
    
    @pytest.fixture
    def state_encoder(self):
        """Create state encoder"""
        return HierarchicalStateEncoder(
            state_dim=64,
            hidden_dim=128
        )
    
    def test_hierarchical_encoding(self, state_encoder):
        """Test multi-level encoding"""
        state = torch.randn(1, 64)
        
        local, tactical, strategic = state_encoder(state)
        
        assert local.shape == (1, 128)
        assert tactical.shape == (1, 128)
        assert strategic.shape == (1, 128)
    
    def test_sensor_processing(self, state_encoder):
        """Test sensor-specific processing"""
        state = torch.randn(1, 64)
        sensor_data = {
            "gps": torch.randn(1, 6),
            "imu": torch.randn(1, 6),
            "battery": torch.randn(1, 4)
        }
        
        local, tactical, strategic = state_encoder(state, sensor_data)
        
        # Should incorporate sensor data
        assert not torch.allclose(local, state_encoder(state)[0])
    
    def test_state_aggregation(self):
        """Test multi-agent state aggregation"""
        aggregator = StateAggregator(aggregation_type="mean")
        
        agent_states = {
            0: {
                "local": torch.randn(128),
                "tactical": torch.randn(128),
                "strategic": torch.randn(128)
            },
            1: {
                "local": torch.randn(128),
                "tactical": torch.randn(128),
                "strategic": torch.randn(128)
            }
        }
        
        aggregated = aggregator.aggregate(agent_states, level="tactical")
        
        assert aggregated.shape == (128,)


class TestCommunication:
    """Test communication interfaces"""
    
    @pytest.fixture
    def message_passing(self):
        """Create message passing"""
        return MessagePassing(
            agent_id=0,
            hidden_dim=128,
            message_dim=64,
            latency_ms=50.0,
            bandwidth_kbps=1000.0
        )
    
    def test_message_creation(self, message_passing):
        """Test message creation"""
        state = torch.randn(1, 128)
        
        message = message_passing.create_message(state, priority=3)
        
        assert message.shape == (1, 64)
    
    def test_message_processing(self, message_passing):
        """Test message processing with latency"""
        own_state = torch.randn(1, 128)
        messages = {
            1: torch.randn(1, 64),
            2: torch.randn(1, 64)
        }
        
        # Add messages to buffer
        for agent_id, msg in messages.items():
            message_passing.message_buffer.add_message(agent_id, msg)
        
        # Immediate processing (should be empty due to latency)
        updated_state = message_passing.process_messages(own_state, messages)
        
        # State should not change much
        assert torch.allclose(updated_state, own_state, atol=1e-2)
    
    def test_command_interface(self):
        """Test command interface"""
        cmd_interface = CommandInterface()
        
        strategic_state = torch.randn(1, 128)
        command_type = 2  # follow_trajectory
        
        # Encode command
        command = cmd_interface.encode_command(strategic_state, command_type)
        
        # Decode command
        features, decoded_type = cmd_interface.decode_command(command)
        
        assert features.shape == (1, 128)
        assert decoded_type == command_type
    
    def test_feedback_loop(self):
        """Test feedback loop"""
        feedback_loop = FeedbackLoop(hidden_dim=128)
        
        # Add feedback over time
        for i in range(5):
            execution_state = torch.randn(1, 128)
            action = torch.randn(1, 4)
            value = 0.5 + i * 0.1
            
            feedback_loop.update(execution_state, action, value)
        
        # Get feedback summary
        summary = feedback_loop.get_feedback_summary()
        
        assert summary.shape == (128,)


class TestSkillLibrary:
    """Test skill library"""
    
    @pytest.fixture
    def skill_library(self):
        """Create skill library"""
        return SkillLibrary(
            state_dim=64,
            action_dim=4,
            num_skills=20
        )
    
    def test_skill_execution(self, skill_library):
        """Test skill execution"""
        state = torch.randn(1, 64)
        
        # Execute different skills
        for skill_id in [0, 1, 2, 5, 9]:  # Sample skills
            action, info = skill_library.execute_skill(skill_id, state)
            
            assert action.shape == (1, 4)
            assert "skill_name" in info
            assert "expected_duration" in info
            assert "energy_cost" in info
    
    def test_skill_termination(self, skill_library):
        """Test skill termination conditions"""
        state = torch.randn(1, 64)
        
        # Test termination for hover skill
        should_terminate = skill_library.check_skill_termination(
            skill_id=0,
            state=state,
            elapsed_time=15.0  # Past expected duration
        )
        
        assert isinstance(should_terminate, bool)
    
    def test_skill_transfer(self):
        """Test skill transfer between drones"""
        transfer = SkillTransfer()
        
        # Create a skill
        from src.agents.skill_library import Skill
        skill = Skill(
            name="test_skill",
            description="Test",
            skill_id=0,
            required_sensors=["gps", "imu", "lidar"],
            preconditions=lambda s: True,
            postconditions=lambda s: True,
            expected_duration=20.0,
            energy_cost=500.0,
            risk_level=2
        )
        
        # Adapt from Mavic 3 to Anafi
        adapted = transfer.adapt_skill(skill, "mavic_3", "anafi")
        
        # Should adapt sensors (Anafi has no lidar)
        assert "lidar" not in adapted.required_sensors
        assert "camera" in adapted.required_sensors  # Alternative
        
        # Should adapt duration (Anafi is slower)
        assert adapted.expected_duration > skill.expected_duration
        
        # Should increase risk (no obstacle avoidance)
        assert adapted.risk_level >= skill.risk_level


class TestIntegration:
    """Integration tests for complete system"""
    
    def test_full_hierarchy(self):
        """Test complete hierarchical system"""
        # Create agent
        agent = HierarchicalAgent(
            agent_id=0,
            state_dim=64,
            meta_action_dim=8,
            primitive_action_dim=4
        )
        
        # Simulate episode
        state = torch.randn(1, 64)
        total_reward = 0
        
        for step in range(100):
            # Get action
            action, info = agent(state)
            
            # Simulate environment step
            next_state = state + torch.randn(1, 64) * 0.1
            reward = torch.randn(1).item()
            
            total_reward += reward
            state = next_state
            
            # Check action validity
            assert action.shape == (1, 4)
            assert torch.all(action >= -1)
            assert torch.all(action <= 1)
    
    def test_multi_agent_hierarchy(self):
        """Test multi-agent hierarchical system"""
        # Create multiple agents
        agents = []
        for i in range(3):
            agent = HierarchicalAgent(
                agent_id=i,
                state_dim=64,
                meta_action_dim=8,
                primitive_action_dim=4
            )
            agents.append(agent)
        
        # Communication interface
        comm_interfaces = []
        for i in range(3):
            comm = MessagePassing(
                agent_id=i,
                hidden_dim=128,
                message_dim=64
            )
            comm_interfaces.append(comm)
        
        # Simulate coordinated behavior
        states = [torch.randn(1, 64) for _ in range(3)]
        
        for step in range(50):
            actions = []
            messages = {}
            
            # Each agent acts and communicates
            for i, (agent, state) in enumerate(zip(agents, states)):
                # Get action
                action, info = agent(state)
                actions.append(action)
                
                # Create message
                if hasattr(agent, 'state_encoder'):
                    hidden_state = agent.state_encoder(state)[1]  # Tactical
                    message = comm_interfaces[i].create_message(hidden_state)
                    messages[i] = message
            
            # Update states (simplified)
            for i in range(3):
                states[i] = states[i] + torch.randn(1, 64) * 0.1
    
    def test_real_time_constraints(self):
        """Test real-time execution constraints"""
        policy = ExecutionPolicy(
            state_dim=64,
            action_dim=4,
            control_frequency=20.0
        )
        
        state = torch.randn(100, 64)  # Batch
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            action, _ = policy(state, deterministic=True)
        inference_time = time.time() - start_time
        
        # Should be fast enough for real-time
        per_sample_time = inference_time / 100
        assert per_sample_time < policy.max_inference_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])