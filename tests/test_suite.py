"""Comprehensive Test Suite for PI-HMARL

This module provides a comprehensive test suite for the PI-HMARL system,
including unit tests, integration tests, and performance benchmarks.
"""

import unittest
import numpy as np
import torch
import time
import logging
from typing import Dict, List, Any, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PI-HMARL modules
from src.core import MultiAgentSystem, HierarchicalController
from src.models import PhysicsInformedModel, MultiAgentTransformer
from src.environments import PhysicsEnvironment, MultiAgentEnv
from src.training import PIHMARLTrainer
from src.utils import setup_logging

logger = logging.getLogger(__name__)


class TestCore(unittest.TestCase):
    """Test core components"""
    
    def setUp(self):
        """Set up test environment"""
        self.num_agents = 4
        self.state_dim = 12
        self.action_dim = 4
        self.config = {
            'num_agents': self.num_agents,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'communication_range': 30.0
        }
    
    def test_multi_agent_system_initialization(self):
        """Test MultiAgentSystem initialization"""
        mas = MultiAgentSystem(self.config)
        
        self.assertEqual(len(mas.agents), self.num_agents)
        self.assertIsNotNone(mas.communication_network)
        self.assertTrue(mas.initialized)
        
        logger.info("MultiAgentSystem initialization test passed")
    
    def test_hierarchical_controller(self):
        """Test HierarchicalController"""
        controller = HierarchicalController(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents
        )
        
        # Test high-level planning
        global_state = torch.randn(1, self.num_agents * self.state_dim)
        high_level_actions = controller.high_level_policy(global_state)
        
        self.assertEqual(high_level_actions.shape, (1, self.num_agents, self.action_dim))
        
        # Test low-level control
        local_states = torch.randn(self.num_agents, self.state_dim)
        low_level_actions = controller.low_level_policy(local_states, high_level_actions[0])
        
        self.assertEqual(low_level_actions.shape, (self.num_agents, self.action_dim))
        
        logger.info("HierarchicalController test passed")
    
    def test_agent_communication(self):
        """Test agent communication"""
        mas = MultiAgentSystem(self.config)
        
        # Create test message
        message = {
            'type': 'position_update',
            'sender': 'agent_0',
            'data': {'position': [10.0, 20.0, 5.0]}
        }
        
        # Send message
        mas.agents[0].send_message(message, broadcast=True)
        
        # Check if other agents received
        for i in range(1, self.num_agents):
            messages = mas.agents[i].receive_messages()
            self.assertTrue(len(messages) > 0)
            self.assertEqual(messages[0]['sender'], 'agent_0')
        
        logger.info("Agent communication test passed")


class TestModels(unittest.TestCase):
    """Test model components"""
    
    def setUp(self):
        """Set up test environment"""
        self.state_dim = 12
        self.action_dim = 4
        self.hidden_dim = 64
        self.num_agents = 3
    
    def test_physics_informed_model(self):
        """Test PhysicsInformedModel"""
        model = PhysicsInformedModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Test forward pass
        states = torch.randn(32, self.state_dim)
        actions, values = model(states)
        
        self.assertEqual(actions.shape, (32, self.action_dim))
        self.assertEqual(values.shape, (32, 1))
        
        # Test physics loss
        next_states = torch.randn(32, self.state_dim)
        physics_loss = model.compute_physics_loss(states, actions, next_states)
        
        self.assertIsInstance(physics_loss.item(), float)
        self.assertGreaterEqual(physics_loss.item(), 0)
        
        logger.info("PhysicsInformedModel test passed")
    
    def test_multi_agent_transformer(self):
        """Test MultiAgentTransformer"""
        model = MultiAgentTransformer(
            num_agents=self.num_agents,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Test forward pass
        agent_states = torch.randn(1, self.num_agents, self.state_dim)
        agent_actions = torch.randn(1, self.num_agents, self.action_dim)
        
        q_values = model(agent_states, agent_actions)
        
        self.assertEqual(q_values.shape, (1, self.num_agents))
        
        logger.info("MultiAgentTransformer test passed")
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        model = PhysicsInformedModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Save model
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(model.state_dict(), tmp.name)
            
            # Load model
            model2 = PhysicsInformedModel(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim
            )
            model2.load_state_dict(torch.load(tmp.name))
            
            # Compare parameters
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
            
            # Clean up
            os.unlink(tmp.name)
        
        logger.info("Model save/load test passed")


class TestEnvironments(unittest.TestCase):
    """Test environment components"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'num_agents': 3,
            'world_size': (100.0, 100.0, 50.0),
            'physics_dt': 0.01,
            'render': False
        }
    
    def test_physics_environment(self):
        """Test PhysicsEnvironment"""
        env = PhysicsEnvironment(self.config)
        
        # Test reset
        states = env.reset()
        self.assertEqual(len(states), self.config['num_agents'])
        
        # Test step
        actions = [np.random.randn(4) for _ in range(self.config['num_agents'])]
        next_states, rewards, dones, info = env.step(actions)
        
        self.assertEqual(len(next_states), self.config['num_agents'])
        self.assertEqual(len(rewards), self.config['num_agents'])
        self.assertEqual(len(dones), self.config['num_agents'])
        
        logger.info("PhysicsEnvironment test passed")
    
    def test_multi_agent_env(self):
        """Test MultiAgentEnv wrapper"""
        base_env = PhysicsEnvironment(self.config)
        env = MultiAgentEnv(base_env)
        
        # Test observation space
        obs_space = env.observation_space
        self.assertIsNotNone(obs_space)
        
        # Test action space
        action_space = env.action_space
        self.assertIsNotNone(action_space)
        
        # Test episode
        obs = env.reset()
        total_reward = 0
        
        for _ in range(100):
            actions = env.action_space.sample()
            obs, rewards, dones, info = env.step(actions)
            total_reward += sum(rewards)
            
            if all(dones):
                break
        
        self.assertIsInstance(total_reward, (int, float))
        
        logger.info("MultiAgentEnv test passed")


class TestTraining(unittest.TestCase):
    """Test training components"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'num_agents': 2,
            'state_dim': 8,
            'action_dim': 3,
            'learning_rate': 1e-3,
            'batch_size': 32,
            'physics_weight': 0.1
        }
    
    def test_trainer_initialization(self):
        """Test PIHMARLTrainer initialization"""
        trainer = PIHMARLTrainer(self.config)
        
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertEqual(trainer.config['batch_size'], 32)
        
        logger.info("PIHMARLTrainer initialization test passed")
    
    def test_training_step(self):
        """Test single training step"""
        trainer = PIHMARLTrainer(self.config)
        
        # Create dummy batch
        batch = {
            'states': torch.randn(32, self.config['state_dim']),
            'actions': torch.randn(32, self.config['action_dim']),
            'rewards': torch.randn(32, 1),
            'next_states': torch.randn(32, self.config['state_dim']),
            'dones': torch.zeros(32, 1)
        }
        
        # Perform training step
        initial_loss = trainer.train_step(batch)
        
        self.assertIsInstance(initial_loss, float)
        self.assertGreater(initial_loss, 0)
        
        logger.info("Training step test passed")
    
    def test_distributed_training(self):
        """Test distributed training setup"""
        # Note: This is a simplified test
        config = self.config.copy()
        config['distributed'] = True
        config['world_size'] = 2
        
        # Test configuration
        trainer = PIHMARLTrainer(config)
        
        self.assertTrue(trainer.config['distributed'])
        self.assertEqual(trainer.config['world_size'], 2)
        
        logger.info("Distributed training setup test passed")


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_full_system_integration(self):
        """Test full system integration"""
        # Configuration
        config = {
            'num_agents': 3,
            'state_dim': 12,
            'action_dim': 4,
            'world_size': (50.0, 50.0, 25.0),
            'episode_length': 100,
            'physics_dt': 0.01
        }
        
        # Initialize components
        mas = MultiAgentSystem(config)
        env = PhysicsEnvironment(config)
        trainer = PIHMARLTrainer(config)
        
        # Run short episode
        states = env.reset()
        total_rewards = []
        
        for step in range(10):  # Short test episode
            # Get actions from agents
            actions = []
            for i, agent in enumerate(mas.agents):
                action = agent.select_action(states[i])
                actions.append(action)
            
            # Environment step
            next_states, rewards, dones, info = env.step(actions)
            total_rewards.append(sum(rewards))
            
            # Update states
            states = next_states
            
            if all(dones):
                break
        
        # Verify results
        self.assertEqual(len(total_rewards), step + 1)
        self.assertTrue(all(isinstance(r, (int, float)) for r in total_rewards))
        
        logger.info("Full system integration test passed")
    
    def test_scenario_loading(self):
        """Test scenario loading and execution"""
        from src.scenarios import SearchRescueScenario
        
        # Create scenario
        scenario = SearchRescueScenario(
            area_size=(100.0, 100.0),
            num_victims=5,
            num_agents=3,
            obstacle_density=0.05
        )
        
        # Run for a few steps
        for _ in range(10):
            scenario.step(dt=0.1)
        
        # Check state
        state = scenario.get_state()
        
        self.assertIn('time', state)
        self.assertIn('victims', state)
        self.assertIn('agents', state)
        self.assertGreater(state['time'], 0)
        
        logger.info("Scenario loading test passed")


class PerformanceBenchmark:
    """Performance benchmarking"""
    
    def __init__(self):
        """Initialize benchmark"""
        self.results = {}
    
    def benchmark_model_inference(self, num_agents: List[int], num_iterations: int = 1000):
        """Benchmark model inference speed"""
        results = {}
        
        for n_agents in num_agents:
            model = PhysicsInformedModel(state_dim=12, action_dim=4)
            states = torch.randn(n_agents, 12)
            
            # Warmup
            for _ in range(10):
                _ = model(states)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                _ = model(states)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_iterations
            results[n_agents] = {
                'avg_inference_time': avg_time,
                'throughput': n_agents / avg_time
            }
        
        self.results['model_inference'] = results
        return results
    
    def benchmark_environment_step(self, num_agents: List[int], num_steps: int = 1000):
        """Benchmark environment step speed"""
        results = {}
        
        for n_agents in num_agents:
            config = {
                'num_agents': n_agents,
                'world_size': (100.0, 100.0, 50.0),
                'physics_dt': 0.01
            }
            
            env = PhysicsEnvironment(config)
            env.reset()
            
            # Random actions
            actions = [np.random.randn(4) for _ in range(n_agents)]
            
            # Warmup
            for _ in range(10):
                env.step(actions)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_steps):
                env.step(actions)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_steps
            results[n_agents] = {
                'avg_step_time': avg_time,
                'steps_per_second': 1.0 / avg_time
            }
        
        self.results['environment_step'] = results
        return results
    
    def benchmark_communication(self, num_agents: List[int], num_messages: int = 10000):
        """Benchmark communication performance"""
        results = {}
        
        for n_agents in num_agents:
            config = {'num_agents': n_agents}
            mas = MultiAgentSystem(config)
            
            # Create test message
            message = {
                'type': 'test',
                'data': np.random.randn(100).tolist()
            }
            
            # Benchmark broadcast
            start_time = time.time()
            for _ in range(num_messages):
                mas.agents[0].send_message(message, broadcast=True)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_messages
            results[n_agents] = {
                'avg_broadcast_time': avg_time,
                'messages_per_second': 1.0 / avg_time
            }
        
        self.results['communication'] = results
        return results
    
    def generate_report(self) -> str:
        """Generate performance report"""
        report = "=== PI-HMARL Performance Benchmark Report ===\n\n"
        
        for category, results in self.results.items():
            report += f"{category.upper()}:\n"
            for config, metrics in results.items():
                report += f"  Config: {config}\n"
                for metric, value in metrics.items():
                    report += f"    {metric}: {value:.6f}\n"
            report += "\n"
        
        return report


class TestValidator:
    """Validates system requirements and configurations"""
    
    def __init__(self):
        """Initialize validator"""
        self.validation_results = {}
    
    def validate_dependencies(self) -> bool:
        """Validate required dependencies"""
        required_packages = [
            'torch', 'numpy', 'gymnasium', 'networkx',
            'matplotlib', 'pyyaml', 'tensorboard'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        self.validation_results['dependencies'] = {
            'missing_packages': missing,
            'valid': len(missing) == 0
        }
        
        return len(missing) == 0
    
    def validate_gpu_availability(self) -> bool:
        """Check GPU availability"""
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        self.validation_results['gpu'] = {
            'available': gpu_available,
            'count': gpu_count,
            'device_name': torch.cuda.get_device_name(0) if gpu_available else None
        }
        
        return gpu_available
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters"""
        required_fields = [
            'num_agents', 'state_dim', 'action_dim',
            'learning_rate', 'batch_size'
        ]
        
        missing_fields = [f for f in required_fields if f not in config]
        
        # Type validation
        type_errors = []
        if 'num_agents' in config and not isinstance(config['num_agents'], int):
            type_errors.append('num_agents must be int')
        if 'learning_rate' in config and not isinstance(config['learning_rate'], (int, float)):
            type_errors.append('learning_rate must be float')
        
        self.validation_results['configuration'] = {
            'missing_fields': missing_fields,
            'type_errors': type_errors,
            'valid': len(missing_fields) == 0 and len(type_errors) == 0
        }
        
        return self.validation_results['configuration']['valid']
    
    def generate_report(self) -> str:
        """Generate validation report"""
        report = "=== PI-HMARL System Validation Report ===\n\n"
        
        for category, results in self.validation_results.items():
            report += f"{category.upper()}:\n"
            for key, value in results.items():
                report += f"  {key}: {value}\n"
            report += "\n"
        
        return report


def run_all_tests():
    """Run all tests and generate report"""
    # Setup logging
    setup_logging(log_level='INFO')
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestCore))
    test_suite.addTest(unittest.makeSuite(TestModels))
    test_suite.addTest(unittest.makeSuite(TestEnvironments))
    test_suite.addTest(unittest.makeSuite(TestTraining))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    test_results = runner.run(test_suite)
    
    # Run validation
    validator = TestValidator()
    validator.validate_dependencies()
    validator.validate_gpu_availability()
    validator.validate_configuration({
        'num_agents': 4,
        'state_dim': 12,
        'action_dim': 4,
        'learning_rate': 1e-3,
        'batch_size': 32
    })
    
    # Run benchmarks
    benchmark = PerformanceBenchmark()
    benchmark.benchmark_model_inference([1, 4, 8, 16])
    benchmark.benchmark_environment_step([2, 4, 8])
    benchmark.benchmark_communication([2, 4, 8, 16])
    
    # Generate reports
    print("\n" + validator.generate_report())
    print("\n" + benchmark.generate_report())
    
    # Summary
    print("\n=== TEST SUMMARY ===")
    print(f"Tests run: {test_results.testsRun}")
    print(f"Failures: {len(test_results.failures)}")
    print(f"Errors: {len(test_results.errors)}")
    print(f"Success: {test_results.wasSuccessful()}")
    
    return test_results.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)