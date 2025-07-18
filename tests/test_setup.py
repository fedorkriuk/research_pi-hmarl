"""
Test Suite for PI-HMARL Setup and Configuration

This test suite validates the setup and configuration of the PI-HMARL environment,
including utilities, configuration management, logging, and GPU detection.
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config_manager import ConfigManager, Config, ConfigurationError
from src.utils.logger import PIHMARLLogger, get_logger, setup_logging
from src.utils.gpu_utils import GPUManager, GPUInfo, get_gpu_manager


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
        
        # Create a test config file
        test_config = {
            "experiment": {
                "name": "test_experiment",
                "seed": 42
            },
            "training": {
                "num_episodes": 100,
                "learning_rate": 0.001,
                "batch_size": 32
            },
            "device": {
                "type": "cpu"
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
    
    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.temp_dir)
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        config_manager = ConfigManager(self.config_file)
        assert config_manager is not None
        assert config_manager.config_path == self.config_file
    
    def test_config_loading(self):
        """Test configuration loading from file."""
        config_manager = ConfigManager(self.config_file)
        
        assert config_manager.get("experiment.name") == "test_experiment"
        assert config_manager.get("experiment.seed") == 42
        assert config_manager.get("training.num_episodes") == 100
        assert config_manager.get("training.learning_rate") == 0.001
    
    def test_config_get_with_default(self):
        """Test getting configuration with default values."""
        config_manager = ConfigManager(self.config_file)
        
        # Test existing key
        assert config_manager.get("experiment.name") == "test_experiment"
        
        # Test non-existing key with default
        assert config_manager.get("non.existing.key", "default_value") == "default_value"
        
        # Test non-existing key without default
        assert config_manager.get("non.existing.key") is None
    
    def test_config_set(self):
        """Test setting configuration values."""
        config_manager = ConfigManager(self.config_file)
        
        # Set new value
        config_manager.set("new.key", "new_value")
        assert config_manager.get("new.key") == "new_value"
        
        # Update existing value
        config_manager.set("experiment.name", "updated_experiment")
        assert config_manager.get("experiment.name") == "updated_experiment"
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = ConfigManager(self.config_file)
        
        # Valid configuration should pass
        assert config_manager.validate() is True
        
        # Invalid configuration should fail
        config_manager.set("learning_rate", -1)  # Invalid negative learning rate
        assert config_manager.validate() is False
    
    def test_config_save(self):
        """Test saving configuration to file."""
        config_manager = ConfigManager(self.config_file)
        
        # Modify configuration
        config_manager.set("experiment.name", "saved_experiment")
        
        # Save to new file
        new_config_file = Path(self.temp_dir) / "saved_config.yaml"
        config_manager.save(new_config_file)
        
        # Verify file was created and contains correct data
        assert new_config_file.exists()
        
        # Load from saved file and verify
        new_config_manager = ConfigManager(new_config_file)
        assert new_config_manager.get("experiment.name") == "saved_experiment"
    
    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config_manager = ConfigManager(self.config_file)
        config_dict = config_manager.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "experiment" in config_dict
        assert "training" in config_dict
        assert config_dict["experiment"]["name"] == "test_experiment"
    
    def test_config_env_overrides(self):
        """Test environment variable overrides."""
        # Set environment variable
        os.environ["PIHMARL_EXPERIMENT__NAME"] = "env_experiment"
        os.environ["PIHMARL_TRAINING__BATCH_SIZE"] = "64"
        
        try:
            config_manager = ConfigManager(self.config_file)
            
            # Environment variables should override config file values
            assert config_manager.get("experiment.name") == "env_experiment"
            assert config_manager.get("training.batch_size") == 64
            
        finally:
            # Clean up environment variables
            os.environ.pop("PIHMARL_EXPERIMENT__NAME", None)
            os.environ.pop("PIHMARL_TRAINING__BATCH_SIZE", None)
    
    def test_config_nonexistent_file(self):
        """Test handling of non-existent configuration file."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError):
            ConfigManager(nonexistent_file)
    
    def test_config_invalid_yaml(self):
        """Test handling of invalid YAML file."""
        invalid_yaml_file = Path(self.temp_dir) / "invalid.yaml"
        
        with open(invalid_yaml_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(ConfigurationError):
            ConfigManager(invalid_yaml_file)


class TestPIHMARLLogger:
    """Test cases for PIHMARLLogger class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.temp_dir)
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = PIHMARLLogger(
            name="test_logger",
            log_dir=self.log_dir,
            experiment_id="test_exp_001"
        )
        
        assert logger is not None
        assert logger.name == "test_logger"
        assert logger.experiment_id == "test_exp_001"
        assert logger.log_dir == self.log_dir
    
    def test_logger_basic_logging(self):
        """Test basic logging functionality."""
        logger = PIHMARLLogger(
            name="test_logger",
            log_dir=self.log_dir,
            experiment_id="test_exp_002"
        )
        
        # Test different log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Verify log files were created
        log_files = list(self.log_dir.glob("*.log"))
        assert len(log_files) > 0
    
    def test_logger_context_setting(self):
        """Test setting and clearing context."""
        logger = PIHMARLLogger(
            name="test_logger",
            log_dir=self.log_dir,
            experiment_id="test_exp_003"
        )
        
        # Set context
        logger.set_context(episode=1, step=10)
        logger.info("Message with context")
        
        # Clear context
        logger.clear_context()
        logger.info("Message without context")
        
        # Context should be properly managed
        assert True  # Basic test that no exceptions were raised
    
    def test_logger_metrics_logging(self):
        """Test metrics logging functionality."""
        logger = PIHMARLLogger(
            name="test_logger",
            log_dir=self.log_dir,
            experiment_id="test_exp_004"
        )
        
        # Log metrics
        metrics = {
            "reward": 100.0,
            "loss": 0.5,
            "accuracy": 0.95
        }
        
        logger.log_metrics(metrics, step=1)
        
        # Verify metrics were logged
        assert True  # Basic test that no exceptions were raised
    
    def test_logger_timer_functionality(self):
        """Test timer functionality."""
        logger = PIHMARLLogger(
            name="test_logger",
            log_dir=self.log_dir,
            experiment_id="test_exp_005"
        )
        
        # Test timer context manager
        with logger.timer("test_operation"):
            # Simulate some work
            import time
            time.sleep(0.001)
        
        # Test manual timer
        logger.start_timer("manual_timer")
        time.sleep(0.001)
        elapsed = logger.stop_timer("manual_timer")
        
        assert elapsed > 0
    
    def test_logger_counter_functionality(self):
        """Test counter functionality."""
        logger = PIHMARLLogger(
            name="test_logger",
            log_dir=self.log_dir,
            experiment_id="test_exp_006"
        )
        
        # Test counter increment
        logger.increment_counter("test_counter", 5)
        logger.increment_counter("test_counter", 3)
        
        # Get metrics
        metrics = logger.get_metrics()
        assert "counters" in metrics
        assert "test_counter" in metrics["counters"]
        assert metrics["counters"]["test_counter"] == 8
    
    def test_logger_exception_handling(self):
        """Test exception logging."""
        logger = PIHMARLLogger(
            name="test_logger",
            log_dir=self.log_dir,
            experiment_id="test_exp_007"
        )
        
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.log_exception(e, context={"test": "context"})
        
        # Verify exception was logged
        assert True  # Basic test that no exceptions were raised
    
    def test_get_logger_function(self):
        """Test global logger function."""
        logger = get_logger(
            name="global_test",
            log_level="INFO",
            experiment_id="global_exp_001"
        )
        
        assert logger is not None
        assert isinstance(logger, PIHMARLLogger)
    
    def test_setup_logging_function(self):
        """Test setup logging function."""
        logger = setup_logging(
            log_level="DEBUG",
            log_dir=self.log_dir,
            experiment_id="setup_exp_001"
        )
        
        assert logger is not None
        assert isinstance(logger, PIHMARLLogger)


class TestGPUManager:
    """Test cases for GPUManager class."""
    
    def test_gpu_manager_initialization(self):
        """Test GPUManager initialization."""
        gpu_manager = GPUManager()
        
        assert gpu_manager is not None
        assert hasattr(gpu_manager, 'torch_available')
        assert hasattr(gpu_manager, 'cuda_available')
        assert hasattr(gpu_manager, 'gpu_count')
    
    def test_get_optimal_device(self):
        """Test getting optimal device."""
        gpu_manager = GPUManager()
        
        device = gpu_manager.get_optimal_device()
        assert device in ["cpu"] or device.startswith("cuda:")
    
    def test_get_available_devices(self):
        """Test getting available devices."""
        gpu_manager = GPUManager()
        
        devices = gpu_manager.get_available_devices()
        assert isinstance(devices, list)
        assert "cpu" in devices
    
    def test_get_system_info(self):
        """Test getting system information."""
        gpu_manager = GPUManager()
        
        info = gpu_manager.get_system_info()
        assert isinstance(info, dict)
        assert "platform" in info
        assert "python_version" in info
        assert "cuda_available" in info
    
    def test_memory_monitoring(self):
        """Test memory monitoring functionality."""
        gpu_manager = GPUManager()
        
        memory_stats = gpu_manager.monitor_memory()
        assert isinstance(memory_stats, dict)
        assert "cuda_available" in memory_stats
    
    @patch('torch.cuda.is_available')
    def test_cuda_not_available(self, mock_cuda_available):
        """Test behavior when CUDA is not available."""
        mock_cuda_available.return_value = False
        
        gpu_manager = GPUManager()
        
        assert gpu_manager.cuda_available is False
        assert gpu_manager.get_optimal_device() == "cpu"
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_cuda_available(self, mock_device_count, mock_cuda_available):
        """Test behavior when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2
        
        gpu_manager = GPUManager()
        
        devices = gpu_manager.get_available_devices()
        assert "cuda:0" in devices
        assert "cuda:1" in devices
    
    def test_device_context_manager(self):
        """Test device context manager."""
        gpu_manager = GPUManager()
        
        # Test context manager doesn't raise exceptions
        with gpu_manager.device_context("cpu"):
            assert True
    
    def test_benchmark_device(self):
        """Test device benchmarking."""
        gpu_manager = GPUManager()
        
        # Test CPU benchmarking
        if gpu_manager.torch_available:
            benchmark_result = gpu_manager.benchmark_device("cpu", size=(100, 100))
            assert isinstance(benchmark_result, dict)
            assert "device" in benchmark_result
    
    def test_get_gpu_manager_function(self):
        """Test global GPU manager function."""
        gpu_manager = get_gpu_manager()
        
        assert gpu_manager is not None
        assert isinstance(gpu_manager, GPUManager)
    
    def test_gpu_info_dataclass(self):
        """Test GPUInfo dataclass."""
        gpu_info = GPUInfo(
            id=0,
            name="Test GPU",
            memory_total=8192,
            memory_free=4096,
            memory_used=4096,
            utilization=50.0
        )
        
        assert gpu_info.id == 0
        assert gpu_info.name == "Test GPU"
        assert gpu_info.memory_total == 8192
        assert gpu_info.utilization == 50.0


class TestIntegration:
    """Integration tests for the complete setup."""
    
    def test_imports(self):
        """Test that all required modules can be imported."""
        try:
            import torch
            import numpy as np
            import yaml
            import matplotlib.pyplot as plt
            import gymnasium as gym
            
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import required module: {e}")
    
    def test_basic_functionality(self):
        """Test basic functionality of the system."""
        # Test config manager
        config_manager = ConfigManager()
        assert config_manager is not None
        
        # Test logger
        logger = get_logger(name="integration_test")
        assert logger is not None
        
        # Test GPU manager
        gpu_manager = get_gpu_manager()
        assert gpu_manager is not None
        
        # Test that they work together
        logger.info("Integration test successful")
        device = gpu_manager.get_optimal_device()
        config_manager.set("device.type", device)
        
        assert config_manager.get("device.type") == device
    
    def test_pytorch_functionality(self):
        """Test PyTorch functionality."""
        try:
            import torch
            
            # Test tensor creation
            tensor = torch.randn(10, 10)
            assert tensor.shape == (10, 10)
            
            # Test basic operations
            result = torch.matmul(tensor, tensor.T)
            assert result.shape == (10, 10)
            
            # Test device placement
            gpu_manager = get_gpu_manager()
            device = gpu_manager.get_optimal_device()
            
            tensor_on_device = tensor.to(device)
            assert str(tensor_on_device.device).startswith(device.split(":")[0])
            
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_environment_creation(self):
        """Test environment creation."""
        try:
            import gymnasium as gym
            
            # Test basic environment
            env = gym.make("CartPole-v1")
            assert env is not None
            
            # Test environment reset
            observation, info = env.reset()
            assert observation is not None
            
            # Test environment step
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            env.close()
            
        except ImportError:
            pytest.skip("Gymnasium not available")
    
    def test_configuration_integration(self):
        """Test configuration integration with other components."""
        config_manager = ConfigManager()
        logger = get_logger(name="config_integration_test")
        
        # Test configuration-driven logging
        log_level = config_manager.get("logging.level", "INFO")
        assert log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        # Test configuration-driven device selection
        device_type = config_manager.get("device.type", "auto")
        assert device_type in ["auto", "cpu", "cuda", "cuda:0", "cuda:1"]
        
        logger.info(f"Configuration integration test completed with device: {device_type}")


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])