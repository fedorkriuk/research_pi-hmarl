"""Test setup and installation for PI-HMARL"""

import pytest
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_python_version():
    """Test Python version is 3.8 or higher"""
    assert sys.version_info >= (3, 8), "Python 3.8 or higher required"


def test_pytorch_installation():
    """Test PyTorch is properly installed"""
    # Test basic PyTorch functionality
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = torch.matmul(x, y)
    assert z.shape == (3, 3)


def test_cuda_availability():
    """Test CUDA availability (warning if not available)"""
    if torch.cuda.is_available():
        # Test CUDA functionality
        device = torch.device("cuda:0")
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)
        z = torch.matmul(x, y)
        assert z.device.type == "cuda"
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ Warning: CUDA not available. GPU acceleration disabled.")


def test_config_manager():
    """Test ConfigManager functionality"""
    from utils.config_manager import ConfigManager
    
    # Test default config loading
    config = ConfigManager()
    assert config.get("training.algorithm") == "PI-HMARL"
    assert config.get("training.num_agents") == 10
    assert config.get("physics.engine") == "pybullet"
    
    # Test config updates
    config.update_config({"training": {"num_agents": 20}})
    assert config.get("training.num_agents") == 20
    
    # Test validation
    assert config.validate_config()


def test_logger():
    """Test Logger functionality"""
    from utils.logger import PIHMARLLogger, get_logger
    
    # Test logger creation
    logger = get_logger(name="test_logger", log_dir="./test_logs")
    
    # Test logging functions
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test metrics logging
    logger.log_metrics({"loss": 0.5, "reward": 100}, step=1)
    
    # Cleanup
    logger.close()
    
    # Clean up test logs
    import shutil
    if Path("./test_logs").exists():
        shutil.rmtree("./test_logs")


def test_gpu_utils():
    """Test GPU utilities"""
    from utils.gpu_utils import GPUManager, auto_select_device
    
    # Test GPU manager
    gpu_manager = GPUManager()
    
    # Test device selection
    device = auto_select_device()
    assert device.type in ["cuda", "cpu"]
    
    # Test memory summary
    if torch.cuda.is_available():
        memory_stats = gpu_manager.get_memory_summary()
        assert "total_gb" in memory_stats
        assert "free_gb" in memory_stats


def test_directory_structure():
    """Test that all required directories exist"""
    required_dirs = [
        "src/utils",
        "src/data", 
        "src/environment",
        "src/physics",
        "src/agents",
        "configs",
        "tests",
        "data",
        "experiments",
        "docs"
    ]
    
    project_root = Path(__file__).parent.parent
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        assert full_path.exists(), f"Required directory missing: {dir_path}"


def test_requirements():
    """Test that key packages are installed"""
    required_packages = [
        "torch",
        "gymnasium",
        "ray",
        "pybullet",
        "numpy",
        "scipy",
        "matplotlib",
        "wandb",
        "h5py",
        "yaml",
        "omegaconf",
        "tqdm"
    ]
    
    import importlib
    
    for package in required_packages:
        try:
            if package == "yaml":
                importlib.import_module("yaml")
            else:
                importlib.import_module(package)
        except ImportError:
            pytest.fail(f"Required package not installed: {package}")


def test_real_parameters():
    """Test real parameter configuration"""
    from utils.config_manager import ConfigManager
    
    config = ConfigManager()
    
    # Test real physics parameters are loaded
    real_params = config.get("physics.real_parameters")
    assert real_params is not None
    assert real_params.get("drone_mass") == 0.895  # DJI Mavic 3 mass
    assert real_params.get("max_flight_speed") == 19.0  # m/s
    assert real_params.get("battery_capacity") == 5000  # mAh


if __name__ == "__main__":
    # Run basic tests
    print("Running PI-HMARL setup tests...")
    
    test_python_version()
    print("✓ Python version check passed")
    
    test_pytorch_installation()
    print("✓ PyTorch installation check passed")
    
    test_cuda_availability()
    
    test_directory_structure()
    print("✓ Directory structure check passed")
    
    test_config_manager()
    print("✓ ConfigManager check passed")
    
    test_logger()
    print("✓ Logger check passed")
    
    test_gpu_utils()
    print("✓ GPU utilities check passed")
    
    test_real_parameters()
    print("✓ Real parameters check passed")
    
    print("\n✅ All setup tests passed\!")
EOF < /dev/null