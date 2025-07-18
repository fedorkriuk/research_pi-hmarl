"""
GPU Detection and CUDA Setup Utilities for PI-HMARL

This module provides utilities for detecting GPU availability, setting up CUDA,
optimizing GPU memory usage, and managing multi-GPU configurations.
"""

import os
import sys
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import platform
from dataclasses import dataclass
from contextlib import contextmanager
import psutil


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    id: int
    name: str
    memory_total: int  # MB
    memory_free: int   # MB
    memory_used: int   # MB
    utilization: float  # %
    temperature: Optional[float] = None
    power_draw: Optional[float] = None
    compute_capability: Optional[Tuple[int, int]] = None


class GPUManager:
    """
    Comprehensive GPU management and CUDA setup utilities.
    
    Features:
    - GPU detection and validation
    - CUDA setup and configuration
    - Memory optimization
    - Multi-GPU support
    - Performance monitoring
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize GPU manager.
        
        Args:
            logger: Logger instance for GPU operations
        """
        self.logger = logger or logging.getLogger(__name__)
        self.torch_available = False
        self.cuda_available = False
        self.gpu_count = 0
        self.gpu_info: List[GPUInfo] = []
        
        # Initialize GPU detection
        self._detect_gpu_libraries()
        self._detect_cuda()
        self._detect_gpus()
        
        # Log GPU status
        self._log_gpu_status()
    
    def _detect_gpu_libraries(self) -> None:
        """Detect available GPU libraries."""
        # Check PyTorch availability
        try:
            import torch
            self.torch = torch
            self.torch_available = True
            self.logger.info(f"PyTorch version: {torch.__version__}")
        except ImportError:
            self.torch = None
            self.logger.warning("PyTorch not available")
        
        # Check other GPU libraries
        try:
            import cupy
            self.cupy_available = True
            self.logger.info(f"CuPy version: {cupy.__version__}")
        except ImportError:
            self.cupy_available = False
        
        try:
            import tensorflow as tf
            self.tensorflow_available = True
            self.logger.info(f"TensorFlow version: {tf.__version__}")
        except ImportError:
            self.tensorflow_available = False
    
    def _detect_cuda(self) -> None:
        """Detect CUDA availability and version."""
        if not self.torch_available:
            self.logger.warning("Cannot detect CUDA without PyTorch")
            return
        
        try:
            self.cuda_available = self.torch.cuda.is_available()
            
            if self.cuda_available:
                self.cuda_version = self.torch.version.cuda
                self.cudnn_version = self.torch.backends.cudnn.version()
                self.gpu_count = self.torch.cuda.device_count()
                
                self.logger.info(f"CUDA available: {self.cuda_available}")
                self.logger.info(f"CUDA version: {self.cuda_version}")
                self.logger.info(f"cuDNN version: {self.cudnn_version}")
                self.logger.info(f"GPU count: {self.gpu_count}")
            else:
                self.logger.warning("CUDA not available")
                
        except Exception as e:
            self.logger.error(f"Error detecting CUDA: {e}")
            self.cuda_available = False
    
    def _detect_gpus(self) -> None:
        """Detect individual GPU information."""
        if not self.cuda_available:
            return
        
        try:
            for i in range(self.gpu_count):
                gpu_info = self._get_gpu_info(i)
                self.gpu_info.append(gpu_info)
                
        except Exception as e:
            self.logger.error(f"Error detecting GPU information: {e}")
    
    def _get_gpu_info(self, gpu_id: int) -> GPUInfo:
        """Get detailed information about a specific GPU."""
        if not self.cuda_available:
            raise RuntimeError("CUDA not available")
        
        try:
            # Get GPU properties
            props = self.torch.cuda.get_device_properties(gpu_id)
            
            # Get memory information
            memory_total = props.total_memory // (1024 * 1024)  # Convert to MB
            memory_free = self.torch.cuda.memory_reserved(gpu_id) // (1024 * 1024)
            memory_used = self.torch.cuda.memory_allocated(gpu_id) // (1024 * 1024)
            
            # Get utilization (requires nvidia-ml-py)
            utilization = self._get_gpu_utilization(gpu_id)
            
            # Get temperature and power (requires nvidia-ml-py)
            temperature, power_draw = self._get_gpu_thermal_info(gpu_id)
            
            return GPUInfo(
                id=gpu_id,
                name=props.name,
                memory_total=memory_total,
                memory_free=memory_free,
                memory_used=memory_used,
                utilization=utilization,
                temperature=temperature,
                power_draw=power_draw,
                compute_capability=(props.major, props.minor)
            )
            
        except Exception as e:
            self.logger.error(f"Error getting GPU {gpu_id} info: {e}")
            return GPUInfo(
                id=gpu_id,
                name="Unknown",
                memory_total=0,
                memory_free=0,
                memory_used=0,
                utilization=0.0
            )
    
    def _get_gpu_utilization(self, gpu_id: int) -> float:
        """Get GPU utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(utilization.gpu)
        except ImportError:
            self.logger.debug("pynvml not available for GPU utilization")
            return 0.0
        except Exception as e:
            self.logger.debug(f"Error getting GPU utilization: {e}")
            return 0.0
    
    def _get_gpu_thermal_info(self, gpu_id: int) -> Tuple[Optional[float], Optional[float]]:
        """Get GPU temperature and power draw."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            # Get temperature
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Get power draw
            power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            
            return float(temperature), float(power_draw)
            
        except ImportError:
            self.logger.debug("pynvml not available for thermal info")
            return None, None
        except Exception as e:
            self.logger.debug(f"Error getting GPU thermal info: {e}")
            return None, None
    
    def _log_gpu_status(self) -> None:
        """Log comprehensive GPU status."""
        self.logger.info("=== GPU Status ===")
        self.logger.info(f"PyTorch available: {self.torch_available}")
        self.logger.info(f"CUDA available: {self.cuda_available}")
        
        if self.cuda_available:
            self.logger.info(f"CUDA version: {self.cuda_version}")
            self.logger.info(f"GPU count: {self.gpu_count}")
            
            for i, gpu in enumerate(self.gpu_info):
                self.logger.info(f"GPU {i}: {gpu.name}")
                self.logger.info(f"  Memory: {gpu.memory_used}/{gpu.memory_total} MB")
                self.logger.info(f"  Utilization: {gpu.utilization:.1f}%")
                if gpu.temperature:
                    self.logger.info(f"  Temperature: {gpu.temperature:.1f}°C")
                if gpu.power_draw:
                    self.logger.info(f"  Power: {gpu.power_draw:.1f}W")
                if gpu.compute_capability:
                    self.logger.info(f"  Compute Capability: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
        
        self.logger.info("==================")
    
    def get_optimal_device(self, memory_required: Optional[int] = None) -> str:
        """
        Get the optimal device for computation.
        
        Args:
            memory_required: Required memory in MB
            
        Returns:
            Device string (e.g., 'cuda:0', 'cpu')
        """
        if not self.cuda_available:
            self.logger.info("Using CPU (CUDA not available)")
            return "cpu"
        
        if self.gpu_count == 0:
            self.logger.info("Using CPU (no GPUs available)")
            return "cpu"
        
        # Find best GPU based on available memory
        best_gpu = 0
        best_free_memory = 0
        
        for i, gpu in enumerate(self.gpu_info):
            free_memory = gpu.memory_total - gpu.memory_used
            
            if memory_required and free_memory < memory_required:
                continue
            
            if free_memory > best_free_memory:
                best_free_memory = free_memory
                best_gpu = i
        
        if memory_required and best_free_memory < memory_required:
            self.logger.warning(f"Insufficient GPU memory. Required: {memory_required}MB, Available: {best_free_memory}MB")
            return "cpu"
        
        device = f"cuda:{best_gpu}"
        self.logger.info(f"Using device: {device} (Free memory: {best_free_memory}MB)")
        return device
    
    def get_available_devices(self) -> List[str]:
        """
        Get list of available devices.
        
        Returns:
            List of device strings
        """
        devices = ["cpu"]
        
        if self.cuda_available:
            devices.extend([f"cuda:{i}" for i in range(self.gpu_count)])
        
        return devices
    
    def set_device(self, device: str) -> None:
        """
        Set the current device for PyTorch operations.
        
        Args:
            device: Device string (e.g., 'cuda:0', 'cpu')
        """
        if not self.torch_available:
            self.logger.warning("PyTorch not available, cannot set device")
            return
        
        try:
            if device.startswith("cuda") and not self.cuda_available:
                self.logger.warning(f"CUDA not available, falling back to CPU")
                device = "cpu"
            
            self.torch.cuda.set_device(device)
            self.logger.info(f"Set device to: {device}")
            
        except Exception as e:
            self.logger.error(f"Error setting device to {device}: {e}")
            raise
    
    def optimize_memory(self, enable_memory_fraction: bool = True, memory_fraction: float = 0.9) -> None:
        """
        Optimize GPU memory usage.
        
        Args:
            enable_memory_fraction: Whether to limit memory usage
            memory_fraction: Fraction of GPU memory to use
        """
        if not self.cuda_available:
            self.logger.warning("CUDA not available, cannot optimize memory")
            return
        
        try:
            # Set memory fraction for each GPU
            if enable_memory_fraction:
                for i in range(self.gpu_count):
                    total_memory = self.gpu_info[i].memory_total * 1024 * 1024  # Convert to bytes
                    memory_limit = int(total_memory * memory_fraction)
                    
                    # PyTorch doesn't have direct memory fraction setting
                    # but we can use CUDA_VISIBLE_DEVICES or set memory limit
                    self.logger.info(f"GPU {i}: Setting memory limit to {memory_limit // (1024*1024)}MB")
            
            # Enable memory caching optimization
            if hasattr(self.torch.cuda, 'empty_cache'):
                self.torch.cuda.empty_cache()
                self.logger.info("Cleared CUDA cache")
            
            # Set memory allocation strategy
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
        except Exception as e:
            self.logger.error(f"Error optimizing memory: {e}")
    
    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        if not self.cuda_available:
            return
        
        try:
            if hasattr(self.torch.cuda, 'empty_cache'):
                self.torch.cuda.empty_cache()
                self.logger.info("Cleared CUDA memory cache")
        except Exception as e:
            self.logger.error(f"Error clearing memory: {e}")
    
    def monitor_memory(self) -> Dict[str, Any]:
        """
        Monitor current GPU memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.cuda_available:
            return {"cuda_available": False}
        
        memory_stats = {"cuda_available": True, "gpus": []}
        
        try:
            for i in range(self.gpu_count):
                allocated = self.torch.cuda.memory_allocated(i)
                reserved = self.torch.cuda.memory_reserved(i)
                total = self.torch.cuda.get_device_properties(i).total_memory
                
                gpu_stats = {
                    "id": i,
                    "allocated_mb": allocated // (1024 * 1024),
                    "reserved_mb": reserved // (1024 * 1024),
                    "total_mb": total // (1024 * 1024),
                    "utilization_percent": (allocated / total) * 100
                }
                
                memory_stats["gpus"].append(gpu_stats)
            
        except Exception as e:
            self.logger.error(f"Error monitoring memory: {e}")
        
        return memory_stats
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dictionary with system information
        """
        info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total // (1024**3),
            "cuda_available": self.cuda_available,
            "gpu_count": self.gpu_count,
        }
        
        if self.torch_available:
            info["pytorch_version"] = self.torch.__version__
            info["pytorch_cuda_version"] = self.torch.version.cuda
        
        if self.cuda_available:
            info["gpus"] = [
                {
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_total_mb": gpu.memory_total,
                    "compute_capability": gpu.compute_capability
                }
                for gpu in self.gpu_info
            ]
        
        return info
    
    @contextmanager
    def device_context(self, device: str):
        """Context manager for temporary device switching."""
        if not self.torch_available:
            yield
            return
        
        original_device = self.torch.cuda.current_device() if self.cuda_available else None
        
        try:
            self.set_device(device)
            yield
        finally:
            if original_device is not None:
                self.set_device(f"cuda:{original_device}")
    
    def benchmark_device(self, device: str, size: Tuple[int, ...] = (1000, 1000)) -> Dict[str, float]:
        """
        Benchmark device performance.
        
        Args:
            device: Device to benchmark
            size: Tensor size for benchmarking
            
        Returns:
            Dictionary with benchmark results
        """
        if not self.torch_available:
            return {"error": "PyTorch not available"}
        
        try:
            import time
            
            # Create test tensors
            a = self.torch.randn(size, device=device)
            b = self.torch.randn(size, device=device)
            
            # Warmup
            for _ in range(10):
                c = self.torch.matmul(a, b)
            
            # Benchmark matrix multiplication
            start_time = time.time()
            for _ in range(100):
                c = self.torch.matmul(a, b)
            
            if device.startswith("cuda"):
                self.torch.cuda.synchronize()
            
            end_time = time.time()
            
            matmul_time = (end_time - start_time) / 100
            
            # Benchmark memory transfer (if GPU)
            transfer_time = 0.0
            if device.startswith("cuda"):
                cpu_tensor = self.torch.randn(size)
                
                start_time = time.time()
                gpu_tensor = cpu_tensor.to(device)
                self.torch.cuda.synchronize()
                end_time = time.time()
                
                transfer_time = end_time - start_time
            
            return {
                "device": device,
                "matmul_time_ms": matmul_time * 1000,
                "transfer_time_ms": transfer_time * 1000,
                "tensor_size": size
            }
            
        except Exception as e:
            return {"error": str(e)}


# Global GPU manager instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


def setup_gpu(device: str = "auto", optimize_memory: bool = True) -> str:
    """
    Setup GPU for PI-HMARL experiments.
    
    Args:
        device: Device to use ('auto', 'cpu', 'cuda:0', etc.)
        optimize_memory: Whether to optimize memory usage
        
    Returns:
        Selected device string
    """
    gpu_manager = get_gpu_manager()
    
    if device == "auto":
        device = gpu_manager.get_optimal_device()
    
    gpu_manager.set_device(device)
    
    if optimize_memory:
        gpu_manager.optimize_memory()
    
    return device


def get_device_info() -> Dict[str, Any]:
    """Get device information."""
    return get_gpu_manager().get_system_info()


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    get_gpu_manager().clear_memory()


def monitor_gpu_memory() -> Dict[str, Any]:
    """Monitor GPU memory usage."""
    return get_gpu_manager().monitor_memory()