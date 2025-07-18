"""GPU utilities for PI-HMARL Framework

This module provides utilities for GPU detection, memory management,
and CUDA setup for optimal performance in multi-agent training.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import nvidia_ml_py as nvml
import psutil
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Container for GPU information"""
    index: int
    name: str
    total_memory: int  # in bytes
    free_memory: int   # in bytes
    used_memory: int   # in bytes
    temperature: float  # in Celsius
    utilization: float  # percentage
    power_draw: float   # in Watts
    power_limit: float  # in Watts
    compute_capability: Tuple[int, int]


class GPUManager:
    """Manages GPU resources for PI-HMARL training"""
    
    def __init__(self):
        """Initialize GPU Manager"""
        self.cuda_available = torch.cuda.is_available()
        self.device_count = 0
        self.devices = []
        
        if self.cuda_available:
            self.device_count = torch.cuda.device_count()
            
            # Initialize NVML for detailed GPU info
            try:
                nvml.nvmlInit()
                self.nvml_available = True
            except Exception as e:
                logger.warning(f"NVML initialization failed: {e}")
                self.nvml_available = False
            
            self._detect_devices()
        else:
            logger.warning("CUDA not available. CPU will be used.")
    
    def _detect_devices(self):
        """Detect and store information about available GPUs"""
        self.devices = []
        
        for i in range(self.device_count):
            device_info = self._get_device_info(i)
            self.devices.append(device_info)
            
            logger.info(
                f"GPU {i}: {device_info.name} - "
                f"Memory: {device_info.free_memory / 1e9:.1f}GB free / "
                f"{device_info.total_memory / 1e9:.1f}GB total"
            )
    
    def _get_device_info(self, device_index: int) -> GPUInfo:
        """Get detailed information about a specific GPU
        
        Args:
            device_index: GPU device index
            
        Returns:
            GPUInfo object with device details
        """
        # Basic PyTorch info
        torch.cuda.set_device(device_index)
        name = torch.cuda.get_device_name(device_index)
        total_memory = torch.cuda.get_device_properties(device_index).total_memory
        
        # Memory info
        allocated = torch.cuda.memory_allocated(device_index)
        reserved = torch.cuda.memory_reserved(device_index)
        free_memory = total_memory - reserved
        
        # Default values
        temperature = 0.0
        utilization = 0.0
        power_draw = 0.0
        power_limit = 0.0
        compute_capability = (0, 0)
        
        # Detailed NVML info if available
        if self.nvml_available:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
                
                # Temperature
                temperature = nvml.nvmlDeviceGetTemperature(
                    handle, nvml.NVML_TEMPERATURE_GPU
                )
                
                # Utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util.gpu
                
                # Power
                power_draw = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                power_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                
                # Compute capability
                major, minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability = (major, minor)
                
            except Exception as e:
                logger.debug(f"Error getting NVML info for GPU {device_index}: {e}")
        
        return GPUInfo(
            index=device_index,
            name=name,
            total_memory=total_memory,
            free_memory=free_memory,
            used_memory=allocated,
            temperature=temperature,
            utilization=utilization,
            power_draw=power_draw,
            power_limit=power_limit,
            compute_capability=compute_capability
        )
    
    def get_optimal_device(self, min_memory_gb: float = 2.0) -> Optional[int]:
        """Get the optimal GPU device based on available memory
        
        Args:
            min_memory_gb: Minimum required free memory in GB
            
        Returns:
            Device index or None if no suitable device found
        """
        if not self.cuda_available:
            return None
        
        # Update device info
        self._detect_devices()
        
        # Find devices with enough memory
        suitable_devices = [
            d for d in self.devices
            if d.free_memory >= min_memory_gb * 1e9
        ]
        
        if not suitable_devices:
            logger.warning(f"No GPU with {min_memory_gb}GB free memory available")
            return None
        
        # Sort by free memory (descending) and return the best
        suitable_devices.sort(key=lambda d: d.free_memory, reverse=True)
        best_device = suitable_devices[0]
        
        logger.info(f"Selected GPU {best_device.index}: {best_device.name}")
        return best_device.index
    
    def set_device(self, device_index: Optional[int] = None) -> torch.device:
        """Set the active CUDA device
        
        Args:
            device_index: GPU index. If None, uses optimal device.
            
        Returns:
            torch.device object
        """
        if not self.cuda_available:
            device = torch.device("cpu")
            logger.info("Using CPU device")
            return device
        
        if device_index is None:
            device_index = self.get_optimal_device()
        
        if device_index is None:
            device = torch.device("cpu")
            logger.warning("No suitable GPU found. Using CPU.")
        else:
            torch.cuda.set_device(device_index)
            device = torch.device(f"cuda:{device_index}")
            logger.info(f"Using GPU {device_index}")
        
        return device
    
    def get_memory_summary(self, device_index: int = None) -> Dict[str, float]:
        """Get memory usage summary for a device
        
        Args:
            device_index: GPU index. If None, uses current device.
            
        Returns:
            Dictionary with memory statistics in GB
        """
        if not self.cuda_available:
            return {"error": "CUDA not available"}
        
        if device_index is None:
            device_index = torch.cuda.current_device()
        
        stats = {
            "allocated_gb": torch.cuda.memory_allocated(device_index) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(device_index) / 1e9,
            "free_gb": (
                torch.cuda.get_device_properties(device_index).total_memory -
                torch.cuda.memory_reserved(device_index)
            ) / 1e9,
            "total_gb": torch.cuda.get_device_properties(device_index).total_memory / 1e9
        }
        
        return stats
    
    def optimize_batch_size(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        initial_batch_size: int = 32,
        safety_factor: float = 0.9
    ) -> int:
        """Find optimal batch size for a model
        
        Args:
            model: PyTorch model
            input_shape: Shape of single input (without batch dimension)
            initial_batch_size: Starting batch size for search
            safety_factor: Memory safety factor (0-1)
            
        Returns:
            Optimal batch size
        """
        if not self.cuda_available:
            return initial_batch_size
        
        device = next(model.parameters()).device
        
        # Binary search for optimal batch size
        min_batch = 1
        max_batch = initial_batch_size * 4
        optimal_batch = initial_batch_size
        
        while min_batch <= max_batch:
            batch_size = (min_batch + max_batch) // 2
            
            try:
                # Clear cache
                torch.cuda.empty_cache()
                
                # Try forward pass
                dummy_input = torch.randn(batch_size, *input_shape).to(device)
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # Check memory usage
                memory_used = torch.cuda.memory_allocated(device)
                memory_total = torch.cuda.get_device_properties(device).total_memory
                
                if memory_used < memory_total * safety_factor:
                    optimal_batch = batch_size
                    min_batch = batch_size + 1
                else:
                    max_batch = batch_size - 1
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    max_batch = batch_size - 1
                else:
                    raise e
            finally:
                torch.cuda.empty_cache()
        
        logger.info(f"Optimal batch size: {optimal_batch}")
        return optimal_batch
    
    def enable_mixed_precision(self) -> bool:
        """Check and enable mixed precision training if supported
        
        Returns:
            True if mixed precision is supported
        """
        if not self.cuda_available:
            return False
        
        # Check compute capability
        device_info = self._get_device_info(torch.cuda.current_device())
        major, minor = device_info.compute_capability
        
        # Mixed precision requires compute capability >= 7.0
        if major >= 7:
            logger.info(
                f"Mixed precision training enabled "
                f"(Compute capability {major}.{minor})"
            )
            return True
        else:
            logger.warning(
                f"Mixed precision not supported "
                f"(Compute capability {major}.{minor} < 7.0)"
            )
            return False
    
    def cleanup(self):
        """Cleanup GPU resources"""
        if self.cuda_available:
            torch.cuda.empty_cache()
            
        if self.nvml_available:
            try:
                nvml.nvmlShutdown()
            except:
                pass


# Global GPU manager instance
_gpu_manager = None


def get_gpu_manager() -> GPUManager:
    """Get global GPU manager instance
    
    Returns:
        GPUManager instance
    """
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


def auto_select_device(min_memory_gb: float = 2.0) -> torch.device:
    """Automatically select best available device
    
    Args:
        min_memory_gb: Minimum required GPU memory in GB
        
    Returns:
        torch.device object
    """
    manager = get_gpu_manager()
    return manager.set_device()


def get_device_info() -> List[GPUInfo]:
    """Get information about all available GPUs
    
    Returns:
        List of GPUInfo objects
    """
    manager = get_gpu_manager()
    return manager.devices


def log_gpu_memory():
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            logger.info(
                f"GPU {i} memory: {allocated:.2f}GB allocated, "
                f"{reserved:.2f}GB reserved"
            )


def clear_gpu_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
EOF < /dev/null