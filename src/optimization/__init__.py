"""Performance Optimization Module

This module provides tools for optimizing and profiling the PI-HMARL system.
"""

from .profiler import (
    PerformanceProfiler, GPUProfiler, MemoryProfiler,
    profile_function, profile_method
)
from .optimizer import (
    ModelOptimizer, ComputationOptimizer,
    CommunicationOptimizer, MemoryOptimizer
)
from .accelerator import (
    GPUAccelerator, ParallelProcessor,
    VectorizedOperations, JITCompiler
)
from .cache_manager import (
    CacheManager, LRUCache, ComputationCache,
    ResultCache
)

__all__ = [
    # Profiling
    'PerformanceProfiler', 'GPUProfiler', 'MemoryProfiler',
    'profile_function', 'profile_method',
    
    # Optimization
    'ModelOptimizer', 'ComputationOptimizer',
    'CommunicationOptimizer', 'MemoryOptimizer',
    
    # Acceleration
    'GPUAccelerator', 'ParallelProcessor',
    'VectorizedOperations', 'JITCompiler',
    
    # Caching
    'CacheManager', 'LRUCache', 'ComputationCache',
    'ResultCache'
]