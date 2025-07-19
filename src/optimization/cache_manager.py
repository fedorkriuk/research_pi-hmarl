"""Cache Management System

This module provides caching utilities for improving performance
by storing and reusing computation results.
"""

import time
import hashlib
import pickle
from typing import Dict, Any, Optional, Callable, Tuple, List
from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np
import torch
import logging
from pathlib import Path
import lmdb
import json
from functools import wraps
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    size: int
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # Time to live in seconds


class CacheManager:
    """Main cache management system"""
    
    def __init__(
        self,
        max_size: int = 1024 * 1024 * 1024,  # 1GB default
        cache_dir: Optional[Path] = None,
        enable_persistence: bool = True
    ):
        """Initialize cache manager
        
        Args:
            max_size: Maximum cache size in bytes
            cache_dir: Directory for persistent cache
            enable_persistence: Enable disk persistence
        """
        self.max_size = max_size
        self.current_size = 0
        self.cache_dir = cache_dir or Path("cache")
        self.enable_persistence = enable_persistence
        
        # In-memory cache
        self.memory_cache: Dict[str, CacheEntry] = {}
        
        # Persistent cache
        self.persistent_cache = None
        if enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._init_persistent_cache()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Initialized CacheManager (max_size: {max_size/1024/1024:.1f}MB)")
    
    def _init_persistent_cache(self):
        """Initialize LMDB persistent cache"""
        try:
            self.persistent_cache = lmdb.open(
                str(self.cache_dir / "persistent_cache"),
                map_size=self.max_size * 2,  # Allow 2x for overhead
                max_dbs=10
            )
        except Exception as e:
            logger.error(f"Failed to initialize persistent cache: {e}")
            self.enable_persistence = False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        with self._lock:
            self.stats['total_requests'] += 1
            
            # Check memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                # Check TTL
                if entry.ttl and time.time() - entry.timestamp > entry.ttl:
                    self._evict(key)
                    self.stats['misses'] += 1
                    return None
                
                # Update access info
                entry.access_count += 1
                entry.last_access = time.time()
                
                self.stats['hits'] += 1
                return entry.value
            
            # Check persistent cache
            if self.enable_persistence:
                value = self._get_from_persistent(key)
                if value is not None:
                    # Promote to memory cache
                    self._put_to_memory(key, value)
                    self.stats['hits'] += 1
                    return value
            
            self.stats['misses'] += 1
            return None
    
    def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> bool:
        """Put value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        with self._lock:
            # Calculate size
            size = self._estimate_size(value)
            
            # Check if we need to evict
            while self.current_size + size > self.max_size:
                if not self._evict_lru():
                    logger.warning("Cache full, cannot add new entry")
                    return False
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                timestamp=time.time(),
                ttl=ttl
            )
            
            # Add to memory cache
            self.memory_cache[key] = entry
            self.current_size += size
            
            # Add to persistent cache
            if self.enable_persistence:
                self._put_to_persistent(key, value)
            
            return True
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes
        
        Args:
            value: Value to estimate
            
        Returns:
            Estimated size in bytes
        """
        if isinstance(value, torch.Tensor):
            return value.element_size() * value.nelement()
        elif isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, (list, dict)):
            # Rough estimate using pickle
            return len(pickle.dumps(value))
        else:
            return 1024  # Default estimate
    
    def _evict_lru(self) -> bool:
        """Evict least recently used entry
        
        Returns:
            Success status
        """
        if not self.memory_cache:
            return False
        
        # Find LRU entry
        lru_key = min(
            self.memory_cache.keys(),
            key=lambda k: self.memory_cache[k].last_access
        )
        
        self._evict(lru_key)
        return True
    
    def _evict(self, key: str):
        """Evict entry from cache
        
        Args:
            key: Key to evict
        """
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            self.current_size -= entry.size
            del self.memory_cache[key]
            self.stats['evictions'] += 1
    
    def _get_from_persistent(self, key: str) -> Optional[Any]:
        """Get value from persistent cache
        
        Args:
            key: Cache key
            
        Returns:
            Value or None
        """
        if not self.persistent_cache:
            return None
        
        try:
            with self.persistent_cache.begin() as txn:
                data = txn.get(key.encode())
                if data:
                    return pickle.loads(data)
        except Exception as e:
            logger.error(f"Error reading from persistent cache: {e}")
        
        return None
    
    def _put_to_persistent(self, key: str, value: Any):
        """Put value to persistent cache
        
        Args:
            key: Cache key
            value: Value to store
        """
        if not self.persistent_cache:
            return
        
        try:
            with self.persistent_cache.begin(write=True) as txn:
                data = pickle.dumps(value)
                txn.put(key.encode(), data)
        except Exception as e:
            logger.error(f"Error writing to persistent cache: {e}")
    
    def _put_to_memory(self, key: str, value: Any):
        """Put value to memory cache
        
        Args:
            key: Cache key
            value: Value to store
        """
        size = self._estimate_size(value)
        entry = CacheEntry(
            key=key,
            value=value,
            size=size,
            timestamp=time.time()
        )
        
        self.memory_cache[key] = entry
        self.current_size += size
    
    def clear(self):
        """Clear all caches"""
        with self._lock:
            self.memory_cache.clear()
            self.current_size = 0
            
            if self.persistent_cache:
                with self.persistent_cache.begin(write=True) as txn:
                    txn.drop(txn.cursor().db)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics
        
        Returns:
            Cache statistics
        """
        with self._lock:
            hit_rate = self.stats['hits'] / self.stats['total_requests'] if self.stats['total_requests'] > 0 else 0
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'current_size_mb': self.current_size / 1024 / 1024,
                'max_size_mb': self.max_size / 1024 / 1024,
                'num_entries': len(self.memory_cache),
                'utilization': self.current_size / self.max_size
            }


class LRUCache:
    """Least Recently Used cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        """Initialize LRU cache
        
        Args:
            max_size: Maximum number of entries
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any):
        """Put value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            if key in self.cache:
                # Update and move to end
                self.cache.move_to_end(key)
            else:
                # Add new entry
                if len(self.cache) >= self.max_size:
                    # Remove oldest
                    self.cache.popitem(last=False)
            
            self.cache[key] = value


class ComputationCache:
    """Cache for expensive computations"""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize computation cache
        
        Args:
            cache_manager: Cache manager instance
        """
        self.cache_manager = cache_manager or CacheManager()
    
    def cached(
        self,
        ttl: Optional[float] = None,
        key_func: Optional[Callable] = None
    ):
        """Decorator for caching function results
        
        Args:
            ttl: Time to live for cache entries
            key_func: Custom key generation function
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_key(func.__name__, args, kwargs)
                
                # Check cache
                result = self.cache_manager.get(cache_key)
                if result is not None:
                    return result
                
                # Compute result
                result = func(*args, **kwargs)
                
                # Cache result
                self.cache_manager.put(cache_key, result, ttl=ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function call
        
        Args:
            func_name: Function name
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Cache key
        """
        # Create hashable representation
        key_parts = [func_name]
        
        # Add args
        for arg in args:
            if isinstance(arg, (torch.Tensor, np.ndarray)):
                # Use shape and dtype for arrays
                key_parts.append(f"{arg.shape}_{arg.dtype}")
            elif isinstance(arg, (list, dict)):
                # Use hash of serialized data
                key_parts.append(hashlib.md5(
                    json.dumps(arg, sort_keys=True).encode()
                ).hexdigest()[:8])
            else:
                key_parts.append(str(arg))
        
        # Add kwargs
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        return "_".join(key_parts)


class ResultCache:
    """Cache for operation results with dependencies"""
    
    def __init__(self):
        """Initialize result cache"""
        self.cache = {}
        self.dependencies = {}
        self._lock = threading.RLock()
    
    def cache_with_dependencies(
        self,
        key: str,
        value: Any,
        dependencies: List[str]
    ):
        """Cache value with dependencies
        
        Args:
            key: Cache key
            value: Value to cache
            dependencies: List of dependency keys
        """
        with self._lock:
            self.cache[key] = value
            self.dependencies[key] = set(dependencies)
            
            # Update reverse dependencies
            for dep in dependencies:
                if dep not in self.dependencies:
                    self.dependencies[dep] = set()
    
    def invalidate(self, key: str):
        """Invalidate cache entry and dependents
        
        Args:
            key: Key to invalidate
        """
        with self._lock:
            # Find all dependent keys
            to_invalidate = {key}
            queue = [key]
            
            while queue:
                current = queue.pop(0)
                
                # Find keys that depend on current
                for k, deps in self.dependencies.items():
                    if current in deps and k not in to_invalidate:
                        to_invalidate.add(k)
                        queue.append(k)
            
            # Remove all invalidated entries
            for k in to_invalidate:
                if k in self.cache:
                    del self.cache[k]
                if k in self.dependencies:
                    del self.dependencies[k]
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        with self._lock:
            return self.cache.get(key)


# Example usage patterns
class CachedOperations:
    """Example of cached operations"""
    
    def __init__(self):
        self.cache = ComputationCache()
    
    @property
    def cached_distance_matrix(self):
        """Cached distance matrix computation"""
        @self.cache.cached(ttl=60.0)  # Cache for 1 minute
        def compute_distance_matrix(positions: np.ndarray) -> np.ndarray:
            n = len(positions)
            distances = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances[i, j] = dist
                    distances[j, i] = dist
            
            return distances
        
        return compute_distance_matrix
    
    @property
    def cached_pathfinding(self):
        """Cached pathfinding results"""
        @self.cache.cached(ttl=30.0)
        def find_path(start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
            # Expensive pathfinding algorithm
            # (simplified for example)
            return [start, goal]
        
        return find_path