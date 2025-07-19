"""Performance Profiling Tools

This module provides comprehensive profiling capabilities for identifying
performance bottlenecks and optimization opportunities.
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from functools import wraps
import cProfile
import pstats
import io
import tracemalloc
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import threading
import GPUtil

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Profiling result data"""
    function_name: str
    execution_time: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    call_count: int = 1
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """Comprehensive performance profiler"""
    
    def __init__(
        self,
        output_dir: Path = Path("profiling_results"),
        track_gpu: bool = True,
        track_memory: bool = True,
        sampling_interval: float = 0.1
    ):
        """Initialize performance profiler
        
        Args:
            output_dir: Directory for profiling results
            track_gpu: Track GPU metrics
            track_memory: Track memory usage
            sampling_interval: Sampling interval for continuous monitoring
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.track_gpu = track_gpu and torch.cuda.is_available()
        self.track_memory = track_memory
        self.sampling_interval = sampling_interval
        
        # Profile storage
        self.profiles: Dict[str, List[ProfileResult]] = defaultdict(list)
        self.active_profiles: Dict[str, Any] = {}
        
        # Continuous monitoring
        self._monitoring = False
        self._monitor_thread = None
        self._monitor_data = []
        
        logger.info(f"Initialized PerformanceProfiler (GPU: {self.track_gpu})")
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a function
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._profile_execution(func.__name__, func, args, kwargs)
        
        return wrapper
    
    def profile_context(self, name: str):
        """Context manager for profiling code blocks
        
        Args:
            name: Name for the profiled section
        """
        class ProfileContext:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
                self.start_time = None
                self.start_cpu = None
                self.start_memory = None
                self.start_gpu = None
            
            def __enter__(self):
                self.start_time = time.time()
                self.start_cpu = psutil.cpu_percent(interval=0)
                
                if self.profiler.track_memory:
                    process = psutil.Process()
                    self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                if self.profiler.track_gpu:
                    torch.cuda.synchronize()
                    self.start_gpu = self.profiler._get_gpu_stats()
                
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Calculate metrics
                execution_time = time.time() - self.start_time
                cpu_usage = psutil.cpu_percent(interval=0) - self.start_cpu
                
                memory_usage = 0
                if self.profiler.track_memory:
                    process = psutil.Process()
                    memory_usage = process.memory_info().rss / 1024 / 1024 - self.start_memory
                
                gpu_usage = None
                gpu_memory = None
                if self.profiler.track_gpu:
                    torch.cuda.synchronize()
                    gpu_stats = self.profiler._get_gpu_stats()
                    if gpu_stats and self.start_gpu:
                        gpu_usage = gpu_stats['usage'] - self.start_gpu['usage']
                        gpu_memory = gpu_stats['memory'] - self.start_gpu['memory']
                
                # Store result
                result = ProfileResult(
                    function_name=self.name,
                    execution_time=execution_time,
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    gpu_usage=gpu_usage,
                    gpu_memory=gpu_memory
                )
                
                self.profiler.profiles[self.name].append(result)
        
        return ProfileContext(self, name)
    
    def _profile_execution(
        self,
        name: str,
        func: Callable,
        args: tuple,
        kwargs: dict
    ) -> Any:
        """Profile function execution
        
        Args:
            name: Function name
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        # Start metrics
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=0)
        
        start_memory = 0
        if self.track_memory:
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024
        
        start_gpu = None
        if self.track_gpu:
            torch.cuda.synchronize()
            start_gpu = self._get_gpu_stats()
        
        # Execute function
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in profiled function {name}: {e}")
            raise
        
        # End metrics
        execution_time = time.time() - start_time
        cpu_usage = psutil.cpu_percent(interval=0) - start_cpu
        
        memory_usage = 0
        if self.track_memory:
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024 - start_memory
        
        gpu_usage = None
        gpu_memory = None
        if self.track_gpu:
            torch.cuda.synchronize()
            gpu_stats = self._get_gpu_stats()
            if gpu_stats and start_gpu:
                gpu_usage = gpu_stats['usage'] - start_gpu['usage']
                gpu_memory = gpu_stats['memory'] - start_gpu['memory']
        
        # Store profile
        profile_result = ProfileResult(
            function_name=name,
            execution_time=execution_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory
        )
        
        self.profiles[name].append(profile_result)
        
        return result
    
    def _get_gpu_stats(self) -> Optional[Dict[str, float]]:
        """Get current GPU statistics
        
        Returns:
            GPU stats or None
        """
        if not self.track_gpu:
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'usage': gpu.load * 100,
                    'memory': gpu.memoryUsed,
                    'temperature': gpu.temperature
                }
        except:
            pass
        
        return None
    
    def start_continuous_monitoring(self):
        """Start continuous performance monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        logger.info("Started continuous performance monitoring")
    
    def stop_continuous_monitoring(self):
        """Stop continuous performance monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        
        logger.info("Stopped continuous performance monitoring")
    
    def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self._monitoring:
            # Collect metrics
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=0),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_mb': psutil.virtual_memory().used / 1024 / 1024
            }
            
            if self.track_gpu:
                gpu_stats = self._get_gpu_stats()
                if gpu_stats:
                    metrics.update({
                        'gpu_usage': gpu_stats['usage'],
                        'gpu_memory': gpu_stats['memory'],
                        'gpu_temperature': gpu_stats['temperature']
                    })
            
            self._monitor_data.append(metrics)
            time.sleep(self.sampling_interval)
    
    def get_summary(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """Get profiling summary
        
        Args:
            function_name: Specific function to summarize (None for all)
            
        Returns:
            Summary statistics
        """
        if function_name:
            profiles = self.profiles.get(function_name, [])
            functions = [function_name]
        else:
            profiles = [p for profile_list in self.profiles.values() for p in profile_list]
            functions = list(self.profiles.keys())
        
        if not profiles:
            return {}
        
        summary = {
            'total_calls': len(profiles),
            'total_time': sum(p.execution_time for p in profiles),
            'functions': {}
        }
        
        for func in functions:
            func_profiles = self.profiles.get(func, [])
            if func_profiles:
                exec_times = [p.execution_time for p in func_profiles]
                summary['functions'][func] = {
                    'calls': len(func_profiles),
                    'total_time': sum(exec_times),
                    'avg_time': np.mean(exec_times),
                    'min_time': np.min(exec_times),
                    'max_time': np.max(exec_times),
                    'std_time': np.std(exec_times)
                }
                
                if self.track_memory:
                    memory_usage = [p.memory_usage for p in func_profiles]
                    summary['functions'][func]['avg_memory_mb'] = np.mean(memory_usage)
                
                if self.track_gpu:
                    gpu_usage = [p.gpu_usage for p in func_profiles if p.gpu_usage is not None]
                    if gpu_usage:
                        summary['functions'][func]['avg_gpu_usage'] = np.mean(gpu_usage)
        
        return summary
    
    def generate_report(self, output_file: Optional[Path] = None):
        """Generate comprehensive profiling report
        
        Args:
            output_file: Output file path (auto-generated if None)
        """
        if not output_file:
            output_file = self.output_dir / f"profile_report_{int(time.time())}.html"
        
        # Get summary
        summary = self.get_summary()
        
        # Generate plots
        self._generate_plots()
        
        # Create HTML report
        html_content = self._generate_html_report(summary)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated profiling report: {output_file}")
    
    def _generate_plots(self):
        """Generate profiling visualization plots"""
        # 1. Execution time by function
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Function execution times
        ax = axes[0, 0]
        functions = []
        avg_times = []
        
        for func, profiles in self.profiles.items():
            if profiles:
                functions.append(func)
                avg_times.append(np.mean([p.execution_time for p in profiles]))
        
        if functions:
            ax.bar(functions, avg_times)
            ax.set_xlabel('Function')
            ax.set_ylabel('Average Time (s)')
            ax.set_title('Average Execution Time by Function')
            ax.tick_params(axis='x', rotation=45)
        
        # 2. Memory usage over time
        ax = axes[0, 1]
        if self._monitor_data:
            timestamps = [d['timestamp'] for d in self._monitor_data]
            memory = [d['memory_mb'] for d in self._monitor_data]
            
            ax.plot(timestamps, memory)
            ax.set_xlabel('Time')
            ax.set_ylabel('Memory (MB)')
            ax.set_title('Memory Usage Over Time')
        
        # 3. CPU usage distribution
        ax = axes[1, 0]
        cpu_usage = []
        for profiles in self.profiles.values():
            cpu_usage.extend([p.cpu_usage for p in profiles])
        
        if cpu_usage:
            ax.hist(cpu_usage, bins=20, alpha=0.7)
            ax.set_xlabel('CPU Usage (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('CPU Usage Distribution')
        
        # 4. GPU usage (if available)
        ax = axes[1, 1]
        if self.track_gpu and self._monitor_data:
            gpu_data = [d for d in self._monitor_data if 'gpu_usage' in d]
            if gpu_data:
                timestamps = [d['timestamp'] for d in gpu_data]
                gpu_usage = [d['gpu_usage'] for d in gpu_data]
                
                ax.plot(timestamps, gpu_usage)
                ax.set_xlabel('Time')
                ax.set_ylabel('GPU Usage (%)')
                ax.set_title('GPU Usage Over Time')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'profile_plots.png', dpi=150)
        plt.close()
    
    def _generate_html_report(self, summary: Dict[str, Any]) -> str:
        """Generate HTML report content
        
        Args:
            summary: Profiling summary
            
        Returns:
            HTML content
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PI-HMARL Performance Profile Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; color: #0066cc; }}
            </style>
        </head>
        <body>
            <h1>Performance Profile Report</h1>
            <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <p>Total Calls: <span class="metric">{summary.get('total_calls', 0)}</span></p>
            <p>Total Time: <span class="metric">{summary.get('total_time', 0):.3f}s</span></p>
            
            <h2>Function Details</h2>
            <table>
                <tr>
                    <th>Function</th>
                    <th>Calls</th>
                    <th>Total Time (s)</th>
                    <th>Avg Time (s)</th>
                    <th>Min Time (s)</th>
                    <th>Max Time (s)</th>
        """
        
        if self.track_memory:
            html += "<th>Avg Memory (MB)</th>"
        if self.track_gpu:
            html += "<th>Avg GPU Usage (%)</th>"
        
        html += "</tr>"
        
        # Add function rows
        for func, stats in summary.get('functions', {}).items():
            html += f"""
                <tr>
                    <td>{func}</td>
                    <td>{stats['calls']}</td>
                    <td>{stats['total_time']:.3f}</td>
                    <td>{stats['avg_time']:.3f}</td>
                    <td>{stats['min_time']:.3f}</td>
                    <td>{stats['max_time']:.3f}</td>
            """
            
            if self.track_memory:
                html += f"<td>{stats.get('avg_memory_mb', 0):.1f}</td>"
            if self.track_gpu:
                html += f"<td>{stats.get('avg_gpu_usage', 0):.1f}</td>"
            
            html += "</tr>"
        
        html += """
            </table>
            
            <h2>Performance Plots</h2>
            <img src="profile_plots.png" alt="Performance Plots" style="max-width: 100%;">
            
        </body>
        </html>
        """
        
        return html


class GPUProfiler:
    """Specialized GPU profiler using PyTorch profiler"""
    
    def __init__(self, wait: int = 1, warmup: int = 1, active: int = 5):
        """Initialize GPU profiler
        
        Args:
            wait: Number of steps to wait
            warmup: Number of warmup steps
            active: Number of active profiling steps
        """
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.profiler = None
        
    def start(self):
        """Start GPU profiling"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, GPU profiling disabled")
            return
        
        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./gpu_profiles'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        
        self.profiler.start()
        logger.info("Started GPU profiling")
    
    def step(self):
        """Step the profiler"""
        if self.profiler:
            self.profiler.step()
    
    def stop(self):
        """Stop GPU profiling"""
        if self.profiler:
            self.profiler.stop()
            logger.info("Stopped GPU profiling")


class MemoryProfiler:
    """Detailed memory profiler"""
    
    def __init__(self):
        """Initialize memory profiler"""
        self.snapshots = []
        self.tracking = False
        
    def start(self):
        """Start memory profiling"""
        tracemalloc.start()
        self.tracking = True
        self.initial_snapshot = tracemalloc.take_snapshot()
        logger.info("Started memory profiling")
    
    def take_snapshot(self, label: str):
        """Take memory snapshot
        
        Args:
            label: Snapshot label
        """
        if not self.tracking:
            return
        
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append({
            'label': label,
            'snapshot': snapshot,
            'timestamp': time.time()
        })
    
    def get_top_allocations(self, limit: int = 10) -> List[str]:
        """Get top memory allocations
        
        Args:
            limit: Number of top allocations
            
        Returns:
            List of allocation descriptions
        """
        if not self.snapshots:
            return []
        
        latest = self.snapshots[-1]['snapshot']
        top_stats = latest.compare_to(self.initial_snapshot, 'lineno')
        
        results = []
        for stat in top_stats[:limit]:
            results.append(f"{stat.traceback}: {stat.size_diff / 1024 / 1024:.1f} MB")
        
        return results
    
    def stop(self):
        """Stop memory profiling"""
        tracemalloc.stop()
        self.tracking = False
        logger.info("Stopped memory profiling")


# Decorator functions
def profile_function(profiler: PerformanceProfiler):
    """Decorator factory for function profiling
    
    Args:
        profiler: Performance profiler instance
        
    Returns:
        Decorator function
    """
    def decorator(func):
        return profiler.profile_function(func)
    return decorator


def profile_method(profiler: PerformanceProfiler):
    """Decorator factory for method profiling
    
    Args:
        profiler: Performance profiler instance
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return profiler._profile_execution(
                f"{self.__class__.__name__}.{func.__name__}",
                func,
                (self,) + args,
                kwargs
            )
        return wrapper
    return decorator