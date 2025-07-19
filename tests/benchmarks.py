"""Benchmark Suite for PI-HMARL System

This module provides comprehensive benchmarking tools for
performance evaluation and comparison.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import psutil
import GPUtil
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    name: str
    description: str
    iterations: int = 100
    warmup: int = 10
    timeout: float = 300.0
    save_results: bool = True
    output_dir: Path = Path("benchmark_results")
    device: str = "cpu"
    multi_gpu: bool = False
    profile: bool = False


@dataclass
class BenchmarkResult:
    """Benchmark execution result"""
    benchmark_name: str
    config: BenchmarkConfig
    metrics: Dict[str, Any]
    timings: List[float]
    resource_usage: Dict[str, Any]
    timestamp: float
    hardware_info: Dict[str, Any]


class BenchmarkSuite:
    """Main benchmark suite for PI-HMARL"""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark suite
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.benchmarks = []
        self.results = []
        
        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get hardware info
        self.hardware_info = self._get_hardware_info()
        
        logger.info(f"Initialized benchmark suite: {config.name}")
    
    def add_benchmark(self, benchmark_func: callable, name: str, **kwargs):
        """Add benchmark to suite
        
        Args:
            benchmark_func: Benchmark function
            name: Benchmark name
            **kwargs: Additional arguments
        """
        self.benchmarks.append({
            'func': benchmark_func,
            'name': name,
            'kwargs': kwargs
        })
    
    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks
        
        Returns:
            List of benchmark results
        """
        logger.info(f"Running {len(self.benchmarks)} benchmarks")
        
        for benchmark in self.benchmarks:
            result = self._run_benchmark(benchmark)
            self.results.append(result)
        
        # Save results
        if self.config.save_results:
            self._save_results()
        
        # Generate report
        self._generate_report()
        
        return self.results
    
    def _run_benchmark(self, benchmark: Dict) -> BenchmarkResult:
        """Run individual benchmark
        
        Args:
            benchmark: Benchmark specification
            
        Returns:
            Benchmark result
        """
        logger.info(f"Running benchmark: {benchmark['name']}")
        
        # Warmup
        for _ in range(self.config.warmup):
            benchmark['func'](**benchmark['kwargs'])
        
        # Actual runs
        timings = []
        resource_samples = []
        
        for i in range(self.config.iterations):
            # Monitor resources
            resource_start = self._get_resource_usage()
            
            # Time execution
            start_time = time.time()
            metrics = benchmark['func'](**benchmark['kwargs'])
            end_time = time.time()
            
            # Record timing
            timings.append(end_time - start_time)
            
            # Sample resources
            resource_end = self._get_resource_usage()
            resource_samples.append({
                'cpu_delta': resource_end['cpu'] - resource_start['cpu'],
                'memory_delta': resource_end['memory'] - resource_start['memory'],
                'gpu_usage': resource_end.get('gpu_usage', 0)
            })
        
        # Aggregate results
        result = BenchmarkResult(
            benchmark_name=benchmark['name'],
            config=self.config,
            metrics=metrics if isinstance(metrics, dict) else {'result': metrics},
            timings=timings,
            resource_usage=self._aggregate_resources(resource_samples),
            timestamp=time.time(),
            hardware_info=self.hardware_info
        )
        
        return result
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'platform': {
                'system': psutil.os.name,
                'python': psutil.sys.version
            }
        }
        
        # GPU info if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                info['gpu'] = {
                    'name': gpus[0].name,
                    'memory': gpus[0].memoryTotal,
                    'driver': gpus[0].driver
                }
        except:
            pass
        
        return info
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        usage = {
            'cpu': psutil.cpu_percent(interval=0.1),
            'memory': psutil.virtual_memory().percent
        }
        
        # GPU usage if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                usage['gpu_usage'] = gpus[0].load * 100
                usage['gpu_memory'] = gpus[0].memoryUtil * 100
        except:
            pass
        
        return usage
    
    def _aggregate_resources(self, samples: List[Dict]) -> Dict[str, Any]:
        """Aggregate resource usage samples"""
        if not samples:
            return {}
        
        aggregated = {}
        
        for key in samples[0].keys():
            values = [s[key] for s in samples if key in s]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_max'] = np.max(values)
                aggregated[f'{key}_std'] = np.std(values)
        
        return aggregated
    
    def _save_results(self):
        """Save benchmark results"""
        timestamp = int(time.time())
        
        # Save raw results
        results_file = self.config.output_dir / f"benchmark_results_{timestamp}.json"
        
        results_data = []
        for result in self.results:
            results_data.append({
                'name': result.benchmark_name,
                'timings': {
                    'mean': np.mean(result.timings),
                    'std': np.std(result.timings),
                    'min': np.min(result.timings),
                    'max': np.max(result.timings),
                    'p50': np.percentile(result.timings, 50),
                    'p95': np.percentile(result.timings, 95),
                    'p99': np.percentile(result.timings, 99)
                },
                'metrics': result.metrics,
                'resource_usage': result.resource_usage,
                'hardware': result.hardware_info,
                'timestamp': result.timestamp
            })
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Saved results to {results_file}")
    
    def _generate_report(self):
        """Generate benchmark report with visualizations"""
        if not self.results:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Benchmark Results: {self.config.name}', fontsize=16)
        
        # 1. Timing comparison
        ax = axes[0, 0]
        names = [r.benchmark_name for r in self.results]
        means = [np.mean(r.timings) for r in self.results]
        stds = [np.std(r.timings) for r in self.results]
        
        ax.bar(names, means, yerr=stds, capsize=5)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Execution Time Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. Timing distribution
        ax = axes[0, 1]
        for result in self.results[:5]:  # Limit to 5 for clarity
            ax.hist(result.timings, alpha=0.5, label=result.benchmark_name, bins=20)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Count')
        ax.set_title('Timing Distributions')
        ax.legend()
        
        # 3. Resource usage
        ax = axes[1, 0]
        cpu_means = [r.resource_usage.get('cpu_delta_mean', 0) for r in self.results]
        mem_means = [r.resource_usage.get('memory_delta_mean', 0) for r in self.results]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, cpu_means, width, label='CPU %')
        ax.bar(x + width/2, mem_means, width, label='Memory %')
        ax.set_ylabel('Usage %')
        ax.set_title('Resource Usage')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45)
        ax.legend()
        
        # 4. Performance metrics
        ax = axes[1, 1]
        if self.results[0].metrics:
            metric_names = list(self.results[0].metrics.keys())[:3]  # Top 3 metrics
            
            data = []
            for metric in metric_names:
                metric_values = [r.metrics.get(metric, 0) for r in self.results]
                data.append(metric_values)
            
            if data:
                im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
                ax.set_xticks(np.arange(len(names)))
                ax.set_yticks(np.arange(len(metric_names)))
                ax.set_xticklabels(names, rotation=45)
                ax.set_yticklabels(metric_names)
                ax.set_title('Performance Metrics Heatmap')
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        # Save figure
        report_file = self.config.output_dir / f"benchmark_report_{int(time.time())}.png"
        plt.savefig(report_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated report: {report_file}")


class PerformanceBenchmark:
    """Performance-specific benchmarks"""
    
    @staticmethod
    def benchmark_inference_speed(
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Benchmark model inference speed
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            batch_sizes: Batch sizes to test
            device: Device to use
            
        Returns:
            Benchmark metrics
        """
        model = model.to(device)
        model.eval()
        
        results = {}
        
        for batch_size in batch_sizes:
            # Create input
            input_tensor = torch.randn(batch_size, *input_shape).to(device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(input_tensor)
            
            # Time inference
            torch.cuda.synchronize() if device == "cuda" else None
            
            timings = []
            for _ in range(100):
                start = time.time()
                
                with torch.no_grad():
                    output = model(input_tensor)
                
                torch.cuda.synchronize() if device == "cuda" else None
                
                timings.append(time.time() - start)
            
            # Calculate metrics
            avg_time = np.mean(timings)
            throughput = batch_size / avg_time
            
            results[f'batch_{batch_size}'] = {
                'avg_time': avg_time,
                'throughput': throughput,
                'latency_ms': avg_time * 1000 / batch_size
            }
        
        return results
    
    @staticmethod
    def benchmark_training_speed(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        input_shape: Tuple[int, ...],
        batch_size: int = 32,
        steps: int = 100,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Benchmark training speed
        
        Args:
            model: Model to benchmark
            optimizer: Optimizer
            input_shape: Input shape
            batch_size: Batch size
            steps: Number of training steps
            device: Device to use
            
        Returns:
            Benchmark metrics
        """
        model = model.to(device)
        model.train()
        
        # Timing storage
        forward_times = []
        backward_times = []
        optimizer_times = []
        
        for _ in range(steps):
            # Generate batch
            inputs = torch.randn(batch_size, *input_shape).to(device)
            targets = torch.randn(batch_size, model(inputs).shape[-1]).to(device)
            
            # Forward pass
            torch.cuda.synchronize() if device == "cuda" else None
            forward_start = time.time()
            
            outputs = model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)
            
            torch.cuda.synchronize() if device == "cuda" else None
            forward_time = time.time() - forward_start
            forward_times.append(forward_time)
            
            # Backward pass
            backward_start = time.time()
            
            loss.backward()
            
            torch.cuda.synchronize() if device == "cuda" else None
            backward_time = time.time() - backward_start
            backward_times.append(backward_time)
            
            # Optimizer step
            optimizer_start = time.time()
            
            optimizer.step()
            optimizer.zero_grad()
            
            torch.cuda.synchronize() if device == "cuda" else None
            optimizer_time = time.time() - optimizer_start
            optimizer_times.append(optimizer_time)
        
        # Calculate metrics
        total_time = sum(forward_times) + sum(backward_times) + sum(optimizer_times)
        
        return {
            'total_time': total_time,
            'steps_per_second': steps / total_time,
            'forward_time_avg': np.mean(forward_times),
            'backward_time_avg': np.mean(backward_times),
            'optimizer_time_avg': np.mean(optimizer_times),
            'forward_percent': sum(forward_times) / total_time * 100,
            'backward_percent': sum(backward_times) / total_time * 100,
            'optimizer_percent': sum(optimizer_times) / total_time * 100
        }


class ScalabilityBenchmark:
    """Scalability benchmarks"""
    
    @staticmethod
    def benchmark_agent_scaling(
        environment_class: type,
        agent_class: type,
        agent_counts: List[int] = [1, 5, 10, 20, 50, 100],
        episode_length: int = 100
    ) -> Dict[str, Any]:
        """Benchmark system scalability with agent count
        
        Args:
            environment_class: Environment class
            agent_class: Agent class
            agent_counts: Agent counts to test
            episode_length: Episode length
            
        Returns:
            Scalability metrics
        """
        results = {}
        
        for num_agents in agent_counts:
            # Create environment
            env = environment_class(num_agents=num_agents)
            
            # Create agents
            agents = []
            for i in range(num_agents):
                agent = agent_class(
                    agent_id=i,
                    obs_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.shape[0]
                )
                agents.append(agent)
            
            # Run episode
            observations = env.reset()
            
            step_times = []
            agent_times = []
            env_times = []
            
            for _ in range(episode_length):
                # Agent decisions
                agent_start = time.time()
                
                actions = {}
                for i, agent in enumerate(agents):
                    actions[i] = agent.act(observations[i])
                
                agent_time = time.time() - agent_start
                agent_times.append(agent_time)
                
                # Environment step
                env_start = time.time()
                
                observations, rewards, dones, info = env.step(actions)
                
                env_time = time.time() - env_start
                env_times.append(env_time)
                
                # Total step time
                step_times.append(agent_time + env_time)
            
            # Store results
            results[num_agents] = {
                'avg_step_time': np.mean(step_times),
                'avg_agent_time': np.mean(agent_times),
                'avg_env_time': np.mean(env_times),
                'total_time': sum(step_times),
                'agent_time_percent': sum(agent_times) / sum(step_times) * 100,
                'env_time_percent': sum(env_times) / sum(step_times) * 100
            }
        
        # Calculate scaling factors
        base_agents = agent_counts[0]
        base_time = results[base_agents]['avg_step_time']
        
        for num_agents in agent_counts:
            scaling_factor = results[num_agents]['avg_step_time'] / base_time
            ideal_scaling = num_agents / base_agents
            efficiency = ideal_scaling / scaling_factor if scaling_factor > 0 else 0
            
            results[num_agents]['scaling_factor'] = scaling_factor
            results[num_agents]['scaling_efficiency'] = efficiency
        
        return results
    
    @staticmethod
    def benchmark_communication_scaling(
        protocol_class: type,
        network_sizes: List[int] = [2, 5, 10, 20, 50],
        message_count: int = 100,
        message_size: int = 1024
    ) -> Dict[str, Any]:
        """Benchmark communication protocol scaling
        
        Args:
            protocol_class: Protocol class
            network_sizes: Network sizes to test
            message_count: Messages per test
            message_size: Message size in bytes
            
        Returns:
            Communication scaling metrics
        """
        results = {}
        
        for network_size in network_sizes:
            # Create protocol instances
            protocols = []
            for i in range(network_size):
                protocol = protocol_class(agent_id=i)
                protocols.append(protocol)
            
            # Measure communication performance
            send_times = []
            receive_times = []
            
            for _ in range(message_count):
                # Random sender and receiver
                sender_id = np.random.randint(0, network_size)
                receiver_id = np.random.randint(0, network_size)
                
                if sender_id == receiver_id:
                    continue
                
                # Create message
                payload = {'data': np.random.randn(message_size // 8).tolist()}
                
                # Time sending
                send_start = time.time()
                
                message = protocols[sender_id].create_message(
                    msg_type='TEST',
                    receiver_id=receiver_id,
                    payload=payload
                )
                success = protocols[sender_id].send_message(message)
                
                send_time = time.time() - send_start
                send_times.append(send_time)
                
                if success:
                    # Time receiving
                    receive_start = time.time()
                    
                    # Simulate message propagation
                    protocols[receiver_id].receive_message(b'test_data')
                    
                    receive_time = time.time() - receive_start
                    receive_times.append(receive_time)
            
            # Calculate metrics
            results[network_size] = {
                'avg_send_time': np.mean(send_times) if send_times else 0,
                'avg_receive_time': np.mean(receive_times) if receive_times else 0,
                'total_time': sum(send_times) + sum(receive_times),
                'messages_per_second': len(send_times) / (sum(send_times) + sum(receive_times))
                                      if send_times else 0
            }
        
        return results


class EfficiencyBenchmark:
    """Efficiency benchmarks"""
    
    @staticmethod
    def benchmark_energy_efficiency(
        optimizer_class: type,
        mission_profiles: List[Dict[str, Any]],
        num_agents: int = 5
    ) -> Dict[str, Any]:
        """Benchmark energy optimization efficiency
        
        Args:
            optimizer_class: Energy optimizer class
            mission_profiles: Mission profiles to test
            num_agents: Number of agents
            
        Returns:
            Energy efficiency metrics
        """
        results = {}
        
        for profile in mission_profiles:
            profile_name = profile['name']
            
            # Create optimizer
            optimizer = optimizer_class(num_agents=num_agents)
            
            # Generate mission
            agent_states = {}
            for i in range(num_agents):
                agent_states[i] = {
                    'position': torch.randn(3) * 100,
                    'battery_soc': 0.8 + np.random.rand() * 0.2,
                    'capabilities': ['navigation', 'sensing']
                }
            
            tasks = profile['tasks']
            
            # Time optimization
            opt_start = time.time()
            
            optimization_result = optimizer.optimize_mission(agent_states, tasks)
            
            opt_time = time.time() - opt_start
            
            # Calculate efficiency metrics
            total_energy = sum(optimization_result['energy_usage'].values())
            
            # Baseline: naive assignment
            naive_energy = len(tasks) * 10.0  # Simplified baseline
            
            efficiency = naive_energy / total_energy if total_energy > 0 else 1.0
            
            results[profile_name] = {
                'optimization_time': opt_time,
                'total_energy': total_energy,
                'energy_per_task': total_energy / len(tasks),
                'efficiency_vs_naive': efficiency,
                'feasible': optimization_result.get('feasible', True)
            }
        
        return results
    
    @staticmethod
    def benchmark_computational_efficiency(
        algorithm_implementations: Dict[str, callable],
        test_inputs: List[Any],
        memory_profile: bool = True
    ) -> Dict[str, Any]:
        """Benchmark computational efficiency of algorithms
        
        Args:
            algorithm_implementations: Algorithm implementations
            test_inputs: Test inputs
            memory_profile: Whether to profile memory
            
        Returns:
            Computational efficiency metrics
        """
        results = {}
        
        for algo_name, algo_func in algorithm_implementations.items():
            timings = []
            memory_usage = []
            
            for test_input in test_inputs:
                # Memory before
                if memory_profile:
                    process = psutil.Process()
                    mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Time execution
                start = time.time()
                
                output = algo_func(test_input)
                
                exec_time = time.time() - start
                timings.append(exec_time)
                
                # Memory after
                if memory_profile:
                    mem_after = process.memory_info().rss / 1024 / 1024  # MB
                    memory_usage.append(mem_after - mem_before)
            
            # Calculate metrics
            results[algo_name] = {
                'avg_time': np.mean(timings),
                'std_time': np.std(timings),
                'min_time': np.min(timings),
                'max_time': np.max(timings),
                'throughput': len(test_inputs) / sum(timings)
            }
            
            if memory_profile:
                results[algo_name].update({
                    'avg_memory_mb': np.mean(memory_usage),
                    'max_memory_mb': np.max(memory_usage)
                })
        
        # Calculate relative efficiency
        fastest_time = min(r['avg_time'] for r in results.values())
        
        for algo_name in results:
            results[algo_name]['relative_speed'] = fastest_time / results[algo_name]['avg_time']
        
        return results


def create_standard_benchmarks() -> BenchmarkSuite:
    """Create standard benchmark suite for PI-HMARL
    
    Returns:
        Configured benchmark suite
    """
    config = BenchmarkConfig(
        name="PI-HMARL Standard Benchmarks",
        description="Comprehensive performance evaluation",
        iterations=50,
        warmup=5
    )
    
    suite = BenchmarkSuite(config)
    
    # Add benchmarks
    from src.models.hierarchical import HierarchicalPolicy
    from src.environment import MultiAgentEnvironment
    from src.agents.hierarchical_agent import HierarchicalAgent
    from src.communication.protocol import CommunicationProtocol
    from src.energy.energy_optimizer import EnergyOptimizer
    
    # Model inference benchmark
    def model_benchmark():
        model = HierarchicalPolicy(
            obs_dim=64,
            action_dim=4,
            levels=['high', 'mid', 'low']
        )
        return PerformanceBenchmark.benchmark_inference_speed(
            model,
            input_shape=(64,),
            batch_sizes=[1, 8, 32]
        )
    
    suite.add_benchmark(model_benchmark, "model_inference")
    
    # Scalability benchmark
    def scalability_benchmark():
        return ScalabilityBenchmark.benchmark_agent_scaling(
            MultiAgentEnvironment,
            HierarchicalAgent,
            agent_counts=[5, 10, 20],
            episode_length=50
        )
    
    suite.add_benchmark(scalability_benchmark, "agent_scalability")
    
    # Communication benchmark
    def communication_benchmark():
        return ScalabilityBenchmark.benchmark_communication_scaling(
            CommunicationProtocol,
            network_sizes=[5, 10, 20],
            message_count=50
        )
    
    suite.add_benchmark(communication_benchmark, "communication_scaling")
    
    # Energy efficiency benchmark
    def energy_benchmark():
        mission_profiles = [
            {
                'name': 'surveillance',
                'tasks': [
                    {'position': torch.randn(3) * 500, 'duration': 300}
                    for _ in range(5)
                ]
            },
            {
                'name': 'delivery',
                'tasks': [
                    {'position': torch.randn(3) * 1000, 'duration': 120}
                    for _ in range(10)
                ]
            }
        ]
        
        return EfficiencyBenchmark.benchmark_energy_efficiency(
            EnergyOptimizer,
            mission_profiles,
            num_agents=5
        )
    
    suite.add_benchmark(energy_benchmark, "energy_efficiency")
    
    return suite